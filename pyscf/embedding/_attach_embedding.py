#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Attach Embedding to SCF
'''

import copy

import numpy as np

from pyscf import lib
from pyscf import scf
# for the side effect of defining gen_response on the SCF classes
from pyscf.scf import _response_functions  # noqa: F401
from pyscf.lib import logger


def _couples_to_the_density(is_uhf, args, kwargs):
    '''Whether a response function's trial density reaches the environment.

    The environment is a functional of the total charge density, so it responds
    exactly where the Coulomb term of the response function does -- compare the
    `vj` branches of `pyscf/scf/_response_functions.py`. Two cases leave it out:

    hermi == 2
        the trial density is anti-hermitian, and the Coulomb term of a real
        anti-hermitian density vanishes. `_B_dot_x` would return zero here of
        its own accord, since the field integrals are symmetric; the point of
        testing for it is to not solve for induced dipoles that are known in
        advance to be zero.
    singlet is False
        the RHF triplet kernel, whose trial density is a spin density: the
        alpha and beta blocks are equal and opposite, the total charge density
        does not change, and the environment sees no perturbation at all.
        Nothing makes this one zero by itself, so including it would be wrong
        rather than merely wasteful.

    The two response signatures agree on where these sit: positional argument 2
    is `singlet` for the RHF-like response and `with_j` for the UHF-like one,
    and positional argument 3 is `hermi` in both.
    '''
    if kwargs.get('hermi', args[3] if len(args) > 3 else 0) == 2:
        return False
    if is_uhf:
        return kwargs.get('with_j', args[2] if len(args) > 2 else True)
    singlet = kwargs.get('singlet', args[2] if len(args) > 2 else None)
    return singlet is None or singlet


def _add_embedding_response(vind, with_embedding, is_uhf, args, kwargs):
    '''Wrap a response function so the environment responds to the trial density.

    `vind` is the response of the isolated molecule; what the environment adds
    is the potential of the dipoles the trial density induces, i.e. `_B_dot_x`.
    It is added only when the environment is taken to relax against the
    first-order density -- `equilibrium_solvation` -- and only when it can: a
    frozen potential is a fixed external term, so its response to any
    first-order density is zero however that flag is set. `_attach_solvent`
    tests only the first of those.

    Both flags are read inside `vind` rather than here, so that a caller may set
    them around the call that uses the response function rather than around the
    call that builds it.
    '''
    if not _couples_to_the_density(is_uhf, args, kwargs):
        return vind

    def vind_with_embedding(dm1):
        v = vind(dm1)
        if with_embedding.equilibrium_solvation and not with_embedding.frozen:
            if is_uhf:
                # the environment sees the sum of the spin blocks, and the one
                # potential it answers with applies to both
                v += with_embedding._B_dot_x(dm1[0] + dm1[1])
            else:
                v += with_embedding._B_dot_x(dm1)
        return v
    return vind_with_embedding


def _is_uhf_response(mf):
    '''Whether mf takes the UHF-style response function.

    UHF and ROHF both do, and it hands `vind` a pair of spin blocks. ROHF is not
    a subclass of UHF, so it has to be named; `_attach_solvent` tests only for
    UHF and gives ROHF a per-spin-block environment response as a result.
    '''
    return isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF))


def _for_scf(mf, embedding_obj, dm=None):
    '''Add embedding to SCF (HF and DFT) method.

    Kwargs:
        dm : if given, the embedding is frozen at this density matrix: the
            potential is computed once here and then held, so the environment
            does not respond to the density during the SCF and enters as a
            fixed external term.
    '''
    if dm is not None:
        # Before the re-attachment branch below, so that handing a density to
        # an already-attached method freezes it too.
        embedding_obj.kernel(dm)
        embedding_obj.frozen = True

    if isinstance(mf, _Embedding):
        mf.with_embedding = embedding_obj
        return mf

    sol_mf = SCFWithEmbedding(mf, embedding_obj)
    name = embedding_obj.__class__.__name__ + mf.__class__.__name__
    return lib.set_class(sol_mf, (SCFWithEmbedding, mf.__class__), name)


def _for_post_scf(method, embedding_obj, dm=None):
    '''Add embedding to a post-SCF method (MP, CI, CC).

    The environment is relaxed against the correlated density: the method is run
    to convergence with the potential held fixed, the potential is rebuilt from
    the density that came out, and the two are cycled to
    embedding_obj.conv_tol. Pass dm to skip that and keep the environment at one
    density throughout, in which case the potential is an external term the
    correlated method never has to know about.

    NOTE: as in pyscf.solvent, the macro iteration can be hard to converge.

    Kwargs:
        dm : if given, the embedding is frozen at this density matrix instead of
            following the correlated one.
    '''
    if isinstance(method, _Embedding):
        # with_embedding here is a read-only view of the SCF object's, so the
        # SCF object is where a new potential has to be put. (_attach_solvent
        # assigns to the property itself first, which cannot work.)
        method._scf.with_embedding = embedding_obj
        return method

    if getattr(method._scf, 'with_embedding', None) is not None:
        scf_with_embedding = method._scf
        embedding_obj = scf_with_embedding.with_embedding
        if dm is not None:
            embedding_obj.kernel(dm)
            embedding_obj.frozen = True
    else:
        scf_with_embedding = _for_scf(method._scf, embedding_obj, dm)
        embedding_obj = scf_with_embedding.with_embedding

    if embedding_obj.v is None:
        # Every cycle of the macro iteration runs the method with the potential
        # held fixed, so there has to be one before the first cycle. The density
        # the SCF object carries is the natural starting point; an SCF that has
        # not been run yet has only its initial guess to offer.
        if scf_with_embedding.mo_coeff is None:
            embedding_obj.kernel(scf_with_embedding.get_init_guess())
        else:
            embedding_obj.kernel(scf_with_embedding.make_rdm1())

    postmf = PostSCFWithEmbedding(method, scf_with_embedding)
    name = embedding_obj.__class__.__name__ + method.__class__.__name__
    return lib.set_class(postmf, (PostSCFWithEmbedding, method.__class__), name)


def _for_casci(mc, embedding_obj, dm=None):
    '''Add embedding to a CASCI method.

    The environment is relaxed against the CI density: the CI problem is solved
    with the potential held fixed, the potential is rebuilt from the density
    that came out, and the two are cycled to embedding_obj.conv_tol. Pass dm to
    keep the environment at one density throughout instead.

    Kwargs:
        dm : if given, the embedding is frozen at this density matrix.
    '''
    if isinstance(mc, _Embedding):
        mc.with_embedding = embedding_obj
        return mc

    if dm is not None:
        embedding_obj.kernel(dm)
        embedding_obj.frozen = True

    embedded = CASCIWithEmbedding(mc, embedding_obj)
    name = embedding_obj.__class__.__name__ + mc.__class__.__name__
    return lib.set_class(embedded, (CASCIWithEmbedding, mc.__class__), name)


def _for_casscf(mc, embedding_obj, dm=None):
    '''Add embedding to a CASSCF method.

    The environment follows the density through the macro iterations rather than
    being cycled around a converged solution, so there is no loop of its own
    here; see CASSCFWithEmbedding.

    Kwargs:
        dm : if given, the embedding is frozen at this density matrix.
    '''
    if isinstance(mc, _Embedding):
        mc.with_embedding = embedding_obj
        return mc

    if dm is not None:
        embedding_obj.kernel(dm)
        embedding_obj.frozen = True

    embedded = CASSCFWithEmbedding(mc, embedding_obj)
    name = embedding_obj.__class__.__name__ + mc.__class__.__name__
    return lib.set_class(embedded, (CASSCFWithEmbedding, mc.__class__), name)


def _for_tdscf(method, embedding_obj=None, dm=None, equilibrium_solvation=False):
    '''Add embedding to a TDSCF method (TDA, TDHF, TDDFT).

    The ground state is already in the orbitals the excitation is built on; what
    this adds is whether the environment relaxes against the *transition*
    density. It does not by default: a vertical excitation is fast next to the
    environment, which stays where the ground state left it.

    Kwargs:
        equilibrium_solvation : let the environment relax against the
            first-order density after all, for a process slow enough that it
            can.
        dm : not accepted, see below.
    '''
    if dm is not None:
        # For the SCF and the correlated methods dm chooses the density the
        # potential is held at. A TDSCF object has no potential of its own to
        # hold -- the ground state potential reaches it through the orbitals of
        # ._scf -- so all dm could do here is set an e and a v that nothing
        # reads. Freezing belongs on the ground state calculation.
        raise NotImplementedError(
            'dm does not apply to a TDSCF method: the potential it sees is the '
            'one the ground state converged with. Pass dm when attaching the '
            'embedding to the SCF method instead.')

    scf_obj = method._scf
    if getattr(scf_obj, 'with_embedding', None) is None:
        raise TypeError(
            'The SCF method underneath carries no embedding potential. Attach '
            'the embedding to it first, so that the excitation is built on '
            'orbitals that know about the environment.')

    if isinstance(method, _Embedding):
        if embedding_obj is None:
            return method
        method = method.copy()
        method.with_embedding = embedding_obj
        return method

    if embedding_obj is None or embedding_obj is scf_obj.with_embedding:
        # A copy, so that the excited state flag stays out of the ground
        # state's environment. It is shallow, and deliberately so: the MM
        # subsystems and the integral driver are the same objects, which is what
        # makes this cost nothing. Nothing on this path writes to them --
        # _B_dot_x solves for perturbed dipoles without storing them, and dm,
        # the only argument that would call kernel, is refused above.
        embedding_obj = scf_obj.with_embedding.copy()
        embedding_obj.equilibrium_solvation = equilibrium_solvation

    if embedding_obj.equilibrium_solvation and embedding_obj.frozen:
        raise RuntimeError(
            'The ground state embedding is frozen, which is a fixed external '
            'potential, and equilibrium_solvation asks the same potential to '
            'relax against the excited state. Set '
            '_scf.with_embedding.frozen = False and re-run the ground state.')

    td = TDSCFWithEmbedding(method, embedding_obj)
    name = embedding_obj.__class__.__name__ + method.__class__.__name__
    return lib.set_class(td, (TDSCFWithEmbedding, method.__class__), name)


class _Embedding:
    pass


class _CASWithEmbedding(_Embedding):
    '''What the CASCI and CASSCF attachments share.

    A multiconfigurational method has no get_veff to hang the potential on, so
    it goes into get_hcore instead. That puts Tr(D v) into every energy the
    method computes, which is not the embedding energy: the correction is made
    where each of them reports its total.
    '''

    _keys = {'with_embedding'}

    def __init__(self, mc, embedding_obj):
        self.__dict__.update(mc.__dict__)
        self.with_embedding = embedding_obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_embedding.check_sanity()
        self.with_embedding.dump_flags(verbose)
        return self

    def reset(self, mol=None):
        self.with_embedding.reset(mol)
        return super().reset(mol)

    def get_hcore(self, mol=None):
        '''The one-electron Hamiltonian, carrying the embedding potential.

        There is no hook for an effective potential in CASCI or CASSCF, so the
        potential is added here. It reaches the one-electron operator of
        everything downstream -- gen_h_op, update_casdm, casci -- and with it
        the core energy, which is why the total energy each of those reports
        holds a contribution that _embedding_energy_correction takes back out.
        '''
        hcore = self._scf.get_hcore(mol)
        if self.with_embedding.v is not None:
            hcore = hcore + self.with_embedding.v
        return hcore

    def _embedding_energy_correction(self, dm):
        '''What to add to a total energy computed with v inside hcore.

        That energy counts the interaction of the density with the potential,
        Tr(D v), where the embedding energy is what belongs in the total. For
        the permanent multipoles the difference is precisely the nuclear part
        of the electrostatic interaction, which no electronic energy could
        carry; for the induced dipoles it is what makes the -1/2 of the
        induction energy come out right.
        '''
        with_embedding = self.with_embedding
        if with_embedding.e is None:
            return 0.0
        return with_embedding.e - np.einsum('ij,ji->', with_embedding.v, dm)

    def to_gpu(self):
        # See SCFWithEmbedding.to_gpu.
        raise NotImplementedError(
            'Embedding has no GPU implementation. Call '
            '.undo_embedding().to_gpu() to move the bare method to the GPU.')

    def nuc_grad_method(self):
        raise NotImplementedError(
            'Nuclear gradients are not implemented for a multiconfigurational '
            'method in an embedding potential. The energy is not variational in '
            'the environment -- the potential is built from the density it acts '
            'on -- so the gradient carries terms the SCF one does not; see the '
            'notes on parity with pyscf.solvent.')

    Gradients = nuc_grad_method


class SCFWithEmbedding(_Embedding):
    _keys = {'with_embedding'}

    def __init__(self, mf, embedding_obj):
        self.__dict__.update(mf.__dict__)
        self.with_embedding = embedding_obj

    def undo_embedding(self):
        '''Revert to the bare SCF method the embedding was attached to.'''
        cls = self.__class__
        name_mixin = self.with_embedding.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, SCFWithEmbedding, name_mixin))
        del obj.with_embedding
        return obj

    def to_gpu(self):
        # Without this, the generic conversion would hand back a GPU method
        # whose embedding potential is silently absent.
        raise NotImplementedError(
            'Embedding has no GPU implementation. Call '
            '.undo_embedding().to_gpu() to move the bare method to the GPU.')

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_embedding.check_sanity()
        self.with_embedding.dump_flags(verbose)
        return self

    def reset(self, mol=None):
        self.with_embedding.reset(mol)
        return super().reset(mol)

    def get_veff(self, mol=None, dm=None, *args, **kwargs):
        vhf = super().get_veff(mol, dm, *args, **kwargs)
        # kernel stores e and v on the embedding object itself
        e_embedding, v_embedding = self.with_embedding.kernel(dm)

        # NOTE: v_embedding should not be added to vhf in this place. This is
        # because vhf is used as the reference for direct_scf in the next
        # iteration. If v_embedding is added here, it may break direct SCF.
        return lib.tag_array(vhf, e_embedding=e_embedding, v_embedding=v_embedding)

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        with_embedding = self.with_embedding
        if with_embedding.e is not None:
            logger.info(self, '\n******** %s Energy Contributions ********',
                        with_embedding.__class__.__name__)
            for label, energy in with_embedding.energy_contributions().items():
                logger.info(self, '%s = %.15g', label, energy)
            logger.info(self, '\n')
        return super()._finalize()

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1,
                 diis=None, diis_start_cycle=None,
                 level_shift_factor=None, damp_factor=None, fock_last=None):
        if dm is None:
            dm = self.make_rdm1()

        # DIIS was called inside super().get_fock. v_embedding, as a function of
        # dm, should be extrapolated as well. To enable it, v_embedding has to be
        # added to the fock matrix before DIIS was called.
        if getattr(vhf, 'v_embedding', None) is None:
            vhf = self.get_veff(self.mol, dm)
        return super().get_fock(h1e, s1e, vhf + vhf.v_embedding, dm, cycle, diis,
                                diis_start_cycle, level_shift_factor, damp_factor,
                                fock_last)

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None:
            dm = self.make_rdm1()
        if getattr(vhf, 'e_embedding', None) is None:
            vhf = self.get_veff(self.mol, dm)
        e_tot, e_coul = super().energy_elec(dm, h1e, vhf)
        e_tot += vhf.e_embedding
        self.scf_summary['e_embedding'] = vhf.e_embedding.real
        logger.debug(self, 'Embedding Energy = %.15g', vhf.e_embedding)
        return e_tot, e_coul

    def gen_response(self, *args, **kwargs):
        '''The response function, with the response of the environment added.

        See _add_embedding_response, which is shared with the TDSCF attachment.
        '''
        return _add_embedding_response(
            self.undo_embedding().gen_response(*args, **kwargs),
            self.with_embedding, _is_uhf_response(self), args, kwargs)

    def stability(self, *args, **kwargs):
        '''Stability analysis, with the environment in the orbital Hessian.

        The analysis diagonalizes the second derivative of the energy, and the
        embedding energy has one: the environment relaxes along with the
        density during the SCF, so its response belongs in the Hessian of the
        solution being tested. That is what `equilibrium_solvation` turns on,
        and it is off by default, so it is set here for the duration.

        A frozen potential contributes nothing, being an external term rather
        than a functional of the density -- `gen_response` already refuses it,
        which leaves the flag below saying what it means rather than doing the
        work.
        '''
        with lib.temporary_env(self.with_embedding,
                               equilibrium_solvation=not self.with_embedding.frozen):
            return super().stability(*args, **kwargs)

    def nuc_grad_method(self):
        grad_method = super().nuc_grad_method()
        return self.with_embedding.nuc_grad_method(grad_method)

    Gradients = nuc_grad_method

    def Hessian(self):
        from pyscf.embedding import embedding_hessian
        return embedding_hessian.make_hess_object(self)

    # The post-SCF methods reachable from an SCF object, each handed the same
    # environment. super().MP2 and super().CCSD may resolve to the density
    # fitted variants, which is why they are not named directly. Each hook is
    # attached to the SCF classes by its own package rather than defined on
    # them, so that package is imported here -- without it these would shadow a
    # missing attribute with a confusing error about super().
    def MP2(self):
        from pyscf import mp  # noqa: F401
        return _for_post_scf(super().MP2(), self.with_embedding)

    def CISD(self):
        from pyscf import ci  # noqa: F401
        return _for_post_scf(super().CISD(), self.with_embedding)

    def CCSD(self):
        from pyscf import cc  # noqa: F401
        return _for_post_scf(super().CCSD(), self.with_embedding)

    def _for_tdscf_method(self, name, equilibrium_solvation):
        '''Build a TDSCF method of the given name and attach the embedding.

        The hooks are attached to the SCF classes by pyscf.tdscf rather than
        defined on them, and some of them are attached as None -- ROHF has no
        TDA or TDHF, RKS no TDHF -- so both cases are checked here rather than
        surfacing as an AttributeError about super() or a call to None.
        '''
        from pyscf import tdscf  # noqa: F401
        factory = getattr(super(), name, None)
        if factory is None:
            raise NotImplementedError(
                '%s is not available for %s.'
                % (name, self.undo_embedding().__class__.__name__))
        return _for_tdscf(factory(), equilibrium_solvation=equilibrium_solvation)

    def TDA(self, equilibrium_solvation=False):
        return self._for_tdscf_method('TDA', equilibrium_solvation)

    def TDHF(self, equilibrium_solvation=False):
        return self._for_tdscf_method('TDHF', equilibrium_solvation)

    def TDDFT(self, equilibrium_solvation=False):
        return self._for_tdscf_method('TDDFT', equilibrium_solvation)

    def CASCI(self, ncas, nelecas, **kwargs):
        from pyscf import mcscf  # noqa: F401
        return _for_casci(super().CASCI(ncas, nelecas, **kwargs),
                          self.with_embedding)

    def CASSCF(self, ncas, nelecas, **kwargs):
        from pyscf import mcscf  # noqa: F401
        return _for_casscf(super().CASSCF(ncas, nelecas, **kwargs),
                           self.with_embedding)


class PostSCFWithEmbedding(_Embedding):
    '''A correlated method whose SCF carries an embedding potential.

    The environment is reached through the SCF object rather than held here, so
    the correlated method inherits it through the orbitals and the Fock matrix
    it is handed. What this class adds is the loop that lets the environment
    respond to the correlated density rather than stopping at the SCF one.
    '''

    def __init__(self, method, scf_with_embedding):
        self.__dict__.update(method.__dict__)
        self._scf = scf_with_embedding
        # The macro iteration below runs the whole method, SCF included, once
        # per cycle. A scanner of the bare method over the embedded SCF is what
        # does that, so it is built once here rather than per cycle.
        self._basic_scanner = method.as_scanner()
        self._basic_scanner._scf = scf_with_embedding.as_scanner()

    def undo_embedding(self):
        '''Revert to the bare correlated method, over a bare SCF.'''
        cls = self.__class__
        name_mixin = self.with_embedding.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, PostSCFWithEmbedding, name_mixin))
        obj._scf = self._scf.undo_embedding()
        del obj._basic_scanner
        return obj

    @property
    def with_embedding(self):
        return self._scf.with_embedding

    def to_gpu(self):
        # See SCFWithEmbedding.to_gpu.
        raise NotImplementedError(
            'Embedding has no GPU implementation. Call '
            '.undo_embedding().to_gpu() to move the bare method to the GPU.')

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_embedding.check_sanity()
        self.with_embedding.dump_flags(verbose)
        return self

    def reset(self, mol=None):
        self.with_embedding.reset(mol)
        return super().reset(mol)

    def _run_once(self):
        '''Run the method, SCF included, with the potential held where it is.

        The scanner is what keeps the two in step: it re-converges the SCF in
        the current potential and hands the correlated method the orbitals that
        came out, rather than whatever orbitals this object was built on. The
        results are copied back, so that what the object holds was computed with
        the potential it holds.
        '''
        basic_scanner = self._basic_scanner
        with lib.temporary_env(self.with_embedding, frozen=True):
            e_tot = basic_scanner(self.mol)
            # _scf first, while it is still ours to update from the scanner's;
            # then everything else, leaving out the two attributes that make
            # this object what it is.
            self._scf.__dict__.update(basic_scanner._scf.__dict__)
            self.__dict__.update(
                {key: value for key, value in basic_scanner.__dict__.items()
                 if key not in ('_scf', '_basic_scanner')})
        return e_tot

    def kernel(self, *args, **kwargs):
        '''The correlated method, with the environment relaxed against it.

        A frozen potential is part of the SCF problem rather than of the
        correlated one, so there is a single pass: the SCF is converged in the
        potential and the correlated method is run on the orbitals that come
        out. Otherwise the environment and the correlated density are brought
        into step by a macro iteration, each cycle running that same pass and
        then rebuilding the potential from the density it produced.

        Either way the SCF is re-converged here rather than taken on trust,
        since the potential this object carries need not be the one the SCF
        underneath was last run with.
        '''
        if args or kwargs:
            raise NotImplementedError(
                'An initial guess cannot be handed to a correlated method in an '
                'embedding potential: the method is run more than once, through '
                'a scanner that takes none.')

        with_embedding = self.with_embedding
        log = logger.new_logger(self)
        if not with_embedding.frozen:
            log.info('\n** Relaxing %s against the %s density **',
                     with_embedding.__class__.__name__, self.__class__.__name__)
            e_last = 0
            de = np.inf
            for cycle in range(with_embedding.max_cycle):
                log.info('\n** Embedding macro cycle %d:', cycle)
                # _run_once holds the potential at the density of the previous
                # cycle while the SCF relaxes against it -- without that the
                # environment would follow the SCF density and the correlated
                # density would never get a say.
                e_tot = self._run_once()
                with_embedding.kernel(self.make_rdm1(ao_repr=True))

                de = np.max(np.abs(e_tot - e_last))
                log.info('Embedding macro cycle %d  E_tot = %.15g  dE = %g',
                         cycle, e_tot, de)
                if de < with_embedding.conv_tol:
                    break
                e_last = e_tot
            else:
                log.warn('The embedding did not settle against the %s density '
                         'in %d macro cycles (dE = %g, conv_tol = %g).',
                         self.__class__.__name__, with_embedding.max_cycle, de,
                         with_embedding.conv_tol)
            log.info('\n** Extra cycle for the embedding **')

        # For a frozen model this is the whole calculation; after the macro
        # iteration it is the pass that matches the potential it settled on.
        self._run_once()
        self._finalize()
        return self.e_corr, None

    def nuc_grad_method(self):
        raise NotImplementedError(
            'Nuclear gradients are not implemented for a correlated method in '
            'an embedding potential. The environment relaxes against the '
            'correlated density, so the gradient carries terms that the SCF '
            'one does not; see the notes on parity with pyscf.solvent.')

    Gradients = nuc_grad_method


class CASCIWithEmbedding(_CASWithEmbedding):
    '''CASCI in an embedding potential that relaxes against the CI density.'''

    def undo_embedding(self):
        '''Revert to the bare CASCI method.'''
        cls = self.__class__
        name_mixin = self.with_embedding.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, CASCIWithEmbedding, name_mixin))
        del obj.with_embedding
        return obj

    def _ci_density(self, ci):
        '''The density the environment is polarized by.

        One state, even where the solver returned several: the environment sees
        a density, not a spectrum, and state_id says whose.
        '''
        if isinstance(self.e_cas, (float, np.number)):
            return self.make_rdm1(ci=ci)
        return self.make_rdm1(ci=ci[self.with_embedding.state_id])

    def kernel(self, mo_coeff=None, ci0=None, verbose=None):
        '''CASCI, with the environment brought into step with the CI density.

        Each cycle solves the CI problem in the potential of the previous one,
        then rebuilds the potential from the density that came out. A frozen
        potential skips the loop: there is nothing to bring into step.

        Orbital canonicalization is left off inside the loop and done by the
        pass at the end, so that it happens once, on the converged solution.
        '''
        with_embedding = self.with_embedding
        log = logger.new_logger(self, verbose)
        # one verbosity step down, so the repeated CASCI output does not bury
        # the cycle-by-cycle report
        quiet = copy.copy(log)
        quiet.verbose -= 1

        def casci_cycle(ci0, log):
            # e_tot, e_cas and ci are set on self by the call
            e_tot, e_cas, ci0 = super(CASCIWithEmbedding, self).kernel(
                mo_coeff, ci0, log)[:3]
            dm = self._ci_density(ci0)
            self.e_tot = e_tot = e_tot + self._embedding_energy_correction(dm)
            if not with_embedding.frozen:
                with_embedding.kernel(dm)
            return e_tot, e_cas, ci0

        if with_embedding.frozen:
            with lib.temporary_env(self, _finalize=lambda: None):
                casci_cycle(ci0, log)
            self._finalize()
            return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

        log.info('\n** Relaxing %s against the %s density **',
                 with_embedding.__class__.__name__, self.__class__.__name__)
        self.converged = False
        with lib.temporary_env(self, canonicalization=False):
            e_last = 0
            for cycle in range(with_embedding.max_cycle):
                log.info('\n** Embedding macro cycle %d:', cycle)
                e_tot, e_cas, ci0 = casci_cycle(ci0, quiet)

                de = np.max(np.abs(e_tot - e_last))
                if isinstance(e_cas, (float, np.number)):
                    log.info('Embedding macro cycle %d  E(CASCI) = %.15g  '
                             'dE = %g', cycle, e_tot, de)
                else:
                    for i, e in enumerate(np.atleast_1d(e_tot)):
                        log.info('Embedding macro cycle %d  root %d  '
                                 'E(CASCI) = %.15g', cycle, i, e)
                if de < with_embedding.conv_tol:
                    self.converged = True
                    break
                e_last = e_tot
            else:
                log.warn('The embedding did not settle against the %s density '
                         'in %d macro cycles (dE = %g, conv_tol = %g).',
                         self.__class__.__name__, with_embedding.max_cycle, de,
                         with_embedding.conv_tol)

        # One more pass, to canonicalize the orbitals of the solution that was
        # converged to rather than of every one on the way there.
        with lib.temporary_env(self, _finalize=lambda: None):
            casci_cycle(ci0, log)
        self._finalize()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy


class CASSCFWithEmbedding(_CASWithEmbedding):
    '''CASSCF in an embedding potential that follows its macro iterations.

    Where CASCI cycles the environment around a converged solution, CASSCF has
    its own iteration to ride on: the potential is rebuilt wherever the density
    changes -- at the end of each CI solution and at each update of the active
    density -- so the environment arrives at self-consistency with the orbitals
    rather than after them.
    '''

    def __init__(self, mc, embedding_obj):
        super().__init__(mc, embedding_obj)
        self._e_tot_without_embedding = 0

    def undo_embedding(self):
        '''Revert to the bare CASSCF method.'''
        cls = self.__class__
        name_mixin = self.with_embedding.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, CASSCFWithEmbedding, name_mixin))
        del obj.with_embedding
        del obj._e_tot_without_embedding
        return obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        if self.conv_tol < 1e-7:
            logger.warn(self, 'CASSCF in an embedding potential may not reach '
                        'conv_tol = %g: the environment is rebuilt from a '
                        'density that is itself still converging.',
                        self.conv_tol)
        return self

    def update_casdm(self, mo, u, fcivec, e_cas, eris, envs={}):
        casdm1, casdm2, gci, fcivec = super().update_casdm(
            mo, u, fcivec, e_cas, eris, envs)

        # The potential follows the density of the current micro iteration,
        # which is not quite the CASSCF density that convergence is measured on
        # -- by the last macro iteration the two have come together.
        with_embedding = self.with_embedding
        if not with_embedding.frozen:
            # what self.make_rdm1(ci=fcivec) would give for these orbitals
            mocore = mo[:, :self.ncore]
            mocas = mo[:, self.ncore:self.ncore + self.ncas]
            dm = mocas.dot(casdm1).dot(mocas.conj().T)
            dm += mocore.dot(mocore.conj().T) * 2
            with_embedding.kernel(dm)

        return casdm1, casdm2, gci, fcivec

    def casci(self, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
        log = logger.new_logger(self, verbose)

        # super().casci reports dE against envs['elast'], which is the last
        # corrected total energy while the one it computes is not corrected yet.
        # Handing it the uncorrected predecessor is what makes that difference
        # the right one.
        envs['elast'] = self._e_tot_without_embedding
        e_tot, e_cas, fcivec = super().casci(mo_coeff, ci0, eris, verbose, envs)
        self._e_tot_without_embedding = e_tot

        # The orbitals have to come from the argument: mc1step.kernel keeps the
        # current ones in a local and does not put them on the object until it
        # returns, so self.mo_coeff is still the starting guess here. Building
        # the density from those and the CI vector of these would polarize the
        # environment with a density belonging to neither.
        dm = self.make_rdm1(mo_coeff=mo_coeff, ci=fcivec, ao_repr=True)
        correction = self._embedding_energy_correction(dm)
        e_tot = e_tot + correction
        log.debug('Embedding correction to the total energy: %.15g', correction)

        with_embedding = self.with_embedding
        if not with_embedding.frozen:
            with_embedding.kernel(dm)

        return e_tot, e_cas, fcivec


class TDSCFWithEmbedding(_Embedding):
    '''A TDSCF method whose SCF carries an embedding potential.

    The ground state environment is already in the orbitals, so what is left to
    decide is whether it relaxes against the transition density. It does not by
    default: a vertical excitation is fast next to the environment. See
    equilibrium_solvation.

    This describes a linear-response environment, not a state-specific one: the
    excited state does not get an environment polarized by its own density.
    '''

    _keys = {'with_embedding'}

    def __init__(self, method, embedding_obj):
        self.__dict__.update(method.__dict__)
        self.with_embedding = embedding_obj

    def undo_embedding(self):
        '''Revert to the bare TDSCF method, over a bare SCF.'''
        cls = self.__class__
        name_mixin = self.with_embedding.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, TDSCFWithEmbedding, name_mixin))
        obj._scf = self._scf.undo_embedding()
        del obj.with_embedding
        return obj

    @property
    def equilibrium_solvation(self):
        '''Whether the environment relaxes against the transition density.

        False, the default, is the non-equilibrium description and the right one
        for a vertical excitation: the environment stays as the ground state
        polarized it, and contributes nothing to the excitation energy beyond
        the orbitals it already shaped. True is for a process slow enough that
        the environment can follow.
        '''
        return self.with_embedding.equilibrium_solvation

    @equilibrium_solvation.setter
    def equilibrium_solvation(self, value):
        if value and self.with_embedding.frozen:
            raise RuntimeError(
                'The ground state embedding is frozen, which is a fixed '
                'external potential, and equilibrium_solvation asks the same '
                'potential to relax against the excited state. Set '
                '_scf.with_embedding.frozen = False and re-run the ground '
                'state.')
        self.with_embedding.equilibrium_solvation = value

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_embedding.check_sanity()
        self.with_embedding.dump_flags(verbose)
        return self

    def reset(self, mol=None):
        self.with_embedding.reset(mol)
        return super().reset(mol)

    def gen_response(self, *args, **kwargs):
        '''The response function of the excitation, environment included.

        Built from the bare SCF rather than from the embedded one, so that the
        flag consulted is this object's own -- the ground state's says what the
        ground state's orbital Hessian should contain, which is a different
        question. See _add_embedding_response.
        '''
        vind = self._scf.undo_embedding().gen_response(
            *args, with_nlc=not self.exclude_nlc, **kwargs)
        return _add_embedding_response(
            vind, self.with_embedding, _is_uhf_response(self._scf), args, kwargs)

    def get_ab(self, mf=None, frozen=None):
        raise NotImplementedError(
            'The A and B matrices are built from integrals directly, with no '
            'hook the embedding potential could reach, so they would silently '
            'describe an isolated molecule. Use the iterative solver instead.')

    def to_gpu(self):
        # See SCFWithEmbedding.to_gpu.
        raise NotImplementedError(
            'Embedding has no GPU implementation. Call '
            '.undo_embedding().to_gpu() to move the bare method to the GPU.')

    def nuc_grad_method(self):
        raise NotImplementedError(
            'Nuclear gradients are not implemented for an excited state in an '
            'embedding potential; see the notes on parity with pyscf.solvent.')

    Gradients = nuc_grad_method
