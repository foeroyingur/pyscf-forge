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
Polarizable and electrostatic embedding through an interface to PyFraME.

PolarizableEmbedding describes the environment by the multipoles of its sites
together with the dipoles they induce in the field of the quantum region;
ElectrostaticEmbedding keeps the multipoles and drops the induction, i.e. the
environment does not respond to the density.

TODO: fill in before merging.
GitHub:      XXX
Code:        Zenodo.XXX
Publication: XXX
'''

from __future__ import annotations

import contextlib
import warnings

import numpy as np

try:
    import pyframe.embedding  # noqa: F401
except ImportError as err:
    raise ImportError(
        'Unable to import PyFraME. Please install PyFraME.') from err

# The embedding models of PyFraME 0.5 are what everything below rests on, and
# relax is the newest part of them used here, so a PyFraME without it is
# refused up front rather than failing on the first model built or the first
# correlated calculation.
try:
    from pyframe.embedding import model as pyframe_model
except ImportError:
    pyframe_model = None
if not hasattr(getattr(pyframe_model, 'EmbeddingModel', None), 'relax'):
    raise ImportError(
        'The installed PyFraME is too old for pyscf.embedding, which needs the '
        'embedding models of PyFraME 0.5.0 or newer.')
from pyframe.embedding import read_input

from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf.data.elements import _std_symbol_without_ghost
from pyscf.embedding import _attach_embedding
from pyscf.embedding.integral_driver import EmbeddingIntegralDriver


@lib.with_doc(_attach_embedding._for_scf.__doc__)
def embedding_for_scf(mf, solvent_obj, dm=None, cls=None):
    if cls is None:
        cls = PolarizableEmbedding
    if not isinstance(solvent_obj, EmbeddingBase):
        solvent_obj = cls(mf.mol, solvent_obj)
    return _attach_embedding._for_scf(mf, solvent_obj, dm)


@lib.with_doc(_attach_embedding._for_post_scf.__doc__)
def embedding_for_post_scf(method, solvent_obj, dm=None, cls=None):
    if cls is None:
        cls = PolarizableEmbedding
    if not isinstance(solvent_obj, EmbeddingBase):
        solvent_obj = cls(method.mol, solvent_obj)
    return _attach_embedding._for_post_scf(method, solvent_obj, dm)


@lib.with_doc(_attach_embedding._for_tdscf.__doc__)
def embedding_for_tdscf(method, solvent_obj=None, dm=None, cls=None,
                        equilibrium_solvation=False):
    if solvent_obj is not None and not isinstance(solvent_obj, EmbeddingBase):
        if cls is None:
            cls = PolarizableEmbedding
        solvent_obj = cls(method.mol, solvent_obj)
    return _attach_embedding._for_tdscf(method, solvent_obj, dm,
                                        equilibrium_solvation)


@lib.with_doc(_attach_embedding._for_casci.__doc__)
def embedding_for_casci(mc, solvent_obj, dm=None, cls=None):
    if cls is None:
        cls = PolarizableEmbedding
    if not isinstance(solvent_obj, EmbeddingBase):
        solvent_obj = cls(mc.mol, solvent_obj)
    return _attach_embedding._for_casci(mc, solvent_obj, dm)


@lib.with_doc(_attach_embedding._for_casscf.__doc__)
def embedding_for_casscf(mc, solvent_obj, dm=None, cls=None):
    if cls is None:
        cls = PolarizableEmbedding
    if not isinstance(solvent_obj, EmbeddingBase):
        solvent_obj = cls(mc.mol, solvent_obj)
    return _attach_embedding._for_casscf(mc, solvent_obj, dm)




@contextlib.contextmanager
def _relayed_warnings(obj):
    '''Pass the warnings raised inside the block on to the PySCF output of obj.

    PyFraME warns through the warnings module -- about a molecule that does not
    match the potential, or sites without Lennard-Jones parameters -- which
    reaches stderr and not the output file with everything else. Every call
    into the model goes through here, since a warning may come from any of
    them: the Lennard-Jones parameters, for one, are read when the energy is
    first computed at a geometry.
    '''
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            yield
        finally:
            for warning in caught:
                logger.warn(obj, '%s', warning.message)


def _total_density(dm):
    '''The density the environment sees: a pair of spin blocks is summed.'''
    if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
        # spin-traced DM for UHF or ROHF
        dm = dm[0] + dm[1]
    return np.asarray(dm)


def _integral_key(mol):
    '''What the model's nuclei and integral driver depend on in a Mole.

    The geometry, the charges, and the basis all live in these arrays, so a
    Mole whose arrays are equal to them needs neither new nuclei nor new
    integrals. The Mole object itself says nothing: a scanner moves the same
    object in place with set_geom_.
    '''
    return (mol._atm.tobytes(), mol._bas.tobytes(), mol._env.tobytes(),
            mol._ecpbas.tobytes(), mol.cart)


class EmbeddingBase(lib.StreamObject):
    '''A molecular mechanics (MM) embedding potential attached to a molecule.

    A PySCF object holding one of PyFraME's embedding models, which does the
    embedding itself: the energy and Fock matrix contribution of a density, the
    response, the gradient and the Hessian. What is added here is what is
    PySCF's -- the Mole the model's nuclei and integral driver are made from,
    the state the attachments to PySCF's methods read (e, v, frozen,
    equilibrium_solvation), and the output. The subclasses only say which
    model; this class is not one and is not meant to be instantiated.

    A model is built from a molecule and either the potential itself or a
    dictionary of options:

    potential
        the potential: the path of a PyFraME JSON file or the JSON document as
        a dictionary, a PyFraME MolecularSystem or Snapshot whose embedding
        potential has been created, or the Subsystems of one. Its quantum
        subsystem has to be the molecule, link and capping atoms included.
    vdw
        {"method": "LJ", "combination_rule": "Lorentz-Berthelot"}, to include
        the Lennard-Jones repulsion and dispersion. A rule is named
        "<sigma rule>-<epsilon rule>", Lorentz or Good-Hope for sigma and
        Berthelot or Fender-Halsey for epsilon; the powers are carried by the
        sites of the potential.
    environment_energy
        whether the internal energy of the environment is reported (True)
    induced_dipoles
        PolarizableEmbedding only: "threshold", "max_iterations" and "solver"
        of the induced dipoles

    Every option but the potential is PyFraME's, and is read and checked by
    pyframe.embedding.options.EmbeddingOptions.from_dict.
    '''

    # The PyFraME model class, set by each subclass.
    _model_class = None

    # Labels of the terms of pyframe.embedding.model.EnergyContributions in
    # the energy report.
    _term_labels = {
        'electrostatic': 'Electrostatic contribution  (E_es)  ',
        'induction': 'Induction contribution      (E_ind) ',
        'repulsion': 'Repulsion contribution      (E_rep) ',
        'dispersion': 'Dispersion contribution     (E_disp)',
    }

    _keys = {'mol', 'options', 'model', 'e', 'v', 'equilibrium_solvation',
             'max_cycle', 'conv_tol', 'state_id'}

    def __init__(self, molecule, options_or_potential):
        if self._model_class is None:
            raise TypeError('EmbeddingBase is no kind of embedding; use '
                            'PolarizableEmbedding or ElectrostaticEmbedding.')
        self.stdout = molecule.stdout
        self.verbose = molecule.verbose
        # e (the embedding energy) and v (the additional potential) are
        # updated during the SCF iterations
        self.e = None
        self.v = None
        # The macro iteration that relaxes the environment against a correlated
        # density runs to these; see relax. They are not the SCF's own
        # thresholds -- the environment converges alongside it, not within it --
        # and conv_tol is on the potential rather than the energy: the largest
        # change of any element of v from one macro cycle to the next.
        self.max_cycle = 20
        self.conv_tol = 1e-6
        # Which state the environment is polarized by, when the method solves
        # for several. The environment sees one density, so a multi-root CASCI
        # has to say which.
        self.state_id = 0
        # Whether the environment is taken to relax against a first-order
        # density, which is what gen_response asks before adding _B_dot_x. A
        # ground-state orbital Hessian wants it; a vertical excitation does
        # not, the environment being slow next to the electrons, so the default
        # is the non-equilibrium one, as in pyscf.solvent. The name is
        # solvent's, kept so that the two read alike.
        self.equilibrium_solvation = False

        self.mol = molecule
        self._set_options(options_or_potential)
        self._build_model()

    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s flags ********', self.__class__)
        logger.info(self, 'frozen = %s', self.frozen)
        logger.info(self, 'max_cycle = %s', self.max_cycle)
        logger.info(self, 'conv_tol = %s', self.conv_tol)
        logger.info(self, 'state_id = %s', self.state_id)
        logger.info(self, 'equilibrium_solvation = %s', self.equilibrium_solvation)
        for key, value in self.options.items():
            if key == 'potential':
                # a potential held in memory is described rather than printed
                value = read_input.describe_potential(value)
            logger.info(self, "mm.%s = %s", key, value)
        return self

    def _set_options(self, options_or_potential):
        '''Take the options, or the potential alone, whose options are the defaults.'''
        if isinstance(options_or_potential, dict):
            options = options_or_potential
            if 'potential' not in options:
                raise ValueError(
                    'The options dictionary must contain the key "potential", '
                    'giving the MM potential of the embedding.')
        else:
            options = {'potential': options_or_potential}
        self.options = options

    def _build_model(self):
        '''Build the PyFraME model of the potential, at the geometry of the Mole.'''
        options = {key: value for key, value in self.options.items()
                   if key != 'potential'}
        with _relayed_warnings(self):
            self.model = self._model_class(self.potential,
                                           EmbeddingIntegralDriver(self.mol),
                                           options)
        if self.model.simulation_box is not None:
            logger.info(self, 'The potential carries a simulation box, which is '
                        'not used: the environment is treated as a cluster, '
                        'without the minimum image convention.')
        coordinates, charges = self._checked_nuclei()
        with _relayed_warnings(self):
            self.model.update_nuclei(coordinates=coordinates, charges=charges)
        self._model_key = _integral_key(self.mol)

    def _checked_nuclei(self):
        '''Check the Mole against the potential, and return the nuclei for the model.

        The nuclei of the potential describe the system it was made for, while
        the Mole is the one that is being computed on, and the one that moves in
        a scanner or geometry optimization. They have to be the same molecule,
        atom for atom and in the same order: the terminated core region, link
        and capping atoms included.

        The charges the environment sees are those of the molecule's nuclei as
        PySCF treats them, i.e. mol.atom_charges(): an atom carrying an ECP
        interacts with the environment through its effective charge, as its
        core electrons are not in the density, and a ghost atom not at all. The
        elements are compared through the atomic numbers, ghosts included as
        the element they stand for, since the charges differ wherever there is
        an ECP or a ghost atom.

        Returns:
            The coordinates and the charges of the nuclei.
        '''
        mol = self.mol
        atomic_numbers = [gto.charge(_std_symbol_without_ghost(mol.atom_symbol(i)))
                          for i in range(mol.natm)]
        charges = mol.atom_charges().astype(np.float64)
        with _relayed_warnings(self):
            self.model.check_nuclei(atomic_numbers, charges=charges,
                                    total_charge=mol.charge)
        return mol.atom_coords(), charges

    def reset(self, mol=None, options_or_potential=None):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        if mol is not None:
            self.mol = mol
        if options_or_potential is not None:
            # A different potential was given, so the model has to be built
            # from it, frozen where this one was.
            frozen_at = self.model.density_matrix if self.frozen else None
            self._set_options(options_or_potential)
            self._build_model()
            if frozen_at is not None:
                self.model.freeze(frozen_at)
        elif _integral_key(self.mol) != self._model_key:
            # The potential is unchanged, so only the nuclei and the integrals
            # follow the Mole. Re-reading the potential would rebuild identical
            # subsystems, which dominates the cost of a geometry optimization
            # step once the potential is of a realistic size. A frozen model
            # keeps the density it is frozen at, and rebuilds the potential of
            # that density at the new geometry on the next kernel -- frozen
            # means the environment does not follow the density, not that it
            # does not follow the nuclei.
            coordinates, charges = self._checked_nuclei()
            with _relayed_warnings(self):
                self.model.update_geometry(coordinates,
                                           EmbeddingIntegralDriver(self.mol),
                                           charges=charges)
            self._model_key = _integral_key(self.mol)
        # else: neither the nuclei nor the basis moved, which is every reset
        # of the post-SCF macro iteration, whose scanner resets on the same
        # molecule each cycle. The model keeps its integrals and, frozen, the
        # dipoles of the density it is frozen at, rather than solving for them
        # again.
        self.e = None
        self.v = None
        return self

    @property
    def potential(self):
        '''The potential as it was given.'''
        return self.options['potential']

    @property
    def quantum_subsystem(self):
        return self.model.quantum_subsystem

    @property
    def classical_subsystem(self):
        return self.model.classical_subsystem

    @property
    def simulation_box(self):
        '''The simulation box of the potential, or None. Kept for reference
        only: the environment is treated as a cluster.'''
        return self.model.simulation_box

    @property
    def polarizable(self):
        '''Whether the environment responds to the density.'''
        return self._model_class.polarizable

    @property
    def frozen(self):
        '''Whether the potential is held at one density instead of following it.

        Setting it holds the potential at the density kernel was last run on;
        see _for_scf's dm argument, which is the usual way to set it. The model
        then reports the energy of the density it is handed in that potential,
        which is what an SCF with the potential in its Fock matrix makes
        stationary; see pyframe.embedding.model.EmbeddingModel.freeze.
        '''
        return self.model.frozen

    @frozen.setter
    def frozen(self, value):
        if bool(value) == self.model.frozen:
            return
        if value:
            if self.model.density_matrix is None:
                raise RuntimeError(
                    'The embedding holds no density matrix to freeze the '
                    'potential at. Pass dm when attaching the embedding to a '
                    'method, or run kernel once before setting frozen.')
            self.model.freeze()
        else:
            self.model.unfreeze()

    def kernel(self, dm):
        '''Embedding energy and Fock matrix contribution for a density matrix.

        Args:
            dm : the density matrix, either closed-shell or a pair of spin
                blocks, which is traced over spin.

        Returns:
            (e, v), the embedding energy and the potential to be added to the
            Fock matrix. Both are also stored on the object.

        What a frozen model holds fixed is the potential, not the energy: the
        energy is that of the density it is handed in the potential of the
        density it was frozen at.
        '''
        dm = _total_density(dm)
        with _relayed_warnings(self):
            self.e, self.v = self.model.energy_and_fock(dm)
        return self.e, self.v

    def relax(self, run):
        '''Relax the environment against a method that is run in turn with it.

        For a method whose density comes only from running it, e.g. a
        correlated method or CASCI, which cannot take the environment into its
        own iterations as an SCF does: each macro cycle holds the potential at
        the density of the one before -- the SCF's, first -- runs the method in
        it, and rebuilds the potential from the density that comes out, until v
        changes by less than conv_tol, in at most max_cycle runs. See
        pyframe.embedding.model.EmbeddingModel.relax, which this is.

        Args:
            run : runs the method and returns its density matrix, closed-shell
                or a pair of spin blocks. The potential held is in e and v, and
                frozen is set, while it runs.

        Returns:
            The pyframe.embedding.model.Relaxation: whether it converged, and
            the change of v in each cycle. e and v are afterwards those of the
            density the last run gave.
        '''
        def held_run():
            # e and v are what a method reads the potential from
            self.kernel(self.model.density_matrix)
            return _total_density(run())

        with _relayed_warnings(self):
            relaxation = self.model.relax(held_run, threshold=self.conv_tol,
                                          max_iterations=self.max_cycle)
        for cycle, change in enumerate(relaxation.changes):
            logger.info(self, 'Embedding macro cycle %d  max|dv| = %.3g',
                        cycle, change)
        self.kernel(self.model.density_matrix)
        return relaxation

    def _B_dot_x(self, dm):
        '''The response of the environment to a trial density matrix.

        The change in the embedding potential caused by a first-order density
        dm, i.e. the environment block of the orbital Hessian applied to a trial
        vector: the potential of the dipoles the trial density's electronic
        field alone induces, which is zero for a model without induced dipoles
        and for a frozen one. The model's own dipoles are left as they are.

        Args:
            dm : one density matrix or a stack of them, of any leading shape.

        Returns:
            The potential, shaped like dm.
        '''
        with _relayed_warnings(self):
            return self.model.response(dm)

    def gradient(self, dm):
        '''The gradient of the embedding energy at a fixed density, (natm, 3).'''
        with _relayed_warnings(self):
            return self.model.gradient(_total_density(dm))

    def hessian(self, dm):
        '''The Hessian of the embedding energy at a fixed density, (3 natm, 3 natm).'''
        with _relayed_warnings(self):
            return self.model.hessian(_total_density(dm))

    def fock_derivatives(self, dm=None):
        '''The derivative of v by every nucleus at a fixed density, (natm, 3, nao, nao).

        The density is needed only where the environment is polarizable: the
        potential of the permanent multipoles does not depend on it.
        '''
        with _relayed_warnings(self):
            return self.model.fock_derivatives(
                None if dm is None else _total_density(dm))

    def nuc_grad_method(self, grad_method):
        from pyscf.embedding import embedding_gradient
        return embedding_gradient.make_grad_object(grad_method)

    def hess_method(self, hess_method):
        from pyscf.embedding import embedding_hessian
        return embedding_hessian.make_hess_object(hess_method)

    def energy_contributions(self):
        '''The embedding energy broken down by contribution, for reporting.

        The terms of the last kernel, which sum to self.e. The internal energy
        of the environment is reported alongside them when the
        environment_energy option is set, but it is a constant offset that no
        term of the total energy depends on, so it is labelled as excluded.
        '''
        contributions = self.model.energy_contributions()
        labelled = {self._term_labels.get(name, name): value
                    for name, value in contributions.terms.items()}
        if contributions.environment is not None:
            labelled['Environment energy (not in E_tot)   '] = contributions.environment
        return labelled

    def environment_energy(self):
        '''Internal interaction energy of the environment.

        A constant for a fixed environment, not part of the total energy that
        kernel returns; see energy_contributions. The repulsion and dispersion
        within the environment are included only when the vdw option is set.
        '''
        return self.model.environment_energy()


class ElectrostaticEmbedding(EmbeddingBase):
    '''Embedding in the permanent multipoles of a molecular mechanics (MM) potential.

    The sites of the environment interact with the quantum region through their
    permanent multipoles. The polarizabilities the potential may carry are
    ignored: no induced dipoles are solved for, so the environment does not
    respond to the density.
    '''

    _model_class = pyframe_model.ElectrostaticEmbedding


class PolarizableEmbedding(EmbeddingBase):
    '''Embedding in the multipoles and induced dipoles of an MM potential.

    The electrostatics of ElectrostaticEmbedding, with the polarizabilities of
    the MM potential put to use on top of them: the dipoles the quantum region
    induces in the environment are solved for at every density matrix, so the
    environment responds and the coupling is self-consistent. A potential
    without polarizabilities is refused; use ElectrostaticEmbedding.
    '''

    _model_class = pyframe_model.PolarizableEmbedding
