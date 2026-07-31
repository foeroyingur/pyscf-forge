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
Analytical nuclear Hessians for electrostatic embedding.

The embedding energy enters the Hessian in three places, and this module covers
the two that are its own:

    the second derivative at fixed density, d2E/dR_A dR_B, which is what
        kernel returns;
    the derivative of the embedding potential, dv/dR_A, which belongs in the
        first-order Fock matrix the CPHF equations are solved against, and is
        what fock_gradient returns;

while the third -- the environment's place in the orbital Hessian itself -- is
already handled by gen_response, which EmbeddingHess.kernel switches on for the
duration.

Induction is not covered; see EmbeddingHess and the module notes.
'''

from __future__ import annotations

import numpy as np

try:
    from pyframe.embedding import (electrostatic_interactions,
                                   repulsion_interactions, dispersion_interactions)
except ImportError as err:
    raise ImportError(
        'Unable to import PyFraME. Please install PyFraME.') from err

from pyscf import lib
from pyscf import gto
from pyscf import df, scf
from pyscf.embedding import embedding
from pyscf.embedding import embedding_gradient
from pyscf.embedding._attach_embedding import _Embedding


class EmbeddingHessianIntegralDriver:
    '''Second AO derivatives of the embedding potential operator.

    The MM sites do not move with the nuclei, so every nuclear derivative lands
    on an atomic orbital. Two of them are distributed over the bra and the ket,
    which is the whole content of the two arrays this returns.
    '''

    def __init__(self, molecule):
        self.mol = molecule
        self.max_memory = molecule.max_memory

    def multipole_potential_hessian_integrals(self,
                                              multipole_coordinates: np.ndarray,
                                              multipole_orders: np.ndarray,
                                              multipoles: list[np.ndarray]):
        """Second derivatives of the multipole potential integrals.

        Args:
            multipole_coordinates: Coordinates of the Multipoles.
                Shape: (number of sites, 3)
            multipole_orders: Multipole orders of all multipoles.
                Shape: (number of sites)
            multipoles: Multipoles multiplied with degeneracy coefficients and
                taylor coefficients.

        Returns:
            (bra_bra, bra_ket), each of shape (3, 3, nao, nao), holding
            <d_x p|V|q> differentiated once more on the bra and once on the ket
            respectively, for the same operator V that
            EmbeddingIntegralDriver.multipole_potential_integrals builds.

        A site of order L carries L derivatives in the operator itself, moved
        onto the atomic orbitals by integration by parts, so these arrays need
        L + 2 derivatives in all. The three-index family stops at two, which is
        why only the charges are batched over sites with a fakemol and the
        dipoles and quadrupoles re-centre rinv on each site in turn -- the same
        wall the gradient meets one order lower.
        """
        multipole_orders = np.asarray(multipole_orders)
        if np.any(multipole_orders > 2):
            raise NotImplementedError("""Multipole potential integrals not
                                             implemented for order > 2.""")
        nao = self.mol.nao
        bra_bra = np.zeros((3, 3, nao, nao))
        bra_ket = np.zeros((3, 3, nao, nao))

        # 0 order
        idx = np.where(multipole_orders >= 0)[0]
        fakemol = gto.fakemol_for_charges(multipole_coordinates[idx])
        charges = np.array([multipoles[i][0:1] for i in idx])
        bra_bra -= np.einsum('cijg,ga->cij', df.incore.aux_e2(
            self.mol, fakemol, intor='int3c2e_ipip1', comp=9),
            charges).reshape(3, 3, nao, nao)
        bra_ket -= np.einsum('cijg,ga->cij', df.incore.aux_e2(
            self.mol, fakemol, intor='int3c2e_ipvip1', comp=9),
            charges).reshape(3, 3, nao, nao)

        # 1 order
        if np.any(multipole_orders >= 1):
            idx = np.where(multipole_orders >= 1)[0]
            dipoles = np.array([multipoles[i][1:4] for i in idx])
            embedding._blocked_range(self.mol, 1, 2 * 27 * nao * nao * 8,
                                     self.max_memory, 'dipole hessian integrals')
            for site, dipole in zip(multipole_coordinates[idx], dipoles):
                with self.mol.with_rinv_orig(site):
                    b3 = self.mol.intor('int1e_ipipiprinv', comp=27).reshape(3, 3, 3, nao, nao)
                    b2k1 = self.mol.intor('int1e_ipiprinvip', comp=27).reshape(3, 3, 3, nao, nao)
                # the operator's own derivative goes on the bra or on the ket,
                # and the two nuclear ones on top of it
                bra_bra += np.einsum('axyij,a->xyij', b3, dipole)
                bra_bra += np.einsum('xyaij,a->xyij', b2k1, dipole)
                bra_ket += np.einsum('axyij,a->xyij', b2k1, dipole)
                # (1 on the bra, 2 on the ket) is (2 on the bra, 1 on the ket)
                # read backwards, the integrals being real
                bra_ket += np.einsum('ayxji,a->xyij', b2k1, dipole)

        # 2 order
        if np.any(multipole_orders >= 2):
            idx = np.where(multipole_orders >= 2)[0]
            raw = np.array([multipoles[i][4:10] for i in idx])
            quadrupoles = np.zeros((idx.size, 9))
            quadrupoles[:, [0, 1, 2, 4, 5, 8]] = raw
            quadrupoles[:, [0, 3, 6, 4, 7, 8]] += raw
            quadrupoles *= -0.5
            embedding._blocked_range(self.mol, 1, 3 * 81 * nao * nao * 8,
                                     self.max_memory, 'quadrupole hessian integrals')
            for site, quadrupole in zip(multipole_coordinates[idx],
                                        quadrupoles.reshape(-1, 3, 3)):
                with self.mol.with_rinv_orig(site):
                    b4 = self.mol.intor('int1e_ipipipiprinv', comp=81).reshape(3, 3, 3, 3, nao, nao)
                    b3k1 = self.mol.intor('int1e_ipipiprinvip', comp=81).reshape(3, 3, 3, 3, nao, nao)
                    b2k2 = self.mol.intor('int1e_ipiprinvipip', comp=81).reshape(3, 3, 3, 3, nao, nao)
                # the operator's two derivatives split over bra and ket in the
                # four ways, each carrying the two nuclear ones
                bra_bra += np.einsum('abxyij,ab->xyij', b4, quadrupole)
                bra_bra += np.einsum('axybij,ab->xyij', b3k1, quadrupole)
                bra_bra += np.einsum('bxyaij,ab->xyij', b3k1, quadrupole)
                bra_bra += np.einsum('xyabij,ab->xyij', b2k2, quadrupole)
                bra_ket += np.einsum('abxyij,ab->xyij', b3k1, quadrupole)
                bra_ket += np.einsum('axbyij,ab->xyij', b2k2, quadrupole)
                bra_ket += np.einsum('bxayij,ab->xyij', b2k2, quadrupole)
                bra_ket += np.einsum('abyxji,ab->xyij', b3k1, quadrupole)
        return bra_bra, bra_ket


def _hess_from_operators(mol, bra_bra, bra_ket, dm, atmlst):
    '''The second derivative of Tr(dm V) over the nuclei, from the two arrays.

    A nuclear derivative acts on the orbitals centred on that nucleus and
    nowhere else, so the double derivative is the bra-bra term on the diagonal
    of the atom pairs plus the bra-ket term on every pair. The terms where one
    or both derivatives land on the ket are the same numbers read backwards for
    a symmetric density, which is where the factors of two come from.
    '''
    ao_slices = mol.aoslice_by_atom()
    natoms = len(atmlst)
    hess = np.zeros((natoms, natoms, 3, 3))
    for i0, ia in enumerate(atmlst):
        p0, p1 = ao_slices[ia, 2:]
        hess[i0, i0] += 2 * np.einsum('xypq,pq->xy', bra_bra[:, :, p0:p1], dm[p0:p1])
        for j0, ja in enumerate(atmlst):
            q0, q1 = ao_slices[ja, 2:]
            hess[i0, j0] += 2 * np.einsum('xypq,pq->xy',
                                          bra_ket[:, :, p0:p1, q0:q1],
                                          dm[p0:p1, q0:q1])
    return hess


def _operator_derivative(mol, op, atmlst):
    '''Turn <d_x p|O|q> into the derivative of O with respect to each nucleus.

    The counterpart of embedding_gradient._grad_from_operator, which is the same
    thing already contracted with a density matrix: what comes back here is the
    operator itself, one (3, nao, nao) array per atom, for the Fock matrix
    derivative the CPHF equations are solved against.
    '''
    ao_slices = mol.aoslice_by_atom()
    derivatives = []
    for ia in atmlst:
        p0, p1 = ao_slices[ia, 2:]
        derivative = np.zeros_like(op)
        derivative[:, p0:p1] -= op[:, p0:p1]
        derivatives.append(derivative + derivative.transpose(0, 2, 1))
    return derivatives


def fock_gradient(embedding_obj, atmlst=None):
    '''The derivative of the embedding potential with respect to the nuclei.

    Returns one (3, nao, nao) array per atom of atmlst, to be added to the
    first-order Fock matrix that the CPHF equations are solved against.
    '''
    mol = embedding_obj.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    driver = embedding_gradient.EmbeddingGradientIntegralDriver(molecule=mol)
    f_es_grad = electrostatic_interactions.es_fock_matrix_gradient_contributions(
        classical_subsystem=embedding_obj.classical_subsystem,
        integral_driver=driver)
    return _operator_derivative(mol, f_es_grad, atmlst)


def kernel(embedding_obj, dm, atmlst=None, verbose=None):
    '''The second derivative of the embedding energy at a fixed density matrix.

    Args:
        embedding_obj : the ElectrostaticEmbedding carrying the potential.
        dm : the density matrix, either closed-shell or a pair of spin blocks,
            which is traced over spin.
        atmlst : the atoms the Hessian is computed for. Defaults to all.

    Returns:
        Shape (len(atmlst), len(atmlst), 3, 3), matching pyscf's Hessian layout.
    '''
    if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
        dm = dm[0] + dm[1]

    mol = embedding_obj.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = list(atmlst)
    quantum_subsystem = embedding_obj.quantum_subsystem
    classical_subsystem = embedding_obj.classical_subsystem

    driver = EmbeddingHessianIntegralDriver(molecule=mol)
    bra_bra, bra_ket = driver.multipole_potential_hessian_integrals(
        classical_subsystem.coordinates,
        classical_subsystem.multipole_orders,
        classical_subsystem.degenerate_multipoles_with_taylor_coefficients)
    de = _hess_from_operators(mol, bra_bra, bra_ket, dm, atmlst)

    # The interaction of the nuclei with the permanent multipoles is classical
    # and comes back as a flat (3 natm, 3 natm) matrix. A nucleus reaches the
    # sites through it but not the other nuclei, so only the 3x3 blocks on the
    # diagonal of the atom pairs are non-zero; it is reshaped rather than
    # sliced, since that is a property of the term and not of the layout.
    nuclear = electrostatic_interactions.compute_electrostatic_nuclear_hessian(
        quantum_subsystem=quantum_subsystem,
        classical_subsystem=classical_subsystem)
    nuclear = nuclear.reshape(mol.natm, 3, mol.natm, 3).transpose(0, 2, 1, 3)
    de += nuclear[np.ix_(atmlst, atmlst)]

    if embedding_obj.vdw_method is not None:
        # Lennard-Jones couples each nucleus to the sites and not to the other
        # nuclei, so only the diagonal of the atom pairs is non-zero and that is
        # all PyFraME returns: (natm, 3, 3).
        vdw = np.asarray(repulsion_interactions.compute_repulsion_interactions_hessian(
            quantum_subsystem=quantum_subsystem,
            classical_subsystem=classical_subsystem,
            method=embedding_obj.vdw_method,
            combination_rule=embedding_obj.vdw_combination_rule))
        vdw = vdw + np.asarray(dispersion_interactions.compute_dispersion_interactions_hessian(
            quantum_subsystem=quantum_subsystem,
            classical_subsystem=classical_subsystem,
            method=embedding_obj.vdw_method,
            combination_rule=embedding_obj.vdw_combination_rule))
        for i0, ia in enumerate(atmlst):
            de[i0, i0] += vdw[ia]
    return de


def make_hess_object(hess_method):
    '''Create a Hessian object including the embedding contributions.

    Args:
        hess_method : the Hessian object of an embedding-attached method, or
            the embedding-attached method itself.
    '''
    from pyscf.hessian.rhf import HessianBase
    if isinstance(hess_method, HessianBase):
        base_method = hess_method.base
    else:
        # For symmetry with pyscf.solvent, whose make_hess_object takes the
        # method rather than its Hessian object.
        base_method = hess_method
    if not isinstance(base_method, _Embedding):
        raise TypeError('%s does not carry an embedding potential.' % base_method)
    if not isinstance(base_method, scf.hf.SCF):
        raise NotImplementedError(
            'Embedding Hessians only implemented for SCF methods.')

    with_embedding = base_method.with_embedding
    if with_embedding.frozen:
        raise NotImplementedError(
            'Nuclear Hessians are not implemented for a frozen embedding '
            'potential. The energy is stationary, so the gradient is available, '
            'but its second derivative needs how the potential of the frozen '
            'density answers a displacement twice over -- see the notes on '
            'parity with pyscf.solvent.')
    if with_embedding.with_induction:
        raise NotImplementedError(
            'Nuclear Hessians are not implemented for a polarizable embedding '
            'potential. The induced dipoles respond to a nuclear displacement, '
            'and the terms that describes are not derived here; see the notes '
            'on parity with pyscf.solvent. Use embedding.electrostatic for a '
            'Hessian in the permanent multipoles alone.')

    vac_hess = base_method.undo_embedding().Hessian()
    vac_hess.base = base_method
    name = with_embedding.__class__.__name__ + vac_hess.__class__.__name__
    return lib.set_class(EmbeddingHess(vac_hess),
                         (EmbeddingHess, vac_hess.__class__), name)


class EmbeddingHess:
    '''The Hessian of a method in an embedding potential.

    The environment reaches the Hessian three ways, and they are kept apart
    here: its own second derivative at a fixed density (kernel, below), its
    place in the first-order Fock matrix (make_h1), and its place in the orbital
    Hessian that the CPHF equations are solved with (gen_response, switched on
    for the duration by equilibrium_solvation).
    '''

    _keys = {'de_classical_subsystem', 'de_quantum_subsystem'}

    def __init__(self, hess_method):
        self.__dict__.update(hess_method.__dict__)
        self.de_classical_subsystem = None
        self.de_quantum_subsystem = None

    def undo_embedding(self):
        '''Revert to the Hessian of the bare method.'''
        cls = self.__class__
        name_mixin = self.base.with_embedding.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, EmbeddingHess, name_mixin))
        del obj.de_classical_subsystem
        del obj.de_quantum_subsystem
        return obj

    def to_gpu(self):
        # See SCFWithEmbedding.to_gpu.
        raise NotImplementedError(
            'Embedding has no GPU implementation. Call '
            '.undo_embedding().to_gpu() to move the bare Hessian to the GPU.')

    def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        '''The Hessian of the embedded SCF energy.

        Kwargs:
            dm : the density matrix the embedding contributions are evaluated
                with. Defaults to that of the converged base method.
            atmlst : the atoms the Hessian is computed for. Defaults to all.
        '''
        if atmlst is not None:
            self.atmlst = atmlst
        atmlst = self.atmlst

        # The environment relaxes along with the density, so its response is
        # part of the orbital Hessian the CPHF equations are solved with.
        with lib.temporary_env(self.base.with_embedding,
                               equilibrium_solvation=True):
            self.de_quantum_subsystem = super().kernel(*args, atmlst=atmlst,
                                                       **kwargs)

        if dm is None:
            dm = self.base.make_rdm1(ao_repr=True)
        self.de_classical_subsystem = kernel(self.base.with_embedding, dm,
                                             atmlst=atmlst)
        self.de = self.de_quantum_subsystem + self.de_classical_subsystem
        return self.de

    hess = kernel

    def make_h1(self, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
        '''The first-order Fock matrix, with the embedding potential's share.

        The potential is a functional of the density, so displacing a nucleus
        changes it; that change belongs in the operator the CPHF equations are
        solved against, alongside the derivative of the core Hamiltonian.
        '''
        if atmlst is None:
            atmlst = range(self.mol.natm)
        h1ao = super().make_h1(mo_coeff, mo_occ, atmlst=atmlst, verbose=verbose)
        dv = fock_gradient(self.base.with_embedding, atmlst)
        if isinstance(self.base, (scf.uhf.UHF, scf.rohf.ROHF)):
            # the environment sees the total density and answers both spin
            # blocks with the same potential
            h1ao_a, h1ao_b = h1ao
            for i0, ia in enumerate(atmlst):
                h1ao_a[ia] += dv[i0]
                h1ao_b[ia] += dv[i0]
            return h1ao_a, h1ao_b
        for i0, ia in enumerate(atmlst):
            h1ao[ia] += dv[i0]
        return h1ao
