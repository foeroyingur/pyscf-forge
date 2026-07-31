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
Analytical nuclear gradients for polarizable and electrostatic embedding.
'''

from __future__ import annotations

import numpy as np

try:
    from pyframe.embedding import (electrostatic_interactions, induction_interactions,
                                   repulsion_interactions, dispersion_interactions)
except ImportError as err:
    raise ImportError(
        'Unable to import PyFraME. Please install PyFraME.') from err

from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import df, scf
from pyscf.embedding import embedding
from pyscf.embedding._attach_embedding import _Embedding
from pyscf.grad import rhf as rhf_grad


class EmbeddingGradientIntegralDriver:

    def __init__(self, molecule):
        self.mol = molecule
        self.max_memory = molecule.max_memory
        # See the note on embedding.EmbeddingIntegralDriver: the same
        # intor is evaluated at different sets of sites, so the cache has to be
        # keyed on the coordinates as well.
        self._integrals = {}
        self._cached_bytes = 0

    def _aux_e2_blocks(self, intor, coordinates, comp):
        '''Yield (p0, p1, integrals) over blocks of sites.

        See embedding.EmbeddingIntegralDriver._aux_e2_blocks; this differs
        only in passing comp on to aux_e2, which the gradient intors need.
        '''
        coordinates = np.asarray(coordinates)
        nsites = len(coordinates)
        key = (intor, coordinates.shape, coordinates.tobytes())
        integrals = self._integrals.get(key)
        if integrals is not None:
            yield 0, nsites, integrals
            return

        nao = self.mol.nao
        nbytes_per_site = comp * nao * nao * 8
        cache_budget = self.max_memory * embedding.CACHE_FRACTION * 1e6
        if self._cached_bytes + nsites * nbytes_per_site <= cache_budget:
            integrals = self._build(intor, coordinates, comp)
            self._integrals[key] = integrals
            self._cached_bytes += integrals.nbytes
            yield 0, nsites, integrals
            return

        blksize = embedding._blocked_range(self.mol, nsites, nbytes_per_site,
                                           self.max_memory, '%s integrals' % intor)
        for p0, p1 in lib.prange(0, nsites, blksize):
            yield p0, p1, self._build(intor, coordinates[p0:p1], comp)

    def _build(self, intor, coordinates, comp):
        fakemol = gto.fakemol_for_charges(coordinates)
        return df.incore.aux_e2(self.mol, fakemol, intor=intor, comp=comp)

    def multipole_potential_gradient_integrals(self,
                                               multipole_coordinates: np.ndarray,
                                               multipole_orders: np.ndarray,
                                               multipoles: list[np.ndarray]) -> np.ndarray:
        """Calculate the gradient of the electronic potential integrals and multiply with the multipoles.

        Args:
            multipole_coordinates: Coordinates of the Multipoles.
                Shape: (number of atoms, 3)
                Dtype: np.float64.
            multipole_orders: Multipole orders of all multipoles.
                Shape: (number of atoms)
                Dtype: np.int64
            multipoles: Multipoles multiplied with degeneracy coefficients and taylor coefficients.
                Shape: (number of atoms, number of multipole elements)
                Dtype: np.float64

        Returns:
            Product of gradient of electronic potential integrals and multipoles.
                Shape: (number of nuclei, number of ao functions, number of ao functions)
                Dtype: np.float64
        """
        if np.any(multipole_orders > 2):
            raise NotImplementedError("""Multipole potential integrals not
                                             implemented for order > 2.""")
        op = 0
        # 0 order
        idx = np.where(multipole_orders >= 0)[0]
        charge_coordinates = multipole_coordinates[idx]
        charges = np.array([multipoles[i][0:1] for i in idx])
        for p0, p1, integrals in self._aux_e2_blocks('int3c2e_ip1', charge_coordinates, comp=3):
            op -= np.einsum('cijg,ga->cij', integrals, charges[p0:p1])
        # 1 order
        if np.any(multipole_orders >= 1):
            idx = np.where(multipole_orders >= 1)[0]
            dipole_coordinates = multipole_coordinates[idx]
            dipoles = np.array([multipoles[i][1:4] for i in idx])
            v = 0
            for p0, p1, integrals in self._aux_e2_blocks('int3c2e_ipip1', dipole_coordinates, comp=9):
                integrals = integrals.reshape(3, 3, *integrals.shape[1:])
                v = v + np.einsum('caijg,ga->cij', integrals, dipoles[p0:p1])
            for p0, p1, integrals in self._aux_e2_blocks('int3c2e_ipvip1', dipole_coordinates, comp=9):
                integrals = integrals.reshape(3, 3, *integrals.shape[1:])
                v = v + np.einsum('caijg,ga->cij', integrals, dipoles[p0:p1])
            op += v
        # 2 order
        if np.any(multipole_orders >= 2):
            idx = np.where(multipole_orders >= 2)[0]
            n_sites = idx.size
            quadrupoles_non_symmetrized = np.array([multipoles[i][4:10] for i in idx])
            quadrupoles = np.zeros((n_sites, 9))
            quadrupoles[:, [0, 1, 2, 4, 5, 8]] = quadrupoles_non_symmetrized
            quadrupoles[:, [0, 3, 6, 4, 7, 8]] += quadrupoles_non_symmetrized
            quadrupoles *= -0.5
            quadrupol_coordinates = multipole_coordinates[idx]
            # Unlike the branches above, this one cannot be batched over sites
            # with a fakemol. The order-0/1 gradients need first and second
            # derivatives, which the three-index family has; the quadrupole
            # gradient needs the third, and every int3c2e* intor stops at second
            # order (ip1, ip2, ip1ip2, ipip1, ipip2, ipspsp1, ipvip1 - there is
            # no int3c2e_ipipip1). So rinv is re-centred on each site in turn,
            # at O(nsites * 27 * nao**2) - the dominant cost for a potential
            # with quadrupolar sites. Recasting it in terms of the available
            # second-derivative integrals is a derivation, not a refactor.
            nao = self.mol.nao
            embedding._blocked_range(self.mol, 1, 2 * 27 * nao * nao * 8,
                                     self.max_memory, 'quadrupole gradient integrals')
            for ii, pos in enumerate(quadrupol_coordinates):
                with self.mol.with_rinv_orig(pos):
                    int1 = self.mol.intor('int1e_ipipiprinv', comp=27).reshape(3, 9, nao, nao)
                    int2 = self.mol.intor('int1e_ipiprinvip', comp=27).reshape(3, 9, nao, nao)
                    quadrupole = quadrupoles[ii]
                    op += np.einsum('caij,a->cij', int1, quadrupole)
                    op += 2.0 * np.einsum('caij,a->cij', int2, quadrupole)
                    # int2 with its three derivative indices cycled, i.e. what
                    # int2.reshape(3, 3, 3, ...).transpose(2, 0, 1, ...) holds.
                    # Contracting the reshaped view directly avoids materialising
                    # that transpose, which is a full 27 * nao**2 copy per site.
                    op += np.einsum('abcji,ab->cij', int2.reshape(3, 3, 3, nao, nao),
                                    quadrupole.reshape(3, 3))
        return op

    def electronic_field_nuclear_gradients(self,
                                           coordinates: np.ndarray,
                                           density_matrix: np.ndarray) -> np.ndarray:
        """Derivatives of the electronic field at the sites, by nucleus.

        This module's own, and not one of IntegralDriverTemplate's: it returns
        every nucleus at once, where the template's
        `electronic_field_nuclear_derivatives` asks for one at a time. The
        template used to carry `electronic_field_gradients` as well, for the
        field gradient tensor at each site, which nothing called and PyFraME has
        since removed.

        Args:
            coordinates: Coordinates the fields are evaluated at.
                Shape: (number of sites, 3)
            density_matrix: Density Matrix that is the source of the field.
                Shape: (number of ao functions, number of ao functions)

        Returns:
            Shape (number of nuclei, 3, number of sites, 3): the derivative of
            the field with respect to each Cartesian component of each nucleus.

        The sites do not move, so a nuclear derivative lands on an atomic
        orbital: on the bra of the field integral, which already carries one
        derivative of its own, or on the ket. What comes back is the derivative
        of the electric field of electronic_fields, the sign of the electron
        charge included, so that it can be added to the derivative of the
        nuclear field and handed to the dipole solver. Moving an orbital's
        centre moves it opposite to the derivative on the orbital, hence the
        subtraction.
        """
        mol = self.mol
        nao = mol.nao
        symmetrized = density_matrix + density_matrix.T
        ao_slices = mol.aoslice_by_atom()
        fields = np.zeros((mol.natm, 3, len(coordinates), 3))
        for p0, p1, integrals in self._aux_e2_blocks('int3c2e_ipip1', coordinates, comp=9):
            integrals = integrals.reshape(3, 3, nao, nao, -1)
            for ia in range(mol.natm):
                k0, k1 = ao_slices[ia, 2:]
                fields[ia, :, p0:p1] -= np.einsum(
                    'xaijg,ij->xga', integrals[:, :, k0:k1], symmetrized[k0:k1])
        for p0, p1, integrals in self._aux_e2_blocks('int3c2e_ipvip1', coordinates, comp=9):
            integrals = integrals.reshape(3, 3, nao, nao, -1)
            for ia in range(mol.natm):
                k0, k1 = ao_slices[ia, 2:]
                # the derivative on the ket, so the field's own index comes
                # first and the nuclear one second
                fields[ia, :, p0:p1] -= np.einsum(
                    'axijg,ij->xga', integrals[:, :, :, k0:k1], symmetrized[:, k0:k1])
        return fields

    def induced_fock_matrix_contributions_gradient(self,
                                                   multipole_coordinates: np.ndarray,
                                                   induced_dipoles: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the induced Fock-matrix contributions.

        Args:
            multipole_coordinates: Coordinates of the Multipoles.
                Shape: (number of atoms, 3)
                Dtype: np.float64
            induced_dipoles: Induced dipoles on the Multipoles.
                Shape: (number of atoms, 3)
                Dtype: np.float64

        Returns:
            Gradient of induced Fock-matrix contributions.
                Shape: (number of nuclei, number of ao functions, number of ao functions)
                Dtype: np.float64
        """
        v = 0
        for p0, p1, integrals in self._aux_e2_blocks('int3c2e_ipip1', multipole_coordinates, comp=9):
            integrals = integrals.reshape(3, 3, *integrals.shape[1:])
            v = v + np.einsum('caijg,ga->cij', integrals, induced_dipoles[p0:p1])
        for p0, p1, integrals in self._aux_e2_blocks('int3c2e_ipvip1', multipole_coordinates, comp=9):
            integrals = integrals.reshape(3, 3, *integrals.shape[1:])
            v = v + np.einsum('caijg,ga->cij', integrals, induced_dipoles[p0:p1])
        # same sign convention as the induced dipoles potential integrals of
        # the integral driver of the energy
        return -v


def make_grad_object(grad_method):
    '''Create a nuclear gradients object including the embedding contributions.

    Args:
        grad_method : the gradients object of an embedding-attached method, or
            the embedding-attached method itself.
    '''
    if isinstance(grad_method, rhf_grad.GradientsBase):
        base_method = grad_method.base
    else:
        # For symmetry with pyscf.solvent, whose make_grad_object takes the
        # method rather than its gradients object.
        base_method = grad_method
    if not isinstance(base_method, _Embedding):
        raise TypeError('%s does not carry an embedding potential.' % base_method)
    if not isinstance(base_method, scf.hf.SCF):
        raise NotImplementedError(
            'Embedding gradients only implemented for SCF methods.')

    # Build the gradients in vacuum from the bare method, so that any other
    # dynamic corrections on base_method are preserved, then point it back at
    # the embedding-attached method.
    vac_grad = base_method.undo_embedding().Gradients()
    vac_grad.base = base_method
    name = base_method.with_embedding.__class__.__name__ + vac_grad.__class__.__name__
    return lib.set_class(EmbeddingGrad(vac_grad),
                         (EmbeddingGrad, vac_grad.__class__), name)


class EmbeddingGrad:
    _keys = {'de_classical_subsystem', 'de_quantum_subsystem'}

    def __init__(self, grad_method):
        self.__dict__.update(grad_method.__dict__)
        self.de_classical_subsystem = None
        self.de_quantum_subsystem = None

    def undo_embedding(self):
        '''Revert to the gradients of the bare method.'''
        cls = self.__class__
        name_mixin = self.base.with_embedding.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, EmbeddingGrad, name_mixin))
        del obj.de_classical_subsystem
        del obj.de_quantum_subsystem
        return obj

    def to_gpu(self):
        # See SCFWithEmbedding.to_gpu.
        raise NotImplementedError(
            'Embedding has no GPU implementation. Call '
            '.undo_embedding().to_gpu() to move the bare gradients to the GPU.')

    def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        '''Nuclear gradients of the embedded SCF energy.

        Kwargs:
            dm : the density matrix the embedding contributions are evaluated
                with. Defaults to the density matrix of the converged base
                method.
            atmlst : the atoms the gradients are computed for. Defaults to all
                atoms. Pass it as a keyword argument; the base method picks it
                up through self.atmlst.
        '''
        if dm is None:
            dm = self.base.make_rdm1(ao_repr=True)
        if atmlst is not None:
            self.atmlst = atmlst
        atmlst = self.atmlst

        # The embedding contributions are always evaluated for all atoms
        # and reduced to atmlst afterwards.
        de_classical = kernel(self.base.with_embedding, dm)
        if atmlst is not None:
            de_classical = de_classical[list(atmlst)]
        self.de_classical_subsystem = de_classical
        self.de_quantum_subsystem = super().kernel(*args, **kwargs)
        self.de = self.de_quantum_subsystem + self.de_classical_subsystem

        if self.verbose >= logger.NOTE:
            logger.note(self, '--------------- %s (+%s) gradients ---------------',
                        self.base.__class__.__name__,
                        self.base.with_embedding.__class__.__name__)
            rhf_grad._write(self, self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')
        return self.de

    def _finalize(self):
        # disable _finalize. It is called in grad_method.kernel method
        # where self.de was not yet initialized.
        pass


# The symmetric 3x3 field gradient PyFraME packs into six components, as the
# indices that unpack it: xx, xy, xz, yy, yz, zz.
_SYMMETRIC_3X3 = np.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])


def perturbed_induced_dipoles(embedding_obj, dm):
    """The response of the induced dipoles to a nuclear displacement.

    Args:
        embedding_obj : the PolarizableEmbedding carrying the potential.
        dm : the density matrix the dipoles were induced by.

    Returns:
        Shape (natm, 3, number of sites, 3).

    The dipoles solve mu = B^-1 F, and B is a property of the sites alone, so
    displacing a nucleus moves them only through the field: mu^A = B^-1 F^A.
    PyFraME's perturbed-dipole solver applies B^-1 to a field of its own without
    touching the dipoles the subsystem holds, which is what _B_dot_x uses it
    for as well.
    """
    quantum_subsystem = embedding_obj.quantum_subsystem
    classical_subsystem = embedding_obj.classical_subsystem
    coordinates = classical_subsystem.coordinates

    driver = EmbeddingGradientIntegralDriver(molecule=embedding_obj.mol)
    fields = driver.electronic_field_nuclear_gradients(coordinates, dm)
    # compute_nuclear_field_gradients differentiates the field with respect to
    # the sites, which is the negative of the derivative with respect to the
    # nucleus.
    nuclear = quantum_subsystem.compute_nuclear_field_gradients(coordinates=coordinates)
    fields = fields - np.take(nuclear, _SYMMETRIC_3X3, axis=2).swapaxes(1, 2)

    # as in _B_dot_x: a perturbed solve keeps nothing on the subsystem
    perturbed = np.empty_like(fields)
    for ia in range(embedding_obj.mol.natm):
        for component in range(3):
            perturbed[ia, component] = classical_subsystem.solve_perturbed_induced_dipoles(
                external_fields=fields[ia, component],
                threshold=embedding_obj._threshold,
                max_iterations=embedding_obj._max_iterations,
                solver=embedding_obj._solver)
    return perturbed


def _potential_interaction_gradient(embedding_obj, difference, frozen_dm):
    """The gradient of Tr(difference . v), for a potential frozen at frozen_dm.

    The term that separates the energy a frozen model reports from the
    environment's own energy; see EmbeddingBase._frozen_energy. Unlike the
    energy gradient, the response of the induced dipoles to the displacement
    does not cancel here -- what is being differentiated is the potential, not
    a quantity stationary in the dipoles.
    """
    mol = embedding_obj.mol
    classical_subsystem = embedding_obj.classical_subsystem
    driver = EmbeddingGradientIntegralDriver(molecule=mol)

    f_es_grad = electrostatic_interactions.es_fock_matrix_gradient_contributions(
        classical_subsystem=classical_subsystem, integral_driver=driver)
    de = _grad_from_operator(mol, f_es_grad, difference)

    if embedding_obj.with_induction:
        f_ind_grad = induction_interactions.induced_fock_matrix_contributions_gradient(
            classical_subsystem=classical_subsystem, integral_driver=driver)
        de += _grad_from_operator(mol, f_ind_grad, difference)
        perturbed = perturbed_induced_dipoles(embedding_obj, frozen_dm)
        for ia in range(mol.natm):
            for component in range(3):
                operator = embedding_obj._integral_driver.induced_dipoles_potential_integrals(
                    induced_dipoles=perturbed[ia, component],
                    coordinates=classical_subsystem.coordinates)
                de[ia, component] += np.einsum('ij,ij->', operator, difference)
    return de


def kernel(embedding_obj, dm, verbose=None):
    '''Nuclear gradients of the embedding contributions to the energy.

    Args:
        embedding_obj : the PolarizableEmbedding or ElectrostaticEmbedding
            carrying the potential.
        dm : the density matrix, either closed-shell or a pair of spin blocks,
            which is traced over spin.

    Returns:
        The gradient with respect to every atom of the molecule, shape
        (natm, 3).

    A frozen model contributes the environment's own energy at the density it
    is frozen at, plus the interaction of the difference between that density
    and this one with the potential -- the two halves of what
    EmbeddingBase.kernel reports.
    '''
    if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
        dm = dm[0] + dm[1]

    if embedding_obj.frozen:
        frozen_dm = embedding_obj._dm
        return (_energy_gradient(embedding_obj, frozen_dm)
                + _potential_interaction_gradient(embedding_obj, dm - frozen_dm,
                                                  frozen_dm))
    return _energy_gradient(embedding_obj, dm)


def _energy_gradient(embedding_obj, dm):
    '''The gradient of the embedding energy of a density matrix.'''
    mol = embedding_obj.mol
    natoms = mol.natm
    de = np.zeros((natoms, 3))
    quantum_subsystem = embedding_obj.quantum_subsystem
    classical_subsystem = embedding_obj.classical_subsystem
    e_es_nuc_grad = electrostatic_interactions.compute_electrostatic_nuclear_gradients(
        quantum_subsystem=quantum_subsystem,
        classical_subsystem=classical_subsystem)
    integral_driver = EmbeddingGradientIntegralDriver(molecule=mol)
    f_es_grad = electrostatic_interactions.es_fock_matrix_gradient_contributions(
        classical_subsystem=classical_subsystem,
        integral_driver=integral_driver)
    e_es_el_grad = _grad_from_operator(mol, f_es_grad, dm)
    if embedding_obj.with_induction:
        # The induced dipoles held by the classical subsystem are those of the
        # last density matrix passed to PolarizableEmbedding.kernel. When that is
        # the one the gradient is being evaluated for -- the usual case, a
        # gradient right after a converged SCF -- re-solving would reproduce them
        # exactly, so both the field build and the iterative solve can be
        # skipped.
        if embedding_obj._dm is None or not np.array_equal(dm, embedding_obj._dm):
            el_fields = quantum_subsystem.compute_electronic_fields(coordinates=classical_subsystem.coordinates,
                                                                    density_matrix=dm,
                                                                    integral_driver=embedding_obj._integral_driver)
            nuc_fields = quantum_subsystem.compute_nuclear_fields(classical_subsystem.coordinates)
            classical_subsystem.solve_induced_dipoles(external_fields=(el_fields + nuc_fields),
                                                      threshold=embedding_obj._threshold,
                                                      max_iterations=embedding_obj._max_iterations,
                                                      solver=embedding_obj._solver)
        nuc_field_grad = quantum_subsystem.compute_nuclear_field_gradients(coordinates=classical_subsystem.coordinates)
        f_ind_grad = induction_interactions.induced_fock_matrix_contributions_gradient(
            classical_subsystem=classical_subsystem,
            integral_driver=integral_driver)
        e_ind_el_grad = _grad_from_operator(mol, f_ind_grad, dm)
        e_ind_nuc_grad = induction_interactions.compute_induction_energy_gradient(
            induced_dipoles=classical_subsystem.induced_dipoles.induced_dipoles,
            total_field_gradients=nuc_field_grad)
    else:
        # No induced dipoles, so the induction terms are identically zero and
        # their integrals are not built at all.
        e_ind_el_grad = 0.0
        e_ind_nuc_grad = 0.0
    if embedding_obj.vdw_method is not None:
        e_rep_grad = repulsion_interactions.compute_repulsion_interactions_gradient(
            quantum_subsystem=quantum_subsystem,
            classical_subsystem=classical_subsystem,
            method=embedding_obj.vdw_method,
            combination_rule=embedding_obj.vdw_combination_rule)
        e_disp_grad = dispersion_interactions.compute_dispersion_interactions_gradient(
            quantum_subsystem=quantum_subsystem,
            classical_subsystem=classical_subsystem,
            method=embedding_obj.vdw_method,
            combination_rule=embedding_obj.vdw_combination_rule)
    else:
        e_rep_grad = 0.0
        e_disp_grad = 0.0
    de += e_ind_nuc_grad + e_ind_el_grad + e_es_nuc_grad + e_es_el_grad + e_rep_grad + e_disp_grad
    return de


def _grad_from_operator(mol, op, dm):
    '''Contract an operator gradient with the density matrix, atom by atom.

    For each atom the operator is restricted to the AOs centred on it and
    symmetrized. Contracting R + R.T with dm is the same as contracting R with
    dm + dm.T, so the symmetrization can be moved onto the density matrix, whose
    restriction to the atom's AOs is far smaller than the full (3, nao, nao)
    copy the operator would need. Holds for any dm, symmetric or not.
    '''
    natoms = mol.natm
    ao_slices = mol.aoslice_by_atom()
    grad = np.zeros((natoms, 3))
    dm_sym = dm + dm.T
    for ia in range(natoms):
        k0, k1 = ao_slices[ia, 2:]
        grad[ia] -= np.einsum('xpq,pq->x', op[:, k0:k1], dm_sym[k0:k1])
    return grad
