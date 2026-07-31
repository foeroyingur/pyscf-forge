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
The integrals PyFraME asks of PySCF, for energies, gradients and Hessians.

PyFraME evaluates everything between the environment and the quantum region
except the one-electron integrals, which it takes from the host through the
methods of its IntegralDriverTemplate. EmbeddingIntegralDriver is that driver
for one Mole: one is made per geometry and handed to the model, which asks it
for whatever the calculation needs.

The MM sites do not move with the nuclei, so every nuclear derivative lands on
an atomic orbital. Where PyFraME asks for a derivative nucleus by nucleus the
template also has a batched form, and the integrals here serve every nucleus at
once, so it is the batched forms that are implemented.
'''

from __future__ import annotations

import numpy as np

try:
    from pyframe.embedding.integral_driver_template import IntegralDriverTemplate
except ImportError as err:
    raise ImportError(
        'Unable to import PyFraME. Please install PyFraME.') from err

from pyscf import __config__
from pyscf import lib
from pyscf import gto
from pyscf import df


# Fraction of max_memory the integral drivers may retain as cached three-index
# integrals. The rest is left for the SCF itself and for the block being built.
CACHE_FRACTION = getattr(__config__, 'embedding_cache_fraction', 0.4)


def _blocked_range(mol, nsites, nbytes_per_site, max_memory, description):
    '''Block a site loop so that one block of integrals fits in memory.

    Returns the block size. Raises when not even a single site fits, which is
    far more useful than the MemoryError that would otherwise come out of the
    integral build.
    '''
    available = max_memory - lib.current_memory()[0]
    blksize = int(available * 1e6 / nbytes_per_site)
    if blksize < 1:
        raise MemoryError(
            'Not enough memory for the %s of a single site: %.0f MB required, '
            '%.0f MB of the %.0f MB budget available. Raise mol.max_memory, or '
            'reduce the basis set or the size of the embedding potential.'
            % (description, nbytes_per_site / 1e6, available, max_memory))
    return min(nsites, blksize)


def _symmetrized_quadrupoles(multipoles, idx):
    '''The quadrupoles of the sites idx as full 3x3 tensors, flattened.

    PyFraME hands over the six unique components, weighted by their degeneracy
    and Taylor coefficients; the operator wants the symmetric tensor, halved.
    '''
    raw = np.array([multipoles[i][4:10] for i in idx])
    quadrupoles = np.zeros((idx.size, 9))
    quadrupoles[:, [0, 1, 2, 4, 5, 8]] = raw
    quadrupoles[:, [0, 3, 6, 4, 7, 8]] += raw
    quadrupoles *= -0.5
    return quadrupoles


class EmbeddingIntegralDriver(IntegralDriverTemplate):
    # Subclassed for the check it buys: the template's three abstract methods
    # are the ones an energy needs, so a rename on PyFraME's side stops the
    # driver being made rather than surfacing mid-SCF. The template also
    # derives each per-nucleus method from its batched form, which is the form
    # implemented here.

    def __init__(self, molecule):
        self.mol = molecule
        self.max_memory = molecule.max_memory
        # Three-index integrals, keyed by (intor, coordinates). The methods
        # below evaluate the same intor at different sets of sites -- all
        # classical sites, the sites carrying dipoles, the polarizable sites --
        # so one slot per intor would be invalidated on every call and the
        # integrals would be rebuilt several times per SCF iteration.
        self._integrals = {}
        self._cached_bytes = 0

    def _aux_e2_blocks(self, intor, coordinates, comp):
        '''Yield (p0, p1, integrals) over blocks of sites.

        The whole array is built once and cached when it fits the cache budget,
        in which case a single block spanning every site is yielded. Otherwise
        the integrals are rebuilt in blocks on each call and nothing is cached,
        which keeps the peak memory bounded at the cost of the recomputation.

        The cache is keyed on (intor, coordinates) only: the integrals also
        depend on self.mol, but that is fixed for the lifetime of the driver --
        a new driver is created whenever the molecule changes.
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
        cache_budget = self.max_memory * CACHE_FRACTION * 1e6
        if self._cached_bytes + nsites * nbytes_per_site <= cache_budget:
            integrals = self._build(intor, coordinates, comp)
            self._integrals[key] = integrals
            self._cached_bytes += integrals.nbytes
            yield 0, nsites, integrals
            return

        blksize = _blocked_range(self.mol, nsites, nbytes_per_site,
                                 self.max_memory, '%s integrals' % intor)
        for p0, p1 in lib.prange(0, nsites, blksize):
            yield p0, p1, self._build(intor, coordinates[p0:p1], comp)

    def _build(self, intor, coordinates, comp):
        fakemol = gto.fakemol_for_charges(coordinates)
        return df.incore.aux_e2(self.mol, fakemol, intor=intor, comp=comp)

    # -- energy ------------------------------------------------------------

    def electronic_fields(self,
                          coordinates: np.ndarray,
                          density_matrix: np.ndarray) -> np.ndarray:
        """Calculate the electronic fields on coordinates.

        Args:
            coordinates: Coordinates on which the fields are to be evaluated.
                Shape: (number of atoms, 3)
                Dtype: np.float64
            density_matrix: Density Matrix that is the source of the electronic field.
                Shape: (number of ao functions, number of ao functions)
                Dtype: np.float64

        Returns:
            Electronic fields. Shape: (number of atoms, 3) Dtype: np.float64.

        Note:
            The electric field of the electrons, E = -grad phi with the
            electron charge included, as PyFraME's IntegralDriverTemplate
            defines it; PyFraME takes it as it is.
        """
        fields = np.empty((len(coordinates), 3))
        for p0, p1, integrals in self._aux_e2_blocks('int3c2e_ip1', coordinates, comp=3):
            fields[p0:p1] = (np.einsum('aijg,ij->ga', integrals, density_matrix)
                             + np.einsum('aijg,ji->ga', integrals, density_matrix))
        return fields

    def multipole_potential_integrals(self,
                                      multipole_coordinates: np.ndarray,
                                      multipole_orders: np.ndarray,
                                      multipoles: list[np.ndarray]) -> np.ndarray:
        """Calculate the electronic potential integrals and multiply with the multipoles.

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
            Product of electronic potential integrals and multipoles.
                Shape: (number of ao functions, number of ao functions)
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
        for p0, p1, integrals in self._aux_e2_blocks('int3c2e', charge_coordinates, comp=1):
            op -= np.einsum('ijg,ga->ij', integrals, charges[p0:p1])
        # 1 order
        if np.any(multipole_orders >= 1):
            idx = np.where(multipole_orders >= 1)[0]
            dipole_coordinates = multipole_coordinates[idx]
            dipoles = np.array([multipoles[i][1:4] for i in idx])
            v = 0
            for p0, p1, integrals in self._aux_e2_blocks('int3c2e_ip1', dipole_coordinates, comp=3):
                v = v + np.einsum('aijg,ga->ij', integrals, dipoles[p0:p1])
            op += v + v.T
        # 2 order
        if np.any(multipole_orders >= 2):
            idx = np.where(multipole_orders >= 2)[0]
            quadrupol_coordinates = multipole_coordinates[idx]
            quadrupoles = _symmetrized_quadrupoles(multipoles, idx)
            v = 0
            for p0, p1, integrals in self._aux_e2_blocks('int3c2e_ipip1', quadrupol_coordinates, comp=9):
                v = v + np.einsum('aijg,ga->ij', integrals, quadrupoles[p0:p1])
            op += v + v.T
            for p0, p1, integrals in self._aux_e2_blocks('int3c2e_ipvip1', quadrupol_coordinates, comp=9):
                op += np.einsum('aijg,ga->ij', integrals, quadrupoles[p0:p1]) * 2
        return op

    def induced_dipoles_potential_integrals(self,
                                            induced_dipoles: np.ndarray,
                                            coordinates: np.ndarray) -> np.ndarray:
        """Calculate the electronic potential integrals and contract with the induced dipoles of Atoms.

        Args:
            induced_dipoles: Induced dipoles
                Shape (number of induced dipoles, 3)
                Dtype: np.float64
            coordinates: Coordinates of the induced dipoles on which the integrals are to be evaluated.
                Shape (number of induced dipoles, 3)
                Dtype: np.float64

        Returns:
            Product of the electronic potential integrals and the induced dipoles.
                Shape: (number of ao functions, number of ao functions)
                Dtype: np.float64
        """
        f_el_ind = 0
        for p0, p1, integrals in self._aux_e2_blocks('int3c2e_ip1', coordinates, comp=3):
            f_el_ind = f_el_ind - np.einsum('aijg,ga->ij', integrals, induced_dipoles[p0:p1])
        return f_el_ind + f_el_ind.T

    # -- gradient ----------------------------------------------------------

    def electronic_electrostatic_energy_gradients(self, multipole_coordinates,
                                                  multipole_orders, multipoles,
                                                  density_matrix):
        '''The gradient of Tr(dm V_es) at a fixed density, shape (natm, 3).'''
        op = self.multipole_potential_gradient_integrals(
            multipole_coordinates, np.asarray(multipole_orders), multipoles)
        return _grad_from_operator(self.mol, op, density_matrix)

    def electronic_induction_energy_gradients(self, induced_dipoles, coordinates,
                                              density_matrix):
        '''The gradient of Tr(dm V_ind) at fixed dipoles, i.e. -mu.dF_el/dR.'''
        op = self.induced_fock_matrix_contributions_gradient(coordinates,
                                                             induced_dipoles)
        return _grad_from_operator(self.mol, op, density_matrix)

    def multipole_potential_gradient_integrals(self,
                                               multipole_coordinates: np.ndarray,
                                               multipole_orders: np.ndarray,
                                               multipoles: list[np.ndarray]) -> np.ndarray:
        """Calculate the gradient of the electronic potential integrals and multiply with the multipoles.

        This driver's own, not one of IntegralDriverTemplate's: what it
        returns is in PySCF's layout, which the template methods above and
        below turn into gradients and Fock matrix derivatives.

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
            <d_x p|V|q>, the operator of multipole_potential_integrals with the
            derivative on the bra, over all AOs; see _grad_from_operator and
            _operator_derivative for how it becomes a derivative by nucleus.
                Shape: (3, number of ao functions, number of ao functions)
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
            quadrupoles = _symmetrized_quadrupoles(multipoles, idx)
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
            _blocked_range(self.mol, 1, 2 * 27 * nao * nao * 8,
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

    def all_electronic_field_nuclear_derivatives(self,
                                                 coordinates: np.ndarray,
                                                 density_matrix: np.ndarray) -> np.ndarray:
        """Derivatives of the electronic field at the sites, by nucleus.

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

        This driver's own, not one of IntegralDriverTemplate's; see
        multipole_potential_gradient_integrals.

        Args:
            multipole_coordinates: Coordinates of the Multipoles.
                Shape: (number of atoms, 3)
                Dtype: np.float64
            induced_dipoles: Induced dipoles on the Multipoles.
                Shape: (number of atoms, 3)
                Dtype: np.float64

        Returns:
            <d_x p|V_ind|q>, the operator of induced_dipoles_potential_integrals
            with the derivative on the bra, over all AOs.
                Shape: (3, number of ao functions, number of ao functions)
                Dtype: np.float64
        """
        v = 0
        for p0, p1, integrals in self._aux_e2_blocks('int3c2e_ipip1', multipole_coordinates, comp=9):
            integrals = integrals.reshape(3, 3, *integrals.shape[1:])
            v = v + np.einsum('caijg,ga->cij', integrals, induced_dipoles[p0:p1])
        for p0, p1, integrals in self._aux_e2_blocks('int3c2e_ipvip1', multipole_coordinates, comp=9):
            integrals = integrals.reshape(3, 3, *integrals.shape[1:])
            v = v + np.einsum('caijg,ga->cij', integrals, induced_dipoles[p0:p1])
        # same sign convention as induced_dipoles_potential_integrals
        return -v

    # -- Hessian -----------------------------------------------------------

    def full_electronic_electrostatic_energy_hessian(self, multipole_coordinates,
                                                     multipole_orders, multipoles,
                                                     density_matrix):
        '''The Hessian of Tr(dm V_es) at a fixed density, (3 natm, 3 natm).'''
        natm = self.mol.natm
        hess = self._operator_hessian(multipole_coordinates, multipole_orders,
                                      multipoles, density_matrix)
        return hess.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)

    def electronic_field_nuclear_hessian(self, coordinates, induced_dipoles,
                                         density_matrix):
        '''mu . d2F_el/dR dR', shape (3 natm, 3 natm).

        -mu.F_el is Tr(dm V_ind), and the induced-dipole operator is the order-1
        multipole operator of -mu (see _dipole_multipoles), so this is minus
        the second derivative of that trace.
        '''
        natm = self.mol.natm
        hess = self._operator_hessian(coordinates,
                                      np.ones(len(coordinates), dtype=np.int64),
                                      _dipole_multipoles(induced_dipoles),
                                      density_matrix)
        return -hess.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)

    def all_electronic_electrostatic_fock_gradients(self, multipole_coordinates,
                                                    multipole_orders, multipoles):
        '''The derivative of the operator of multipole_potential_integrals by every nucleus.'''
        return _operator_derivative(
            self.mol, self.multipole_potential_gradient_integrals(
                multipole_coordinates, np.asarray(multipole_orders), multipoles))

    def all_electronic_induction_fock_gradients(self, induced_dipoles, coordinates):
        '''The derivative of the operator of induced_dipoles_potential_integrals by every nucleus.'''
        return _operator_derivative(
            self.mol, self.induced_fock_matrix_contributions_gradient(
                coordinates, induced_dipoles))

    def _operator_hessian(self, multipole_coordinates, multipole_orders,
                          multipoles, density_matrix):
        '''The second derivative of Tr(dm V) for the multipoles, (natm, natm, 3, 3).'''
        bra_bra, bra_ket = self.multipole_potential_hessian_integrals(
            multipole_coordinates, multipole_orders, multipoles)
        return _hess_from_operators(self.mol, bra_bra, bra_ket, density_matrix,
                                    range(self.mol.natm))

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
            multipole_potential_integrals builds.

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
            _blocked_range(self.mol, 1, 2 * 27 * nao * nao * 8,
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
            quadrupoles = _symmetrized_quadrupoles(multipoles, idx)
            _blocked_range(self.mol, 1, 3 * 81 * nao * nao * 8,
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


def _operator_derivative(mol, op):
    '''Turn <d_x p|O|q> into the derivative of O with respect to every nucleus.

    The counterpart of _grad_from_operator, which is the same thing already
    contracted with a density matrix: what comes back here is the operator
    itself, shape (natm, 3, nao, nao), for the Fock matrix derivative the CPHF
    equations are solved against.
    '''
    ao_slices = mol.aoslice_by_atom()
    derivatives = np.empty((mol.natm,) + op.shape)
    for ia in range(mol.natm):
        p0, p1 = ao_slices[ia, 2:]
        derivative = np.zeros_like(op)
        derivative[:, p0:p1] -= op[:, p0:p1]
        derivatives[ia] = derivative + derivative.transpose(0, 2, 1)
    return derivatives


def _dipole_multipoles(dipoles):
    '''Dipoles as the multipoles whose potential operator is theirs.

    The induced-dipole operator is the order-1 multipole operator of -mu, the
    sign being the Taylor coefficient the permanent multipoles carry and the
    induced dipoles do not (IntegralDriverTemplate.induced_dipoles_potential_integrals),
    so the multipole integrals serve both.
    '''
    return [np.concatenate(([0.0], -dipole)) for dipole in dipoles]
