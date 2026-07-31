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

import os
import warnings

import numpy as np

try:
    from pyframe.embedding import (read_input, electrostatic_interactions, induction_interactions,
                                   repulsion_interactions, dispersion_interactions)
except ImportError as err:
    raise ImportError(
        'Unable to import PyFraME. Please install PyFraME.') from err

from pyscf import __config__
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import df
from pyscf.embedding import _attach_embedding


@lib.with_doc(_attach_embedding._for_scf.__doc__)
def embedding_for_scf(mf, solvent_obj, cls=None):
    if cls is None:
        cls = PolarizableEmbedding
    if not isinstance(solvent_obj, EmbeddingBase):
        solvent_obj = cls(mf.mol, solvent_obj)
    return _attach_embedding._for_scf(mf, solvent_obj)


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


class EmbeddingIntegralDriver:

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
            integrals = self._build(intor, coordinates)
            self._integrals[key] = integrals
            self._cached_bytes += integrals.nbytes
            yield 0, nsites, integrals
            return

        blksize = _blocked_range(self.mol, nsites, nbytes_per_site,
                                 self.max_memory, '%s integrals' % intor)
        for p0, p1 in lib.prange(0, nsites, blksize):
            yield p0, p1, self._build(intor, coordinates[p0:p1])

    def _build(self, intor, coordinates):
        fakemol = gto.fakemol_for_charges(coordinates)
        return df.incore.aux_e2(self.mol, fakemol, intor=intor)

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
            The sign of the electron charge is applied by PyFraME in
            QuantumSubsystem.compute_electronic_fields, so the integrals are
            contracted here without it.
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
            n_sites = idx.size
            quadrupoles_non_symmetrized = np.array([multipoles[i][4:10] for i in idx])
            quadrupoles = np.zeros((n_sites, 9))
            quadrupoles[:, [0, 1, 2, 4, 5, 8]] = quadrupoles_non_symmetrized
            quadrupoles[:, [0, 3, 6, 4, 7, 8]] += quadrupoles_non_symmetrized
            quadrupoles *= -0.5
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


class EmbeddingBase(lib.StreamObject):
    '''A molecular mechanics (MM) embedding potential attached to a molecule.

    Everything a model is built from and nothing a model is: the options, the
    MM subsystems and the geometry they are kept in step with, the integral
    driver, the van der Waals interaction with the environment that any model
    may ask for, and the bookkeeping that kernel and the energy report run on.
    What the environment does to the density is left to the subclasses -- here
    it does nothing -- so this class is not a model and is not meant to be
    instantiated on its own.

    A model is written as the terms it adds, each method calling up to its
    parent so that the terms accumulate:

    _compute_static_contributions
        whatever the model can evaluate without a density matrix
    _density_dependent_contributions
        the energy and the Fock matrix contribution of the model, for a density
        matrix, added to those of the parent
    _energy_terms
        the labelled energies of the model, for reporting
    '''

    _keys = {'mol', 'comm', 'options', 'classical_subsystem', 'quantum_subsystem',
             'e', 'v', 'vdw_method', 'vdw_combination_rule'}

    # Whether the polarizabilities of the potential are used, i.e. whether the
    # induced dipoles are solved for. Read by the gradient as well, which skips
    # the induction terms of the gradient when they are not.
    with_induction = False

    # Recognized keys of the options dictionary. Unknown keys are rejected so
    # that a typo does not silently disable a requested contribution. A model
    # that takes options of its own extends the set.
    _valid_option_keys = {'json_file', 'vdw', 'environment_energy'}
    _valid_vdw_keys = {'method', 'combination_rule'}

    def __init__(self, molecule, options_or_json_file):
        self.stdout = molecule.stdout
        self.verbose = molecule.verbose
        # PyFraME distributes the environment over an MPI communicator when it
        # is given one. This interface always runs the environment serially.
        self.comm = None
        # e (the embedding energy) and v (the additional potential) are
        # updated during the SCF iterations
        self.e = None
        self.v = None
        # the density matrix e and v were last computed for
        self._dm = None
        self.vdw_method = None
        self.vdw_combination_rule = None

        self.mol = molecule
        self.max_memory = molecule.max_memory
        self._set_options(options_or_json_file)
        self._create_mm_subsystems()
        self._integral_driver = EmbeddingIntegralDriver(molecule=self.mol)
        self._compute_static_contributions()
        self._static_contributions_stale = False

    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s flags ********', self.__class__)
        for key in self.options.keys():
            logger.info(self, "mm.%s = %s", key, self.options[key])
        return self

    def reset(self, mol=None, options_or_json_file=None):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        if mol is not None:
            self.mol = mol
        if options_or_json_file is not None:
            # A different potential was given, so the MM subsystems have to be
            # built from it.
            self._set_options(options_or_json_file)
            self._create_mm_subsystems()
        else:
            # The potential is unchanged and the classical subsystem is fixed;
            # only the geometry of the quantum subsystem follows the Mole
            # object. Re-reading the potential file would rebuild identical
            # objects, which dominates the cost of a geometry optimization step
            # once the potential is of a realistic size.
            self._sync_quantum_subsystem()
        # the integrals of the driver are those of the previous geometry
        self._integral_driver = EmbeddingIntegralDriver(molecule=self.mol)
        # The static contributions belong to the previous geometry. They are
        # rebuilt on the next kernel rather than here, so that a reset costs
        # nothing until the object is used again.
        self._static_contributions_stale = True
        self._e_rep = None
        self._e_disp = None
        self._dm = None
        self.e = None
        self.v = None
        return self

    def kernel(self, dm):
        '''Embedding energy and Fock matrix contribution for a density matrix.

        Args:
            dm : the density matrix, either closed-shell or a pair of spin
                blocks, which is traced over spin.

        Returns:
            (e, v), the embedding energy and the potential to be added to the
            Fock matrix. Both are also stored on the object.
        '''
        if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
            # spin-traced DM for UHF or ROHF
            dm = dm[0] + dm[1]
        dm = np.asarray(dm)
        # Recorded so that the gradient can tell whether the induced dipoles
        # held by the classical subsystem already belong to the density matrix
        # it was handed.
        self._dm = dm.copy()
        if self._static_contributions_stale:
            self._compute_static_contributions()
            self._static_contributions_stale = False
        e, v = self._density_dependent_contributions(dm)
        self.e = e + self._e_rep + self._e_disp
        self.v = v
        return self.e, self.v

    def _density_dependent_contributions(self, density_matrix):
        '''The energy and Fock matrix contribution of the model.

        Returns:
            (e, v), summed with those of the parent class by every model that
            adds a term. An environment that only carries Lennard-Jones sites
            contributes neither, which is what this class describes.
        '''
        return 0.0, 0.0

    def nuc_grad_method(self, grad_method):
        from pyscf.embedding import embedding_gradient
        return embedding_gradient.make_grad_object(grad_method)

    def energy_contributions(self):
        '''The embedding energy broken down by contribution, for reporting.

        Only the entries that sum to self.e are part of the total energy. The
        internal energy of the environment is reported alongside them when the
        environment_energy option is set, but it is a constant offset that no
        term of the total energy depends on, so it is labelled as excluded.
        '''
        contributions = dict(self._energy_terms())
        if self.vdw_method is not None:
            contributions['Repulsion contribution      (E_rep) '] = self._e_rep
            contributions['Dispersion contribution     (E_disp)'] = self._e_disp
        if self._environment_energy:
            contributions['Environment energy (not in E_tot)   '] = \
                self.environment_energy()
        return contributions

    def _energy_terms(self):
        '''The labelled energy terms of the model, in reporting order.

        The van der Waals and environment terms are those of the potential
        rather than of the model, and are appended by energy_contributions.
        '''
        return []

    def environment_energy(self):
        '''Internal interaction energy of the environment.

        A property of the potential rather than of the model: the sites interact
        among themselves through their multipoles whichever way they are coupled
        to the quantum region. It is a constant for a fixed environment and is
        not part of the total energy that kernel returns; see
        energy_contributions.

        The repulsion and dispersion interactions within the environment are
        only included when van der Waals interactions were requested, since
        the Lennard-Jones parameters are otherwise not part of the embedding
        potential.
        '''
        if self.vdw_method is not None:
            return self.classical_subsystem.environment_energy(
                vdw_method=self.vdw_method,
                vdw_combination_rule=self.vdw_combination_rule)
        return self.classical_subsystem.compute_electrostatic_energy()

    def _set_options(self, options_or_json_file):
        if isinstance(options_or_json_file, (str, os.PathLike)):
            self.options = {"json_file": options_or_json_file}
        else:
            self.options = options_or_json_file
        if not isinstance(self.options, dict):
            raise TypeError("Options should be a dictionary.")
        self._check_option_keys(self.options, self._valid_option_keys, 'options')
        if 'json_file' not in self.options:
            raise ValueError(
                'The options dictionary must contain the key "json_file", '
                'pointing at the MM potential of the embedding.')
        self._parse_options()

    @staticmethod
    def _check_option_keys(options, valid_keys, name):
        unknown = set(options).difference(valid_keys)
        if unknown:
            raise ValueError('Unknown %s: %s. Valid keys are: %s.'
                             % (name, ', '.join(sorted(map(str, unknown))),
                                ', '.join(sorted(valid_keys))))

    def _parse_options(self):
        '''Read the options dictionary onto the object.

        The "vdw" entry takes a method, of which PyFraME implements "LJ", and
        the rule that combines the Lennard-Jones parameters of two sites into a
        pair coefficient. A rule is named "<sigma rule>-<epsilon rule>", i.e.
        Lorentz or Good-Hope for sigma and Berthelot or Fender-Halsey for
        epsilon, and defaults to Lorentz-Berthelot. The powers of the potential
        are not an option here: they are carried by the sites of the potential
        file, and default to the 12-6 form.
        '''
        if 'vdw' in self.options:
            if not isinstance(self.options['vdw'], dict):
                raise TypeError("vdw options should be a dictionary.")
            self._check_option_keys(self.options['vdw'], self._valid_vdw_keys,
                                    'vdw options')
            self.vdw_method = self.options['vdw'].get('method', 'LJ')
            self.vdw_combination_rule = self.options['vdw'].get(
                'combination_rule', 'Lorentz-Berthelot')
        else:
            self.vdw_method = None
            self.vdw_combination_rule = None

        environment_energy = self.options.get('environment_energy', True)
        if not isinstance(environment_energy, bool):
            raise TypeError("environment_energy options should be a bool.")
        self._environment_energy = environment_energy

    def _compute_static_contributions(self):
        '''Compute the contributions that do not depend on the density matrix.'''
        if self.vdw_method is not None:
            self._check_vdw_parameters()
            self._e_rep = repulsion_interactions.compute_repulsion_interactions(
                quantum_subsystem=self.quantum_subsystem,
                classical_subsystem=self.classical_subsystem,
                method=self.vdw_method,
                combination_rule=self.vdw_combination_rule)
            self._e_disp = dispersion_interactions.compute_dispersion_interactions(
                quantum_subsystem=self.quantum_subsystem,
                classical_subsystem=self.classical_subsystem,
                method=self.vdw_method,
                combination_rule=self.vdw_combination_rule)
        else:
            self._e_rep = 0.0
            self._e_disp = 0.0

    def _check_vdw_parameters(self):
        '''Check the Lennard-Jones parameters behind the vdw option.

        A site carrying no parameters contributes nothing rather than ending the
        calculation, so a potential parameterised in part is legitimate -- and
        usual, since hydrogens and virtual sites often carry none. PyFraME warns
        about the sites it skipped; those warnings go to the Python warning
        stream, so they are re-emitted through the PySCF logger here to reach the
        output file with everything else.

        A potential where nothing is parameterised makes the whole vdw
        contribution identically zero, which is quiet enough to be worth refusing
        outright: the option was asked for and would do nothing.
        '''
        epsilons = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            for side, subsystem in (('quantum', self.quantum_subsystem),
                                    ('classical', self.classical_subsystem)):
                for interaction in ('rep', 'disp'):
                    epsilons[side, interaction] = np.asarray(
                        getattr(subsystem, '%s_lj' % interaction).epsilon)
        for warning in caught:
            logger.warn(self, '%s', warning.message)

        # a pair term needs a non-zero epsilon on both sides to survive the
        # combination rule
        if not any(epsilons['quantum', interaction].any()
                   and epsilons['classical', interaction].any()
                   for interaction in ('rep', 'disp')):
            raise ValueError(
                'The "vdw" option was requested, but no site of %s carries '
                'Lennard-Jones parameters on both the quantum and the classical '
                'side, so the repulsion and dispersion contributions would both '
                'be exactly zero. Regenerate the potential with PyFraME including '
                'the parameters, or drop the "vdw" option.'
                % (self.options['json_file'],))

    def _create_mm_subsystems(self):
        '''Read the potential and take the single pair of subsystems from it.'''
        subsystems = read_input.reader(input_data=self.options['json_file'], comm=self.comm)
        if len(subsystems.quantum_subsystems) != 1:
            raise ValueError("Exactly one quantum subsystem is required.")
        if len(subsystems.classical_subsystems) != 1:
            raise ValueError("Exactly one classical subsystem is required.")
        self.quantum_subsystem = subsystems.quantum_subsystems[0]
        self.classical_subsystem = subsystems.classical_subsystems[0]
        self._sync_quantum_subsystem()

    def _sync_quantum_subsystem(self):
        '''Take the geometry of the quantum subsystem from the Mole object.

        The nuclei in the json file describe the system the embedding potential
        was made for, while the Mole object is the one that is being computed
        on, and the one that moves in a scanner or geometry optimization.
        '''
        if self.quantum_subsystem.num_nuclei != self.mol.natm:
            raise ValueError(
                'The quantum subsystem of the json file has %d nuclei, while the '
                'molecule has %d atoms.' % (self.quantum_subsystem.num_nuclei, self.mol.natm))
        if not np.allclose(self.quantum_subsystem.charges, self.mol.atom_charges()):
            logger.warn(self, 'The nuclear charges of the quantum subsystem in the json '
                        'file differ from those of the molecule.')
        self.quantum_subsystem.coordinates = self.mol.atom_coords()


class ElectrostaticEmbedding(EmbeddingBase):
    '''Embedding in the permanent multipoles of a molecular mechanics (MM) potential.

    The sites of the environment interact with the quantum region through their
    permanent multipoles. The polarizabilities the potential may carry are
    ignored: no induced dipoles are solved for, so the environment does not
    respond to the density.
    '''

    def __init__(self, molecule, options_or_json_file):
        self._e_es = None
        self._f_el_es = None
        self._e_nuc_es = None
        super().__init__(molecule, options_or_json_file)

    def reset(self, mol=None, options_or_json_file=None):
        super().reset(mol, options_or_json_file)
        self._e_es = None
        self._f_el_es = None
        self._e_nuc_es = None
        return self

    def _compute_static_contributions(self):
        super()._compute_static_contributions()
        self._f_el_es = electrostatic_interactions.es_fock_matrix_contributions(
            classical_subsystem=self.classical_subsystem,
            integral_driver=self._integral_driver)
        self._e_nuc_es = electrostatic_interactions.compute_electrostatic_nuclear_energy(
            quantum_subsystem=self.quantum_subsystem,
            classical_subsystem=self.classical_subsystem)

    def _energy_terms(self):
        return super()._energy_terms() + [
            ('Electrostatic contribution  (E_es)  ', self._e_es)]

    def _density_dependent_contributions(self, density_matrix):
        '''The interaction of the density with the permanent multipoles.'''
        e, v = super()._density_dependent_contributions(density_matrix)
        self._e_es = self._e_nuc_es + np.einsum('ij,ij->', self._f_el_es,
                                                density_matrix)
        return e + self._e_es, v + self._f_el_es


class PolarizableEmbedding(ElectrostaticEmbedding):
    '''Embedding in the multipoles and induced dipoles of an MM potential.

    The electrostatics of ElectrostaticEmbedding, with the polarizabilities of
    the MM potential put to use on top of them: the dipoles the quantum region
    induces in the environment are solved for at every density matrix, so the
    environment responds and the coupling is self-consistent.
    '''

    with_induction = True

    # The iterative solution of the induced dipoles is configurable.
    _valid_option_keys = ElectrostaticEmbedding._valid_option_keys | {'induced_dipoles'}
    _valid_induced_dipoles_keys = {'threshold', 'max_iterations', 'solver'}

    def __init__(self, molecule, options_or_json_file):
        self._e_ind = None
        super().__init__(molecule, options_or_json_file)

    def reset(self, mol=None, options_or_json_file=None):
        super().reset(mol, options_or_json_file)
        self._e_ind = None
        return self

    def _parse_options(self):
        super()._parse_options()
        induced_dipoles_options = self.options.get('induced_dipoles', {})
        if not isinstance(induced_dipoles_options, dict):
            raise TypeError("induced_dipoles options should be a dictionary.")
        self._check_option_keys(induced_dipoles_options,
                                self._valid_induced_dipoles_keys,
                                'induced_dipoles options')
        self._threshold = induced_dipoles_options.get('threshold', 1e-8)
        self._max_iterations = induced_dipoles_options.get('max_iterations', 100)
        self._solver = induced_dipoles_options.get('solver', 'jacobi')

    def _energy_terms(self):
        return super()._energy_terms() + [
            ('Induction contribution      (E_ind) ', self._e_ind)]

    def _density_dependent_contributions(self, density_matrix):
        '''Solve for the induced dipoles and take their energy and potential.'''
        e, v = super()._density_dependent_contributions(density_matrix)
        el_fields = self.quantum_subsystem.compute_electronic_fields(coordinates=self.classical_subsystem.coordinates,
                                                                     density_matrix=density_matrix,
                                                                     integral_driver=self._integral_driver)
        nuc_fields = self.quantum_subsystem.compute_nuclear_fields(self.classical_subsystem.coordinates)
        self.classical_subsystem.solve_induced_dipoles(external_fields=(el_fields + nuc_fields),
                                                       threshold=self._threshold,
                                                       max_iterations=self._max_iterations,
                                                       solver=self._solver)
        self._e_ind = induction_interactions.compute_induction_energy(
            induced_dipoles=self.classical_subsystem.induced_dipoles.induced_dipoles,
            total_fields=el_fields + nuc_fields + self.classical_subsystem.multipole_fields)
        f_el_ind = induction_interactions.ind_fock_matrix_contributions(classical_subsystem=self.classical_subsystem,
                                                                        integral_driver=self._integral_driver)
        return e + self._e_ind, v - f_el_ind
