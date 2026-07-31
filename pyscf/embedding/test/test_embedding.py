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

import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import pyscf
from pyscf import embedding
from pyscf import dft, mp, qmmm, scf

try:
    # Importing the package above does not pull in PyFraME; the submodule does.
    # Probe this rather than `import pyframe`, which succeeds against released
    # PyFraME 0.4.0, where the `pyframe.embedding` subpackage does not exist.
    from pyscf.embedding import embedding as embedding_impl
    has_pyframe = True
except ImportError:
    has_pyframe = False
    embedding_impl = None

dir_name = os.path.dirname(__file__)
json_file = os.path.join(dir_name, 'butadiene_water.json')

# Butadiene, with the three waters of butadiene_water.json around it.
BUTADIENE = '''
C     32.081000    29.944000    30.110000
C     30.771000    30.374000    29.820000
C     29.571000    29.984000    30.590000
C     28.361000    30.394000    30.239000
H     32.911000    30.224000    29.460000
H     32.181000    29.274000    30.950000
H     30.701000    31.104000    29.020000
H     29.571000    29.434000    31.540000
H     28.241000    31.044000    29.380000
H     27.441000    30.054000    30.720000
'''

# Number of multipole components up to a given order.
_N_COMPONENTS = {0: 1, 1: 4, 2: 10}

# Displacements below are given in Angstrom, since that is what the geometry is
# written in, while a derivative is with respect to Bohr.
BOHR = 0.52917721092


def embedding_gradient_kernel(model, dm):
    '''The embedding part of the gradient, for a model object and a density.'''
    from pyscf.embedding import embedding_gradient
    return embedding_gradient.kernel(model, dm)

# Lennard-Jones parameters used to exercise the vdw option. They are arbitrary
# rather than taken from a force field: the tests below check the summation and
# the combination rule against an independent evaluation of the same closed-form
# expression, which is a statement about the code and not about chemistry, so
# the values only have to be well formed and of a sensible magnitude.
_LJ_SIGMA = {'C': 3.4, 'H': 2.6, 'O': 3.2}
_LJ_EPSILON = {'C': 0.10, 'H': 0.03, 'O': 0.21}

tmpdir = None


def setUpModule():
    global tmpdir
    tmpdir = tempfile.mkdtemp(prefix='pyscf_embedding_test_')


def tearDownModule():
    global tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)
    tmpdir = None


def butadiene(atom=BUTADIENE, **kwargs):
    kwargs.setdefault('basis', 'sto3g')
    kwargs.setdefault('verbose', 0)
    return pyscf.M(atom=atom, **kwargs)


def perturbed_butadiene(atom, component, step):
    '''Butadiene with one Cartesian component of one atom displaced, in Angstrom.'''
    lines = [line.split() for line in BUTADIENE.strip().split('\n')]
    lines[atom][component + 1] = repr(float(lines[atom][component + 1]) + step)
    return butadiene(atom='\n'.join(' '.join(line) for line in lines))


def make_potential(name, orders=None, depolarize=False):
    '''Write a variant of the bundled potential and return its path.

    Args:
        orders : callable mapping a site index to the multipole order to
            truncate that site to. Defaults to leaving every site untouched.
        depolarize : zero every polarizability, which switches the induction
            contribution off.
    '''
    with open(json_file) as handle:
        potential = json.load(handle)
    atoms = potential['classical_subsystems'][0]['classical_fragments'][0]['atoms']
    for index, atom in enumerate(atoms):
        if orders is not None:
            order = orders(index)
            atom['multipoles']['elements'] = \
                atom['multipoles']['elements'][:_N_COMPONENTS[order]]
            atom['multipoles']['order'] = order
        if depolarize:
            atom['polarizabilities']['elements'] = \
                [0.0] * len(atom['polarizabilities']['elements'])
    path = os.path.join(tmpdir, name)
    with open(path, 'w') as handle:
        json.dump(potential, handle)
    return path


def make_vdw_potential(name='vdw.json', parameterised=None, exponents=None):
    '''Write a potential carrying Lennard-Jones parameters.

    Args:
        parameterised : callable deciding, from a particle's element, whether it
            gets parameters at all. Defaults to every particle. A particle
            without them contributes nothing, which PyFraME allows and warns
            about.
        exponents : the powers of the repulsion and the dispersion, as a pair.
            Defaults to the 12-6 form PyFraME assumes for a site that names
            none. A site may carry its own, which is how a potential other than
            12-6 is expressed; the sites of one interaction have to agree on it.
    '''
    def block(element, exponent):
        parameters = {'lj_sigma': _LJ_SIGMA[element],
                      'lj_epsilon': _LJ_EPSILON[element]}
        if exponent is not None:
            parameters['lj_exponent'] = exponent
        return {'method': 'LJ', 'parameters': parameters}

    rep_exponent, disp_exponent = exponents if exponents else (None, None)

    with open(json_file) as handle:
        potential = json.load(handle)
    particles = list(potential['quantum_subsystems'][0]['nuclei'])
    particles += potential['classical_subsystems'][0]['classical_fragments'][0]['atoms']
    for particle in particles:
        if parameterised is None or parameterised(particle['element']):
            particle['repulsion'] = block(particle['element'], rep_exponent)
            particle['dispersion'] = block(particle['element'], disp_exponent)
    path = os.path.join(tmpdir, name)
    with open(path, 'w') as handle:
        json.dump(potential, handle)
    return path


def lennard_jones_parameters(mol, parameterised=None):
    '''Coordinates, sigma and epsilon of the quantum and classical particles.

    Particles the potential gives no parameters carry zeros, which is what
    PyFraME fills in for them.
    '''
    def value(table, element):
        if parameterised is None or parameterised(element):
            return table[element]
        return 0.0

    with open(json_file) as handle:
        potential = json.load(handle)
    atoms = potential['classical_subsystems'][0]['classical_fragments'][0]['atoms']
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    quantum = (mol.atom_coords(),
               np.array([value(_LJ_SIGMA, s) for s in symbols]),
               np.array([value(_LJ_EPSILON, s) for s in symbols]))
    classical = (np.array([atom['coordinate'] for atom in atoms]),
                 np.array([value(_LJ_SIGMA, atom['element']) for atom in atoms]),
                 np.array([value(_LJ_EPSILON, atom['element']) for atom in atoms]))
    return quantum, classical


def combine_sigma(rule, qsigma, csigma):
    '''Combine the distance parameters of two sites under a sigma rule.'''
    if rule == 'Lorentz':
        return 0.5 * (qsigma[:, None] + csigma[None, :])
    if rule == 'Good-Hope':
        return np.sqrt(qsigma[:, None] * csigma[None, :])
    raise ValueError(rule)


def combine_epsilon(rule, qepsilon, cepsilon):
    '''Combine the well depths of two sites under an epsilon rule.

    Fender-Halsey is a harmonic mean and so is 0/0 for a pair where neither
    site is parameterised. PyFraME drops such a pair; zero is what that comes
    to here, since it is multiplied into the pair term either way.
    '''
    product = qepsilon[:, None] * cepsilon[None, :]
    if rule == 'Berthelot':
        return np.sqrt(product)
    if rule == 'Fender-Halsey':
        total = qepsilon[:, None] + cepsilon[None, :]
        return np.where(total > 0.0, 2.0 * product / np.where(total > 0.0, total, 1.0), 0.0)
    raise ValueError(rule)


def combined_lennard_jones(mol, parameterised=None,
                           combination_rule='Lorentz-Berthelot'):
    '''Pairwise separations and combined sigma and epsilon.

    Args:
        combination_rule : the rule to combine the per-site parameters under,
            named "<sigma rule>-<epsilon rule>" as PyFraME names them. Lorentz
            or Good-Hope for sigma, Berthelot or Fender-Halsey for epsilon.
    '''
    # "Good-Hope" and "Fender-Halsey" are hyphenated themselves, so the name
    # splits on the known sigma rules rather than on the first hyphen.
    sigma_rule, epsilon_rule = next(
        (rule, combination_rule[len(rule) + 1:]) for rule in ('Good-Hope', 'Lorentz')
        if combination_rule.startswith(rule + '-'))
    (qcoords, qsigma, qepsilon), (ccoords, csigma, cepsilon) = \
        lennard_jones_parameters(mol, parameterised)
    separation = qcoords[:, None, :] - ccoords[None, :, :]
    distance = np.linalg.norm(separation, axis=-1)
    sigma = combine_sigma(sigma_rule, qsigma, csigma)
    epsilon = combine_epsilon(epsilon_rule, qepsilon, cepsilon)
    return separation, distance, sigma, epsilon


def point_charges():
    '''Coordinates (Bohr) and charges of the bundled potential's sites.'''
    with open(json_file) as handle:
        potential = json.load(handle)
    atoms = potential['classical_subsystems'][0]['classical_fragments'][0]['atoms']
    coordinates = np.array([atom['coordinate'] for atom in atoms])
    charges = np.array([atom['multipoles']['elements'][0] for atom in atoms])
    return coordinates, charges


@unittest.skipIf(not has_pyframe, 'PyFraME is not available')
class TestPolarizableEmbedding(unittest.TestCase):
    def test_polarizable_embedding_scf(self):
        # reference agrees with a separate PE implementation to 1e-9; the
        # cross-checks against pyscf.qmmm below validate it independently of
        # any stored number
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-10
        mf.kernel()
        ref_pe_energy = -0.006102943337
        ref_scf_energy = -153.004117624242
        self.assertAlmostEqual(mf.with_embedding.e, ref_pe_energy, 8)
        self.assertAlmostEqual(mf.e_tot, ref_scf_energy, 8)

    def test_polarizable_embedding_grad(self):
        # reference gradients agree with finite differences of the total
        # energy to 3e-9
        ref_grad = np.array([
            [ 0.125517780, -0.032400417,  0.026610452],
            [-0.134505361,  0.028570663, -0.030749991],
            [ 0.039419883, -0.011472521, -0.013463337],
            [-0.004636938,  0.005287195,  0.011372059],
            [ 0.002516648,  0.002966629, -0.012337378],
            [-0.015121637,  0.003860919,  0.001625664],
            [ 0.014817933,  0.004977027,  0.002220583],
            [-0.023231669,  0.003703727,  0.016121840],
            [ 0.005165109,  0.002973140,  0.000565329],
            [-0.010181512, -0.008083997, -0.001563060]])
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()
        de = mf.nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(de - ref_grad).max(), 0, 7)

    def test_unrestricted_matches_restricted(self):
        '''A closed-shell UHF must reproduce RHF, embedding included.'''
        mol = butadiene()
        rhf = embedding.polarizable(scf.RHF(mol), json_file)
        rhf.conv_tol = 1e-12
        rhf.kernel()
        uhf = embedding.polarizable(scf.UHF(mol), json_file)
        uhf.conv_tol = 1e-12
        uhf.kernel()
        self.assertAlmostEqual(uhf.e_tot, rhf.e_tot, 9)
        self.assertAlmostEqual(uhf.with_embedding.e, rhf.with_embedding.e, 9)
        self.assertAlmostEqual(
            abs(uhf.nuc_grad_method().kernel()
                - rhf.nuc_grad_method().kernel()).max(), 0, 7)

    def test_dft_matches_unrestricted_dft(self):
        '''The DFT path goes through a different get_veff; check it agrees.'''
        mol = butadiene()
        rks = embedding.polarizable(dft.RKS(mol, xc='pbe'), json_file)
        rks.conv_tol = 1e-12
        rks.kernel()
        uks = embedding.polarizable(dft.UKS(mol, xc='pbe'), json_file)
        uks.conv_tol = 1e-12
        uks.kernel()
        self.assertAlmostEqual(uks.e_tot, rks.e_tot, 9)
        self.assertAlmostEqual(uks.with_embedding.e, rks.with_embedding.e, 9)

    def test_charges_only_matches_qmmm(self):
        '''Charges with no polarizabilities is plain point-charge embedding.

        Cross-validates the order-0 multipole branch and the electrostatic
        energy against pyscf.qmmm, which is an independent implementation.
        '''
        potential = make_potential('charges.json', orders=lambda i: 0,
                                   depolarize=True)
        coordinates, charges = point_charges()
        mol = butadiene()

        pe = embedding.polarizable(scf.RHF(mol), potential)
        pe.conv_tol = 1e-12
        pe.kernel()
        mm = qmmm.mm_charge(scf.RHF(mol), coordinates, charges, unit='Bohr')
        mm.conv_tol = 1e-12
        mm.kernel()

        self.assertAlmostEqual(pe.e_tot, mm.e_tot, 10)
        self.assertAlmostEqual(pe.with_embedding._e_ind, 0.0, 12)
        self.assertAlmostEqual(
            abs(pe.nuc_grad_method().kernel()
                - mm.nuc_grad_method().kernel()).max(), 0, 9)

    def test_mixed_multipole_orders_gradient(self):
        '''Gradient against finite differences with sites of differing order.

        Every site of the bundled potential carries multipoles up to order 2,
        so only a mixed potential exercises the branches separately -- and the
        differing sets of sites are what the integral cache has to key on.
        '''
        potential = make_potential('mixed.json',
                                   orders=lambda i: 0 if i % 2 else 2)
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), potential)
        mf.conv_tol = 1e-12
        mf.kernel()
        analytic = mf.nuc_grad_method().kernel()[0]

        step = 1e-4
        numerical = []
        for component in range(3):
            energies = []
            for direction in (1, -1):
                perturbed = perturbed_butadiene(0, component, direction * step)
                scanner = embedding.polarizable(scf.RHF(perturbed), potential)
                scanner.conv_tol = 1e-12
                energies.append(scanner.kernel())
            numerical.append((energies[0] - energies[1])
                             / (2 * step / pyscf.lib.param.BOHR))
        self.assertAlmostEqual(abs(analytic - np.array(numerical)).max(), 0, 6)

    def test_scanner_matches_fresh_objects(self):
        '''reset() must leave the object as good as a freshly built one.

        The potential is only re-read when a new one is given, so this guards
        the geometry sync that replaces it.
        '''
        geometries = [butadiene(),
                      perturbed_butadiene(0, 0, 0.1),
                      perturbed_butadiene(3, 2, -0.05)]
        scanner = embedding.polarizable(scf.RHF(butadiene()), json_file).as_scanner()
        scanned = [scanner(mol) for mol in geometries]
        for mol, energy in zip(geometries, scanned):
            fresh = embedding.polarizable(scf.RHF(mol), json_file)
            fresh.conv_tol = scanner.conv_tol
            self.assertAlmostEqual(fresh.kernel(), energy, 9)

    def test_blocked_integrals_match_cached(self):
        '''Blocking the integrals over sites must not change any result.'''
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        reference_energy = mf.kernel()
        reference_grad = mf.nuc_grad_method().kernel()

        cache_fraction = embedding_impl.CACHE_FRACTION
        blocked_range = embedding_impl._blocked_range
        try:
            # never cache, and force several blocks per intor
            embedding_impl.CACHE_FRACTION = 0.0
            embedding_impl._blocked_range = \
                lambda mol, nsites, nbytes, max_memory, description: 2
            blocked = embedding.polarizable(scf.RHF(mol), json_file)
            blocked.conv_tol = 1e-12
            self.assertAlmostEqual(blocked.kernel(), reference_energy, 10)
            self.assertAlmostEqual(
                abs(blocked.nuc_grad_method().kernel() - reference_grad).max(), 0, 9)
        finally:
            embedding_impl.CACHE_FRACTION = cache_fraction
            embedding_impl._blocked_range = blocked_range

    def test_blocked_range_raises_when_a_site_does_not_fit(self):
        mol = butadiene()
        available = pyscf.lib.current_memory()[0] + 10
        self.assertEqual(
            embedding_impl._blocked_range(mol, 100, 1e6, available, 'test'), 10)
        self.assertEqual(
            embedding_impl._blocked_range(mol, 5, 1e6, available, 'test'), 5)
        with self.assertRaises(MemoryError):
            embedding_impl._blocked_range(mol, 100, 50e6, available, 'test')

    def test_gradient_dm_and_atmlst(self):
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()
        full = mf.nuc_grad_method().kernel()

        subset = mf.nuc_grad_method().kernel(atmlst=[0, 2])
        self.assertEqual(subset.shape, (2, 3))
        self.assertAlmostEqual(abs(subset - full[[0, 2]]).max(), 0, 9)

        # an explicit density matrix has to be used, not silently dropped
        other = mf.nuc_grad_method().kernel(dm=np.zeros_like(mf.make_rdm1()))
        self.assertTrue(abs(other - full).max() > 1e-6)

    def test_environment_energy_excluded_from_total(self):
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()
        with_embedding = mf.with_embedding
        contributions = with_embedding.energy_contributions()
        environment = with_embedding.environment_energy()
        self.assertNotAlmostEqual(environment, 0.0, 12)
        # reported, but not part of e
        self.assertTrue(any('Environment' in label for label in contributions))
        self.assertAlmostEqual(
            with_embedding.e,
            with_embedding._e_es + with_embedding._e_ind
            + with_embedding._e_rep + with_embedding._e_disp, 12)

    def test_undo_embedding(self):
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        bare = mf.undo_embedding()
        self.assertIsInstance(bare, scf.hf.RHF)
        self.assertFalse(hasattr(bare, 'with_embedding'))
        mf.conv_tol = 1e-12
        mf.kernel()
        grad = mf.nuc_grad_method()
        self.assertFalse(hasattr(grad.undo_embedding(), 'de_classical_subsystem'))
        with self.assertRaises(NotImplementedError):
            mf.to_gpu()

    def test_option_validation(self):
        mol = butadiene()
        with self.assertRaises(ValueError):
            # missing json_file
            embedding.polarizable(mol, {'environment_energy': True})
        with self.assertRaises(ValueError):
            embedding.polarizable(mol, {'json_file': json_file, 'typo': 1})
        with self.assertRaises(ValueError):
            embedding.polarizable(mol, {'json_file': json_file, 'vdw': {'methd': 'LJ'}})
        with self.assertRaises(ValueError):
            embedding.polarizable(mol, {'json_file': json_file,
                                        'induced_dipoles': {'tolerance': 1e-8}})
        with self.assertRaises(TypeError):
            embedding.polarizable(mol, {'json_file': json_file, 'vdw': 'LJ'})
        with self.assertRaises(TypeError):
            # no potential at all
            embedding.polarizable(scf.RHF(mol))
        with self.assertRaises(RuntimeError):
            # nothing an embedding potential can be attached to
            embedding.polarizable(object(), json_file)

    def test_vdw_energy_matches_explicit_lennard_jones(self):
        '''Repulsion and dispersion against an independent evaluation.

        PyFraME computes E_rep = 4 sum_ij eps_ij (sig_ij/r_ij)**12 and
        E_disp = -4 sum_ij eps_ij (sig_ij/r_ij)**6 over quantum nuclei i and
        classical sites j, with Lorentz-Berthelot combination. Summing that here
        checks the interface and the engine against the closed form.
        '''
        potential = make_vdw_potential()
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol),
                                   {'json_file': potential, 'vdw': {'method': 'LJ'}})
        mf.conv_tol = 1e-12
        mf.kernel()
        with_embedding = mf.with_embedding

        _, distance, sigma, epsilon = combined_lennard_jones(mol)
        repulsion = 4.0 * np.sum(epsilon * (sigma / distance) ** 12)
        dispersion = -4.0 * np.sum(epsilon * (sigma / distance) ** 6)

        self.assertAlmostEqual(with_embedding._e_rep, repulsion, 12)
        self.assertAlmostEqual(with_embedding._e_disp, dispersion, 12)
        # and they have to reach the total energy
        self.assertAlmostEqual(
            with_embedding.e,
            with_embedding._e_es + with_embedding._e_ind + repulsion + dispersion, 12)
        self.assertIn('vdw', mf.with_embedding.options)

    def test_vdw_gradient_matches_explicit_lennard_jones(self):
        '''The vdw part of the gradient against its closed-form derivative.

        The Lennard-Jones terms couple nuclei to sites and never enter the Fock
        matrix, so switching them on must leave the density untouched and shift
        the gradient by exactly the derivative of the two sums.
        '''
        potential = make_vdw_potential()
        mol = butadiene()
        with_vdw = embedding.polarizable(scf.RHF(mol),
                                         {'json_file': potential, 'vdw': {'method': 'LJ'}})
        with_vdw.conv_tol = 1e-12
        with_vdw.kernel()
        without_vdw = embedding.polarizable(scf.RHF(mol), potential)
        without_vdw.conv_tol = 1e-12
        without_vdw.kernel()

        self.assertAlmostEqual(
            abs(with_vdw.make_rdm1() - without_vdw.make_rdm1()).max(), 0, 9)

        difference = (with_vdw.nuc_grad_method().kernel()
                      - without_vdw.nuc_grad_method().kernel())

        separation, distance, sigma, epsilon = combined_lennard_jones(mol)
        coefficient = (-48.0 * epsilon * sigma ** 12 / distance ** 14
                       + 24.0 * epsilon * sigma ** 6 / distance ** 8)
        reference = np.einsum('ij,ijx->ix', coefficient, separation)
        self.assertAlmostEqual(abs(difference - reference).max(), 0, 10)

    def test_vdw_environment_energy(self):
        '''The environment's internal energy picks up the vdw terms.'''
        potential = make_vdw_potential()
        mol = butadiene()
        plain = embedding.polarizable(mol, potential)
        with_vdw = embedding.polarizable(mol, {'json_file': potential,
                                               'vdw': {'method': 'LJ'}})
        self.assertNotAlmostEqual(with_vdw.environment_energy(),
                                  plain.environment_energy(), 9)

    def test_vdw_with_partial_parameters(self):
        '''Sites carrying no Lennard-Jones parameters contribute nothing.

        PyFraME allows a partly parameterised potential -- the usual case, since
        hydrogens and virtual sites often carry none -- and fills zeros in for
        the rest. Zeros are what the combination rule needs to drop those pairs,
        so the energy must equal the sum restricted to the parameterised ones.
        '''
        heavy_only = (lambda element: element != 'H')
        potential = make_vdw_potential('vdw_partial.json',
                                       parameterised=heavy_only)
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol),
                                   {'json_file': potential, 'vdw': {'method': 'LJ'}})
        mf.conv_tol = 1e-12
        mf.kernel()

        _, distance, sigma, epsilon = combined_lennard_jones(mol, heavy_only)
        repulsion = 4.0 * np.sum(epsilon * (sigma / distance) ** 12)
        dispersion = -4.0 * np.sum(epsilon * (sigma / distance) ** 6)
        self.assertAlmostEqual(mf.with_embedding._e_rep, repulsion, 12)
        self.assertAlmostEqual(mf.with_embedding._e_disp, dispersion, 12)

        # and it really is a smaller interaction than the fully parameterised one
        full = embedding.polarizable(mol, {'json_file': make_vdw_potential(),
                                           'vdw': {'method': 'LJ'}})
        self.assertLess(abs(repulsion), abs(full._e_rep))

    def test_vdw_combination_rules(self):
        """Every combination rule PyFraME implements reaches the energy.

        The rule is a pair of choices, one for sigma and one for epsilon, and
        the interface only forwards the name; what this checks is that it
        forwards it at all, and to both interactions, by comparing against the
        same closed form evaluated under each rule in turn.
        """
        potential = make_vdw_potential()
        mol = butadiene()
        energies = []
        for rule in ('Lorentz-Berthelot', 'Lorentz-Fender-Halsey',
                     'Good-Hope-Berthelot', 'Good-Hope-Fender-Halsey'):
            obj = embedding.polarizable(mol, {'json_file': potential,
                                              'vdw': {'method': 'LJ',
                                                      'combination_rule': rule}})
            _, distance, sigma, epsilon = combined_lennard_jones(
                mol, combination_rule=rule)
            repulsion = 4.0 * np.sum(epsilon * (sigma / distance) ** 12)
            dispersion = -4.0 * np.sum(epsilon * (sigma / distance) ** 6)
            self.assertAlmostEqual(obj._e_rep, repulsion, 12)
            self.assertAlmostEqual(obj._e_disp, dispersion, 12)
            energies.append(repulsion + dispersion)

        # guards the four checks above: a rule that was silently ignored would
        # make them pass against a reference that is itself the default
        self.assertEqual(len(set(np.round(energies, 9))), len(energies))

    def test_vdw_with_an_unknown_combination_rule_is_refused(self):
        """A misspelled rule has to fail rather than fall back on the default."""
        mol = butadiene()
        with self.assertRaises(NotImplementedError):
            embedding.polarizable(mol, {'json_file': make_vdw_potential(),
                                        'vdw': {'method': 'LJ',
                                                'combination_rule': 'Lorentz-Lorentz'}})

    def test_vdw_reads_the_exponents_of_the_potential(self):
        """A potential that is not 12-6 is computed from its own exponents.

        The powers are data carried by the sites rather than constants, so a
        9-6 potential has to come through the interface as 9-6.
        """
        potential = make_vdw_potential('vdw_nine_six.json', exponents=(9, 6))
        mol = butadiene()
        obj = embedding.polarizable(mol, {'json_file': potential,
                                          'vdw': {'method': 'LJ'}})

        _, distance, sigma, epsilon = combined_lennard_jones(mol)
        self.assertAlmostEqual(obj._e_rep,
                               4.0 * np.sum(epsilon * (sigma / distance) ** 9), 12)
        self.assertAlmostEqual(obj._e_disp,
                               -4.0 * np.sum(epsilon * (sigma / distance) ** 6), 12)

    def test_vdw_with_inconsistent_exponents_is_refused(self):
        """Sites of one interaction have to agree on the power.

        Two sites of different powers have no coefficient in common and no rule
        pairs them, so PyFraME refuses. The refusal is raised while the static
        contributions are built, i.e. on construction, and has to reach the
        caller rather than leaving a silently wrong energy behind.
        """
        potential = make_vdw_potential('vdw_mixed_exponents.json',
                                       exponents=(12, 6))
        with open(potential) as handle:
            data = json.load(handle)
        data['classical_subsystems'][0]['classical_fragments'][0]['atoms'][0] \
            ['repulsion']['parameters']['lj_exponent'] = 9
        mixed = os.path.join(tmpdir, 'vdw_mixed.json')
        with open(mixed, 'w') as handle:
            json.dump(data, handle)

        mol = butadiene()
        with self.assertRaises(ValueError) as caught:
            embedding.polarizable(mol, {'json_file': mixed, 'vdw': {'method': 'LJ'}})
        self.assertIn('exponent', str(caught.exception))

    def test_vdw_without_any_parameters_is_refused(self):
        '''A vdw request that could only ever yield zero has to say so.

        The bundled potential carries no Lennard-Jones parameters at all, so
        both contributions would be identically zero. PyFraME warns; this
        interface refuses, since the option was asked for and would do nothing.
        '''
        mol = butadiene()
        with self.assertRaises(ValueError) as caught:
            embedding.polarizable(mol, {'json_file': json_file, 'vdw': {'method': 'LJ'}})
        self.assertIn('Lennard-Jones', str(caught.exception))

    def test_vdw_missing_parameters_are_logged(self):
        '''PyFraME's warnings have to reach the PySCF output, not just stderr.'''
        mol = butadiene(verbose=4)
        mol.stdout = io.StringIO()
        embedding.polarizable(mol, {'json_file': make_vdw_potential(
            'vdw_logged.json', parameterised=lambda element: element != 'H'),
            'vdw': {'method': 'LJ'}})
        output = mol.stdout.getvalue()
        self.assertIn('Lennard-Jones', output)
        self.assertIn('contribute nothing', output)

    def _converged_rhf(self):
        mf = embedding.polarizable(
            scf.RHF(butadiene()),
            {'json_file': json_file, 'induced_dipoles': {'threshold': 1e-12}})
        mf.conv_tol = 1e-12
        mf.kernel()
        return mf

    @staticmethod
    def _trial_density(shape, seed=2):
        x = np.random.RandomState(seed).random_sample(shape) * 1e-3
        return x + x.swapaxes(-1, -2)

    def test_gen_response_is_the_derivative_of_the_fock_matrix(self):
        '''With the environment relaxing, vind is the full second derivative.

        The potential is linear in the density, so the response to a trial
        density is exactly what that density does to the total effective
        potential -- the two-electron part and the embedding part together --
        with no finite-difference step to take.
        '''
        mf = self._converged_rhf()
        dm = mf.make_rdm1()
        x = self._trial_density(dm.shape)

        def total_veff(density):
            vhf = mf.get_veff(mf.mol, density)
            return np.asarray(vhf) + vhf.v_embedding

        reference = total_veff(dm + x) - total_veff(dm)

        mf.with_embedding.equilibrium_solvation = True
        self.assertAlmostEqual(
            abs(mf.gen_response(hermi=1)(x) - reference).max(), 0, 10)

    def test_gen_response_is_gated_on_equilibrium_solvation(self):
        '''Off by default, and then the response is the isolated molecule's.'''
        mf = self._converged_rhf()
        x = self._trial_density(mf.make_rdm1().shape)
        bare = mf.undo_embedding().gen_response(hermi=1)(x)

        self.assertFalse(mf.with_embedding.equilibrium_solvation)
        self.assertAlmostEqual(abs(mf.gen_response(hermi=1)(x) - bare).max(), 0, 12)

        mf.with_embedding.equilibrium_solvation = True
        difference = mf.gen_response(hermi=1)(x) - bare
        self.assertAlmostEqual(
            abs(difference - mf.with_embedding._B_dot_x(x)).max(), 0, 12)
        self.assertGreater(abs(difference).max(), 1e-8)

    def test_gen_response_leaves_out_what_the_environment_cannot_see(self):
        '''A trial density carrying no charge density gets no response.

        The RHF triplet kernel is handed a spin density, whose alpha and beta
        blocks cancel, and an anti-hermitian trial density has no Coulomb term
        at all. The environment couples to the total charge density, so it
        contributes to neither -- the same two cases in which the response
        function itself drops its Coulomb term.
        '''
        mf = self._converged_rhf()
        mf.with_embedding.equilibrium_solvation = True
        x = self._trial_density(mf.make_rdm1().shape)
        bare = mf.undo_embedding()

        for kwargs in ({'singlet': False, 'hermi': 1}, {'singlet': None, 'hermi': 2}):
            with self.subTest(**kwargs):
                self.assertAlmostEqual(
                    abs(mf.gen_response(**kwargs)(x)
                        - bare.gen_response(**kwargs)(x)).max(), 0, 12)

        # the same arguments given positionally, since gen_response is wrapped
        # with *args and the two signatures are read by position
        self.assertAlmostEqual(
            abs(mf.gen_response(mf.mo_coeff, mf.mo_occ, False, 1)(x)
                - bare.gen_response(mf.mo_coeff, mf.mo_occ, False, 1)(x)).max(), 0, 12)

    def test_gen_response_sums_the_spin_blocks(self):
        '''UHF and ROHF: the environment sees the total density, not each spin.'''
        for method in (scf.UHF, scf.ROHF):
            with self.subTest(method=method.__name__):
                mf = embedding.polarizable(method(butadiene()), json_file)
                mf.conv_tol = 1e-12
                mf.kernel()
                mf.with_embedding.equilibrium_solvation = True

                x = self._trial_density(np.asarray(mf.make_rdm1()).shape)
                bare = mf.undo_embedding().gen_response(hermi=1)(x)
                v = mf.gen_response(hermi=1)(x)

                expected = mf.with_embedding._B_dot_x(x[0] + x[1])
                self.assertGreater(abs(expected).max(), 1e-8)
                for spin in (0, 1):
                    self.assertAlmostEqual(
                        abs(v[spin] - bare[spin] - expected).max(), 0, 10)

    def test_frozen_potential_does_not_follow_the_density(self):
        '''Attached with dm, the potential is the one of that density.'''
        mol = butadiene()
        guess = scf.RHF(mol).get_init_guess()
        mf = embedding.polarizable(scf.RHF(mol), json_file, dm=guess)
        pe = mf.with_embedding
        self.assertTrue(pe.frozen)

        # what an unfrozen model gives for the same density
        reference = embedding.polarizable(butadiene(), json_file)
        e_guess, v_guess = reference.kernel(guess)
        self.assertAlmostEqual(pe.e, e_guess, 12)
        self.assertAlmostEqual(pe._e_frozen, e_guess, 12)
        self.assertAlmostEqual(abs(pe.v - v_guess).max(), 0, 12)

        mf.conv_tol = 1e-12
        mf.kernel()
        # The SCF relaxed, the environment did not: the potential is the one of
        # the density it was frozen at, and so is the environment's own energy.
        self.assertAlmostEqual(abs(pe.v - v_guess).max(), 0, 12)
        self.assertAlmostEqual(pe._e_frozen, e_guess, 12)
        self.assertGreater(
            abs(reference.kernel(mf.make_rdm1())[1] - v_guess).max(), 1e-6)

        # what kernel reports is the energy of the density it was handed in
        # that potential, which is what the SCF makes stationary
        self.assertAlmostEqual(
            pe.e, e_guess + np.einsum('ij,ij->', pe.v, mf.make_rdm1() - guess), 12)
        self.assertNotAlmostEqual(pe.e, e_guess, 6)

    def test_freezing_at_the_converged_density_changes_nothing(self):
        '''The self-consistent point is a fixed point of the frozen problem.

        A potential frozen at the density the responding calculation converged
        to is the potential that calculation ended with, so the frozen SCF has
        the same fixed point and the same energy.
        '''
        mol = butadiene()
        responding = embedding.polarizable(scf.RHF(mol), json_file)
        responding.conv_tol = 1e-12
        # the frozen energy is stationary in the density, so the energy
        # criterion is met while the density is still moving; the comparison
        # below is on densities, so it is the gradient criterion that has to bite
        responding.conv_tol_grad = 1e-9
        responding.kernel()

        frozen = embedding.polarizable(scf.RHF(mol), json_file,
                                       dm=responding.make_rdm1())
        frozen.conv_tol = 1e-12
        frozen.conv_tol_grad = 1e-9
        frozen.kernel()

        self.assertAlmostEqual(frozen.e_tot, responding.e_tot, 9)
        # not exact: the frozen SCF settles a hair away from the density the
        # potential was frozen at, and the energy follows it there
        self.assertAlmostEqual(frozen.with_embedding.e,
                               responding.with_embedding.e, 8)
        self.assertAlmostEqual(
            abs(frozen.make_rdm1() - responding.make_rdm1()).max(), 0, 8)

    def test_frozen_potential_still_follows_the_nuclei(self):
        '''A reset rebuilds the frozen potential rather than keeping it stale.'''
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()
        dm = mf.make_rdm1()

        frozen = embedding.polarizable(scf.RHF(mol), json_file, dm=dm).with_embedding
        v_before = frozen.v.copy()

        moved = perturbed_butadiene(0, 0, 0.01)
        frozen.reset(moved)
        # the density it is frozen at survives the reset, so the potential owes
        # nothing to the argument here
        v_moved = frozen.kernel(np.zeros_like(dm))[1]
        self.assertGreater(abs(v_moved - v_before).max(), 1e-6)

        reference = embedding.polarizable(moved, json_file)
        e_reference, v_reference = reference.kernel(dm)
        self.assertAlmostEqual(frozen._e_frozen, e_reference, 12)
        self.assertAlmostEqual(abs(v_moved - v_reference).max(), 0, 12)

    def test_frozen_without_a_density_is_refused(self):
        pe = embedding.polarizable(butadiene(), json_file)
        pe.frozen = True
        with self.assertRaises(RuntimeError):
            pe.kernel(scf.RHF(butadiene()).get_init_guess())

    def test_frozen_gradient_matches_finite_differences(self):
        """The frozen energy is one the gradient can be taken of.

        It is the functional the SCF makes stationary, so no response of the
        orbitals enters and the gradient is the environment's own, at the
        density it is frozen at, plus the interaction of the difference between
        the two densities with the potential. For the polarizable model that
        second part carries the response of the induced dipoles, which does not
        cancel here the way it does in the energy gradient.
        """
        mol = butadiene()
        guess = scf.RHF(mol).get_init_guess()

        for name, attach in (('polarizable', embedding.polarizable),
                             ('electrostatic', embedding.electrostatic)):
            with self.subTest(model=name):
                def energy(molecule):
                    mf = attach(scf.RHF(molecule), json_file, dm=guess)
                    mf.conv_tol = 1e-13
                    return mf.kernel()

                mf = attach(scf.RHF(mol), json_file, dm=guess)
                mf.conv_tol = 1e-13
                mf.kernel()
                gradient = mf.nuc_grad_method().kernel()
                self.assertGreater(abs(gradient).max(), 1e-2)

                step = 1e-3
                for atom in (0, 1):
                    for component in range(3):
                        reference = (
                            energy(perturbed_butadiene(atom, component, step * BOHR))
                            - energy(perturbed_butadiene(atom, component, -step * BOHR))
                        ) / (2 * step)
                        self.assertAlmostEqual(gradient[atom, component],
                                               reference, 5)

    def test_perturbed_induced_dipoles_match_finite_differences(self):
        """How the induced dipoles answer a nuclear displacement.

        The dipoles solve mu = B^-1 F with B a property of the sites alone, so
        only the field moves. Isolated here because the frozen gradient needs
        it and the induction Hessian will.
        """
        from pyscf.embedding import embedding_gradient
        mol = butadiene()
        dm = scf.RHF(mol).get_init_guess()

        def polarized(molecule):
            model = embedding.polarizable(molecule, json_file)
            model.kernel(dm)
            return model

        model = polarized(mol)
        perturbed = embedding_gradient.perturbed_induced_dipoles(model, dm)
        self.assertGreater(abs(perturbed).max(), 1e-4)

        step = 1e-4
        for atom in range(mol.natm):
            for component in range(3):
                dipoles = []
                for sign in (1, -1):
                    moved = polarized(perturbed_butadiene(atom, component,
                                                          sign * step * BOHR))
                    dipoles.append(
                        moved.classical_subsystem.induced_dipoles.induced_dipoles)
                reference = (dipoles[0] - dipoles[1]) / (2 * step)
                self.assertAlmostEqual(
                    abs(perturbed[atom, component] - reference).max(), 0, 7)

    def test_a_density_without_a_method_is_refused(self):
        mol = butadiene()
        guess = scf.RHF(mol).get_init_guess()
        with self.assertRaises(TypeError):
            embedding.polarizable(mol, json_file, dm=guess)

    def test_stability_uses_the_environment_response(self):
        '''The orbital Hessian the analysis diagonalizes includes the environment.

        equilibrium_solvation is off by default, so without stability turning it
        on for the duration the environment would be missing from the Hessian.
        It has to be put back afterwards.
        '''
        mf = self._converged_rhf()
        pe = mf.with_embedding
        self.assertFalse(pe.equilibrium_solvation)

        with mock.patch.object(pe, '_B_dot_x', wraps=pe._B_dot_x) as response:
            mo, _, stable, _ = mf.stability(return_status=True)
        self.assertGreater(response.call_count, 0)
        self.assertFalse(pe.equilibrium_solvation)

        # a closed-shell ground state in a classical environment
        self.assertTrue(stable)
        self.assertEqual(mo.shape, mf.mo_coeff.shape)

    def test_stability_of_a_frozen_potential_has_no_response(self):
        '''A fixed external term has no second derivative to contribute.'''
        mol = butadiene()
        responding = embedding.polarizable(scf.RHF(mol), json_file)
        responding.conv_tol = 1e-12
        responding.kernel()

        mf = embedding.polarizable(scf.RHF(mol), json_file,
                                   dm=responding.make_rdm1())
        mf.conv_tol = 1e-12
        mf.kernel()
        pe = mf.with_embedding

        with mock.patch.object(pe, '_B_dot_x', wraps=pe._B_dot_x) as response:
            mo, _, stable, _ = mf.stability(return_status=True)
        self.assertEqual(response.call_count, 0)
        self.assertTrue(stable)
        self.assertEqual(mo.shape, mf.mo_coeff.shape)

    def test_a_frozen_potential_does_not_respond(self):
        '''frozen wins over equilibrium_solvation, rather than contradicting it.'''
        mf = self._converged_rhf()
        pe = mf.with_embedding
        x = self._trial_density(mf.make_rdm1().shape)
        bare = mf.undo_embedding().gen_response(hermi=1)(x)

        pe.equilibrium_solvation = True
        self.assertGreater(abs(mf.gen_response(hermi=1)(x) - bare).max(), 1e-8)

        pe.frozen = True
        self.assertAlmostEqual(
            abs(mf.gen_response(hermi=1)(x) - bare).max(), 0, 12)

    def test_stability_runs_unrestricted_and_external(self):
        mf = embedding.polarizable(scf.UHF(butadiene()), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()
        pe = mf.with_embedding

        with mock.patch.object(pe, '_B_dot_x', wraps=pe._B_dot_x) as response:
            internal, external = mf.stability(internal=True, external=True)
        # The internal kernel couples to the density, so the environment is in
        # that Hessian. The external one is the UHF -> GHF rotation, built from
        # the with_j=False and hermi=2 kernels, which do not -- it contributes
        # nothing to the count, and its orbitals span both spin blocks.
        self.assertGreater(response.call_count, 0)
        self.assertEqual(np.asarray(internal).shape, np.asarray(mf.mo_coeff).shape)
        nao = mf.mol.nao
        self.assertEqual(np.asarray(external).shape, (2 * nao, 2 * nao))
        self.assertFalse(pe.equilibrium_solvation)

    def test_post_scf_relaxes_the_environment(self):
        '''The macro iteration ends with the potential of the MP2 density.

        Unfrozen, the environment is relaxed against the correlated density
        rather than left at the SCF one, so the converged potential is the one
        the correlated density produces and the SCF underneath has moved off
        where it started.
        '''
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()
        e_scf = mf.e_tot
        v_scf = mf.with_embedding.v.copy()

        mp2 = mf.MP2()
        self.assertIs(mp2.with_embedding, mf.with_embedding)
        e_corr, _ = mp2.kernel()

        pe = mp2.with_embedding
        # the potential belongs to the correlated density, not the SCF one
        self.assertAlmostEqual(
            abs(pe._dm - mp2.make_rdm1(ao_repr=True)).max(), 0, 6)
        self.assertGreater(abs(pe.v - v_scf).max(), 1e-8)
        # and the SCF underneath was re-converged against it
        self.assertNotAlmostEqual(mp2._scf.e_tot, e_scf, 8)
        self.assertAlmostEqual(mp2.e_tot, mp2._scf.e_tot + e_corr, 10)

    def test_post_scf_freezing_at_its_own_density_changes_nothing(self):
        '''The relaxed solution is a fixed point of the frozen problem.'''
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()
        relaxed = mf.MP2()
        relaxed.kernel()

        frozen = embedding.polarizable(
            mp.MP2(scf.RHF(mol)), json_file,
            dm=relaxed.make_rdm1(ao_repr=True))
        self.assertTrue(frozen.with_embedding.frozen)
        frozen._scf.conv_tol = 1e-12
        frozen.kernel()

        self.assertAlmostEqual(frozen.e_tot, relaxed.e_tot, 7)
        self.assertAlmostEqual(frozen._scf.e_tot, relaxed._scf.e_tot, 7)

    def test_post_scf_reconverges_the_scf_it_is_given(self):
        '''The orbitals are the ones belonging to the potential, not the ones
        the object happened to be built on.

        Freezing at a density other than the SCF's changes the SCF problem, and
        an MP2 built on an SCF that has not been run has no orbitals at all.
        Both are the same thing: the pass that runs the correlated method has to
        converge the SCF in the potential first.
        '''
        mol = butadiene()
        gas_phase = scf.RHF(mol)
        gas_phase.conv_tol = 1e-12
        gas_phase.kernel()
        gas_phase_mo = gas_phase.mo_coeff.copy()

        frozen = embedding.polarizable(mp.MP2(gas_phase), json_file,
                                       dm=gas_phase.make_rdm1())
        frozen.kernel()
        self.assertGreater(abs(frozen._scf.mo_coeff - gas_phase_mo).max(), 1e-6)
        self.assertTrue(frozen._scf.converged)

        # the same calculation reached the ordinary way
        reference = embedding.polarizable(scf.RHF(mol), json_file,
                                          dm=gas_phase.make_rdm1())
        reference.conv_tol = 1e-12
        reference.kernel()
        self.assertAlmostEqual(frozen._scf.e_tot, reference.e_tot, 8)
        self.assertAlmostEqual(frozen.e_corr, mp.MP2(reference).kernel()[0], 8)

    def test_post_scf_refuses_an_initial_guess(self):
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()
        with self.assertRaises(NotImplementedError):
            mf.CCSD().kernel(t1=0, t2=0)

    def test_frozen_post_scf_is_the_plain_method_on_the_embedded_scf(self):
        '''Frozen, the potential is part of h and the correlation is untouched.'''
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file, dm=None)
        mf.conv_tol = 1e-12
        mf.kernel()

        frozen = embedding.polarizable(mp.MP2(mf.undo_embedding()), json_file,
                                       dm=mf.make_rdm1())
        e_corr, _ = frozen.kernel()
        self.assertAlmostEqual(e_corr, mp.MP2(mf).kernel()[0], 10)

    def test_post_scf_without_induction_needs_one_macro_cycle(self):
        '''An environment that cannot respond is converged after one pass.

        ElectrostaticEmbedding has no density-dependent potential, so rebuilding
        it from the correlated density reproduces it exactly and the macro
        iteration has nothing left to do.
        '''
        mol = butadiene()
        mf = embedding.electrostatic(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()
        v_scf = mf.with_embedding.v.copy()

        mp2 = mf.MP2()
        e_corr, _ = mp2.kernel()
        self.assertAlmostEqual(abs(mp2.with_embedding.v - v_scf).max(), 0, 12)
        self.assertAlmostEqual(e_corr, mp.MP2(mf).kernel()[0], 10)

    def test_post_scf_wrappers(self):
        '''The hooks build the wrapped method, and undo_embedding unbuilds it.'''
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()

        for name in ('MP2', 'CISD', 'CCSD'):
            with self.subTest(method=name):
                wrapped = getattr(mf, name)()
                self.assertIsInstance(wrapped,
                                      embedding_impl._attach_embedding.PostSCFWithEmbedding)
                self.assertIs(wrapped.with_embedding, mf.with_embedding)
                self.assertIn('PolarizableEmbedding', type(wrapped).__name__)

                bare = wrapped.undo_embedding()
                self.assertNotIsInstance(
                    bare, embedding_impl._attach_embedding.PostSCFWithEmbedding)
                self.assertFalse(hasattr(bare, '_basic_scanner'))
                self.assertFalse(hasattr(bare._scf, 'with_embedding'))

                with self.assertRaises(NotImplementedError):
                    wrapped.nuc_grad_method()
                with self.assertRaises(NotImplementedError):
                    wrapped.to_gpu()

    def test_tdscf_non_equilibrium_adds_nothing_to_the_excitation(self):
        '''The default is the environment the ground state left behind.

        A vertical excitation is fast next to the environment, so the
        environment does not relax against the transition density and the
        excitation is the isolated-molecule one on orbitals that already know
        about the ground state potential.
        '''
        from pyscf import tdscf
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()

        td = mf.TDA()
        td.nstates = 3
        self.assertFalse(td.equilibrium_solvation)
        with mock.patch.object(td.with_embedding, '_B_dot_x',
                               wraps=td.with_embedding._B_dot_x) as response:
            excitations = td.kernel()[0]
        self.assertEqual(response.call_count, 0)

        bare = tdscf.TDA(mf.undo_embedding())
        bare.nstates = 3
        self.assertAlmostEqual(abs(excitations - bare.kernel()[0]).max(), 0, 10)

    def test_tdscf_equilibrium_relaxes_against_the_transition_density(self):
        '''And does so on its own copy of the environment.'''
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()

        reference = mf.TDA()
        reference.nstates = 3
        e_frozen_environment = reference.kernel()[0]

        td = mf.TDA(equilibrium_solvation=True)
        td.nstates = 3
        self.assertTrue(td.equilibrium_solvation)
        self.assertIsNot(td.with_embedding, mf.with_embedding)
        with mock.patch.object(td.with_embedding, '_B_dot_x',
                               wraps=td.with_embedding._B_dot_x) as response:
            excitations = td.kernel()[0]
        self.assertGreater(response.call_count, 0)

        # the environment relaxing around the excited state stabilizes it
        self.assertGreater(abs(excitations - e_frozen_environment).max(), 1e-5)
        self.assertLess(excitations[0], e_frozen_environment[0])

        # and the ground state's environment was left as it was
        self.assertFalse(mf.with_embedding.equilibrium_solvation)

    def test_tdscf_reads_only_its_own_flag(self):
        '''The ground state's equilibrium_solvation says nothing about here.

        It answers what the ground state orbital Hessian should contain, which
        is a different question. Building the excitation's response from the
        embedded SCF rather than the bare one would let that flag through, and
        with the excitation's own flag set as well the environment would be
        counted twice.
        '''
        from pyscf import tdscf
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()

        bare = tdscf.TDA(mf.undo_embedding())
        bare.nstates = 3
        e_bare = bare.kernel()[0]

        td_equilibrium = mf.TDA(equilibrium_solvation=True)
        td_equilibrium.nstates = 3
        e_equilibrium = td_equilibrium.kernel()[0]

        mf.with_embedding.equilibrium_solvation = True
        try:
            td = mf.TDA()
            td.nstates = 3
            self.assertFalse(td.equilibrium_solvation)
            self.assertAlmostEqual(abs(td.kernel()[0] - e_bare).max(), 0, 10)

            td_on = mf.TDA(equilibrium_solvation=True)
            td_on.nstates = 3
            self.assertAlmostEqual(
                abs(td_on.kernel()[0] - e_equilibrium).max(), 0, 10)
        finally:
            mf.with_embedding.equilibrium_solvation = False

    def test_tdscf_triplet_gets_no_response(self):
        '''A triplet excitation does not change the total charge density.'''
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()

        energies = {}
        for equilibrium in (False, True):
            td = mf.TDA(equilibrium_solvation=equilibrium)
            td.singlet = False
            td.nstates = 2
            energies[equilibrium] = td.kernel()[0]
        self.assertAlmostEqual(
            abs(energies[True] - energies[False]).max(), 0, 10)

    def test_tdscf_casida_matches_the_full_solver(self):
        '''Casida stays valid with the environment in the response.

        The environment kernel is real, symmetric and frequency independent, so
        it enters A and B alike and leaves A - B untouched -- which is the
        assumption Casida's (A-B)^(1/2)(A+B)(A-B)^(1/2) rests on. solvent marks
        CasidaTDDFT as unavailable; this is why we do not.
        '''
        from pyscf import tdscf
        mol = butadiene()
        mf = embedding.polarizable(dft.RKS(mol, xc='pbe'), json_file)
        mf.conv_tol = 1e-11
        mf.kernel()

        casida = mf.TDDFT(equilibrium_solvation=True)
        self.assertIn('Casida', type(casida).__name__)
        casida.nstates = 3

        full = embedding.polarizable(tdscf.rks.TDDFT(mf), json_file)
        full.equilibrium_solvation = True
        full.nstates = 3

        self.assertAlmostEqual(
            abs(casida.kernel()[0] - full.kernel()[0]).max(), 0, 9)

    def test_tdscf_unrestricted(self):
        mol = butadiene()
        mf = embedding.polarizable(scf.UHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()

        td = mf.TDA(equilibrium_solvation=True)
        td.nstates = 2
        pe = td.with_embedding
        with mock.patch.object(pe, '_B_dot_x', wraps=pe._B_dot_x) as response:
            td.kernel()
        self.assertGreater(response.call_count, 0)
        # The environment is handed the spin-summed density: the trial density
        # a UHF response function passes around is (2, nset, nao, nao), and what
        # reaches _B_dot_x has lost the spin axis. How the one potential that
        # comes back is applied to both blocks is pinned by
        # test_gen_response_sums_the_spin_blocks, which covers the same helper.
        nao = mf.mol.nao
        for call in response.call_args_list:
            argument = np.asarray(call.args[0])
            self.assertEqual(argument.ndim, 3)
            self.assertEqual(argument.shape[-2:], (nao, nao))

    def test_tdscf_wrappers_and_refusals(self):
        from pyscf import tdscf
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()

        td = mf.TDA()
        self.assertIsInstance(
            td, embedding_impl._attach_embedding.TDSCFWithEmbedding)
        self.assertIn('PolarizableEmbedding', type(td).__name__)
        bare = td.undo_embedding()
        self.assertFalse(hasattr(bare, 'with_embedding'))
        self.assertFalse(hasattr(bare._scf, 'with_embedding'))

        for call in (td.get_ab, td.nuc_grad_method, td.to_gpu):
            with self.subTest(call=call.__name__):
                with self.assertRaises(NotImplementedError):
                    call()

        # the ground state has to carry the potential the excitation sees
        with self.assertRaises(TypeError):
            embedding.polarizable(tdscf.TDA(mf.undo_embedding()), json_file)
        # and dm belongs on the ground state, not here
        with self.assertRaises(NotImplementedError):
            embedding.polarizable(tdscf.TDA(mf), json_file,
                                  dm=mf.make_rdm1())

    def test_tdscf_equilibrium_over_a_frozen_ground_state_is_refused(self):
        mol = butadiene()
        guess = scf.RHF(mol).get_init_guess()
        mf = embedding.polarizable(scf.RHF(mol), json_file, dm=guess)
        mf.conv_tol = 1e-12
        mf.kernel()

        with self.assertRaises(RuntimeError):
            mf.TDA(equilibrium_solvation=True)
        td = mf.TDA()
        with self.assertRaises(RuntimeError):
            td.equilibrium_solvation = True

    def test_cas_of_one_determinant_is_the_scf(self):
        '''A single-configuration active space has the SCF density and energy.

        With one orbital and two electrons there is one determinant, so CASCI
        and CASSCF both come back to the SCF solution -- and with it to the SCF
        embedding energy. Since the potential reaches a CAS method through
        get_hcore rather than through get_veff, that is the statement that the
        duplicated Tr(D v) is taken back out exactly.
        '''
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()

        for name in ('CASCI', 'CASSCF'):
            with self.subTest(method=name):
                mc = getattr(mf, name)(1, 2)
                mc.conv_tol = 1e-10
                self.assertAlmostEqual(mc.kernel()[0], mf.e_tot, 9)
                self.assertAlmostEqual(mc.with_embedding.e, mf.with_embedding.e, 9)

    def test_casci_relaxes_the_environment(self):
        '''The macro iteration ends with the potential of the CI density.'''
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()
        v_scf = mf.with_embedding.v.copy()

        mc = mf.CASCI(4, 4)
        e_tot = mc.kernel()[0]
        pe = mc.with_embedding

        self.assertTrue(mc.converged)
        self.assertAlmostEqual(
            abs(pe._dm - mc.make_rdm1(ao_repr=True)).max(), 0, 9)
        self.assertGreater(abs(pe.v - v_scf).max(), 1e-8)
        self.assertLess(e_tot, mf.e_tot)

    def test_casscf_polarizes_with_the_orbitals_it_is_given(self):
        '''The density the environment sees belongs to the current orbitals.

        mc1step.kernel keeps the orbitals of the macro iteration in a local and
        only puts them on the object when it returns, so a density built from
        self.mo_coeff during the iteration belongs to the starting guess, not to
        the orbitals the CI vector goes with.
        '''
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()

        mc = mf.CASSCF(2, 2)
        mc.conv_tol = 1e-10
        mc.kernel()
        pe = mc.with_embedding

        # the environment ends up polarized by the density the method converged
        # to, rather than by one belonging to orbitals it has left behind
        self.assertAlmostEqual(
            abs(pe._dm - mc.make_rdm1(ao_repr=True)).max(), 0, 9)
        reference = embedding.polarizable(butadiene(), json_file)
        self.assertAlmostEqual(
            pe.e, reference.kernel(mc.make_rdm1(ao_repr=True))[0], 10)

    def test_cas_freezing_at_its_own_density_changes_nothing(self):
        '''The relaxed solution is a fixed point of the frozen problem.'''
        from pyscf import mcscf
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()
        bare = mf.undo_embedding()

        for name in ('CASCI', 'CASSCF'):
            with self.subTest(method=name):
                relaxed = getattr(mf, name)(2, 2)
                relaxed.conv_tol = 1e-10
                e_relaxed = relaxed.kernel()[0]

                frozen = embedding.polarizable(
                    getattr(mcscf, name)(bare, 2, 2), json_file,
                    dm=relaxed.make_rdm1(ao_repr=True))
                frozen.conv_tol = 1e-10
                self.assertTrue(frozen.with_embedding.frozen)
                self.assertAlmostEqual(frozen.kernel()[0], e_relaxed, 7)

    def test_casci_polarizes_with_the_state_it_is_told(self):
        '''One environment, one density: state_id says whose.'''
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()

        energies, potentials = {}, {}
        for state_id in (0, 1):
            mc = mf.CASCI(4, 4)
            mc.fcisolver.nroots = 2
            mc.with_embedding.state_id = state_id
            mc.with_embedding.conv_tol = 1e-10
            energies[state_id] = np.atleast_1d(mc.kernel()[0])
            potentials[state_id] = mc.with_embedding.v.copy()
            # the environment is polarized by that state and no other
            self.assertAlmostEqual(
                abs(mc.with_embedding._dm
                    - mc.make_rdm1(ci=mc.ci[state_id], ao_repr=True)).max(), 0, 9)

        # so it is a different environment, and a different calculation. The
        # energies move by little here -- the potential of three waters is weak
        # and the two states differ only in the active space -- so the potential
        # is the sharper statement of the two.
        self.assertEqual(energies[0].shape, (2,))
        self.assertGreater(abs(potentials[0] - potentials[1]).max(), 1e-6)
        self.assertGreater(abs(energies[0] - energies[1]).max(), 1e-8)

    def test_cas_wrappers(self):
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()

        attach = embedding_impl._attach_embedding
        for name, cls in (('CASCI', attach.CASCIWithEmbedding),
                          ('CASSCF', attach.CASSCFWithEmbedding)):
            with self.subTest(method=name):
                wrapped = getattr(mf, name)(2, 2)
                self.assertIsInstance(wrapped, cls)
                self.assertIs(wrapped.with_embedding, mf.with_embedding)
                self.assertIn('PolarizableEmbedding', type(wrapped).__name__)

                bare = wrapped.undo_embedding()
                self.assertNotIsInstance(bare, cls)
                self.assertFalse(hasattr(bare, 'with_embedding'))

                with self.assertRaises(NotImplementedError):
                    wrapped.nuc_grad_method()
                with self.assertRaises(NotImplementedError):
                    wrapped.to_gpu()

    def test_hessian_is_refused_for_induced_dipoles(self):
        '''The induced dipoles respond to a displacement, and that is unwritten.'''
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()
        with self.assertRaises(NotImplementedError) as raised:
            mf.Hessian()
        self.assertIn('electrostatic', str(raised.exception))

    def test_quantum_subsystem_size_is_checked(self):
        mismatched = pyscf.M(atom='C 0 0 0; O 0 0 1.2', basis='sto3g', verbose=0)
        with self.assertRaises(ValueError):
            embedding.polarizable(mismatched, json_file)

    def test_B_dot_x_is_the_response_of_the_potential(self):
        '''_B_dot_x(x) is what x adds to the potential, and nothing else.

        The potential is affine in the density -- the permanent multipoles
        contribute a constant term and the induced dipoles a linear one -- so
        the response to a trial density is exactly the difference of the
        potential at two densities that differ by it, with no finite-difference
        step to take.
        '''
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), {'json_file': json_file,
                                                  'induced_dipoles': {'threshold': 1e-12}})
        mf.conv_tol = 1e-12
        mf.kernel()
        pe = mf.with_embedding

        dm = mf.make_rdm1()
        x = np.random.RandomState(0).random_sample(dm.shape)
        x = (x + x.T) * 1e-3

        response = pe._B_dot_x(x)
        v0 = pe.kernel(dm)[1].copy()
        v1 = pe.kernel(dm + x)[1]
        self.assertAlmostEqual(abs(response - (v1 - v0)).max(), 0, 10)
        # and it is not zero, i.e. the test above has something to say
        self.assertGreater(abs(response).max(), 1e-8)

    def test_B_dot_x_keeps_the_shape_and_the_state(self):
        mol = butadiene()
        mf = embedding.polarizable(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()
        pe = mf.with_embedding

        dm = mf.make_rdm1()
        rng = np.random.RandomState(1)
        xs = rng.random_sample((3,) + dm.shape) * 1e-3
        xs = xs + xs.transpose(0, 2, 1)

        e, v = pe.e, pe.v.copy()
        induced_dipoles = pe.classical_subsystem.induced_dipoles.induced_dipoles.copy()
        n_cached = len(pe.classical_subsystem.perturbed_induced_dipoles)

        stacked = pe._B_dot_x(xs)
        self.assertEqual(stacked.shape, xs.shape)
        for i, x in enumerate(xs):
            single = pe._B_dot_x(x)
            self.assertEqual(single.shape, x.shape)
            self.assertAlmostEqual(abs(single - stacked[i]).max(), 0, 10)

        # the model is left on the density it was last run on
        self.assertEqual(pe.e, e)
        self.assertAlmostEqual(abs(pe.v - v).max(), 0, 12)
        self.assertAlmostEqual(
            abs(pe.classical_subsystem.induced_dipoles.induced_dipoles
                - induced_dipoles).max(), 0, 12)
        self.assertEqual(len(pe.classical_subsystem.perturbed_induced_dipoles),
                         n_cached)


@unittest.skipIf(not has_pyframe, 'PyFraME is not available')
class TestElectrostaticEmbedding(unittest.TestCase):
    def test_matches_polarizable_on_a_depolarized_potential(self):
        '''Dropping the induction is the same as dropping the polarizabilities.

        The two are the same model reached differently -- one by the class, the
        other by the potential -- so energy, density and gradient have to agree.
        '''
        mol = butadiene()
        depolarized = make_potential('depolarized.json', depolarize=True)

        es = embedding.electrostatic(scf.RHF(mol), json_file)
        es.conv_tol = 1e-12
        es.kernel()
        pe = embedding.polarizable(scf.RHF(mol), depolarized)
        pe.conv_tol = 1e-12
        pe.kernel()

        self.assertAlmostEqual(es.e_tot, pe.e_tot, 10)
        self.assertAlmostEqual(es.with_embedding.e, pe.with_embedding.e, 10)
        self.assertAlmostEqual(
            abs(es.nuc_grad_method().kernel()
                - pe.nuc_grad_method().kernel()).max(), 0, 9)

        # and it is not the polarizable result on the same potential
        polarizable = embedding.polarizable(scf.RHF(mol), json_file)
        polarizable.conv_tol = 1e-12
        polarizable.kernel()
        self.assertNotAlmostEqual(es.e_tot, polarizable.e_tot, 8)

    def test_gen_response_adds_nothing(self):
        '''No induced dipoles, so the response is the isolated molecule's.'''
        mf = embedding.electrostatic(scf.RHF(butadiene()), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()
        mf.with_embedding.equilibrium_solvation = True

        dm = mf.make_rdm1()
        x = np.random.RandomState(2).random_sample(dm.shape) * 1e-3
        x = x + x.T
        self.assertAlmostEqual(
            abs(mf.gen_response(hermi=1)(x)
                - mf.undo_embedding().gen_response(hermi=1)(x)).max(), 0, 12)

    def test_B_dot_x_is_zero(self):
        '''Without induced dipoles there is nothing to respond.'''
        mol = butadiene()
        mf = embedding.electrostatic(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()
        es = mf.with_embedding

        dm = mf.make_rdm1()
        x = np.random.RandomState(0).random_sample(dm.shape)
        x = (x + x.T) * 1e-3
        self.assertEqual(abs(es._B_dot_x(x)).max(), 0)
        self.assertEqual(es._B_dot_x(x).shape, x.shape)
        self.assertEqual(es._B_dot_x(np.array([x, x])).shape, (2,) + x.shape)

        # the potential really is constant in the density, which is what the
        # zero above asserts
        v0 = es.kernel(dm)[1].copy()
        v1 = es.kernel(dm + x)[1]
        self.assertAlmostEqual(abs(v1 - v0).max(), 0, 12)

    def test_hessian_integrals_match_finite_differences(self):
        '''The second derivatives of the multipole potential, order by order.

        Differentiating the gradient integrals, which the gradient tests already
        tie to the energy, isolates the new integrals from everything else in
        the Hessian. The quadrupole branch is the interesting one: the operator
        carries two derivatives of its own, so this needs four.
        '''
        from pyscf.embedding import embedding_gradient, embedding_hessian

        for label, orders in (('charges', lambda i: 0),
                              ('dipoles', lambda i: 1),
                              ('quadrupoles', None),
                              ('mixed', lambda i: i % 3)):
            with self.subTest(sites=label):
                potential = make_potential('hess_%s.json' % label, orders=orders,
                                           depolarize=True)
                mol = butadiene()
                model = embedding.electrostatic(mol, potential)
                classical = model.classical_subsystem
                arguments = (classical.coordinates, classical.multipole_orders,
                             classical.degenerate_multipoles_with_taylor_coefficients)
                dm = scf.RHF(mol).get_init_guess()

                driver = embedding_hessian.EmbeddingHessianIntegralDriver(mol)
                bra_bra, bra_ket = driver.multipole_potential_hessian_integrals(*arguments)
                hess = embedding_hessian._hess_from_operators(
                    mol, bra_bra, bra_ket, dm, range(mol.natm))

                step = 1e-4
                reference = np.zeros_like(hess)
                for atom in range(mol.natm):
                    for component in range(3):
                        gradients = []
                        for sign in (1, -1):
                            moved = perturbed_butadiene(atom, component,
                                                        sign * step * BOHR)
                            operator = embedding_gradient.EmbeddingGradientIntegralDriver(
                                moved).multipole_potential_gradient_integrals(*arguments)
                            gradients.append(embedding_gradient._grad_from_operator(
                                moved, operator, dm))
                        reference[:, atom, :, component] = \
                            (gradients[0] - gradients[1]) / (2 * step)
                self.assertGreater(abs(hess).max(), 1e-4)
                self.assertAlmostEqual(abs(hess - reference).max(), 0, 8)

    def test_hessian_fock_gradient_matches_finite_differences(self):
        '''What make_h1 hands the CPHF equations is the potential's derivative.'''
        from pyscf.embedding import embedding_hessian
        mol = butadiene()
        model = embedding.electrostatic(mol, json_file)
        derivatives = embedding_hessian.fock_gradient(model, range(mol.natm))

        step = 1e-4
        for atom in range(mol.natm):
            for component in range(3):
                potentials = []
                for sign in (1, -1):
                    moved = perturbed_butadiene(atom, component, sign * step * BOHR)
                    potentials.append(
                        embedding.electrostatic(moved, json_file)._f_el_es)
                reference = (potentials[0] - potentials[1]) / (2 * step)
                self.assertAlmostEqual(
                    abs(derivatives[atom][component] - reference).max(), 0, 8)

    def test_hessian_classical_terms_match_finite_differences(self):
        '''The nuclear electrostatic and Lennard-Jones second derivatives.

        Both come from PyFraME and neither touches the density; the vdw one is
        block diagonal over the nuclei, since a nucleus reaches the sites but
        not the other nuclei through it.
        '''
        from pyscf.embedding import embedding_hessian
        potential = make_vdw_potential('hess_vdw.json')
        options = {'json_file': potential, 'vdw': {'method': 'LJ'}}
        mol = butadiene()
        model = embedding.electrostatic(mol, options)
        # the density-dependent part is subtracted out, leaving the classical terms
        dm = np.zeros((mol.nao, mol.nao))
        hess = embedding_hessian.kernel(model, dm)

        step = 1e-4
        reference = np.zeros_like(hess)
        for atom in range(mol.natm):
            for component in range(3):
                gradients = []
                for sign in (1, -1):
                    moved = perturbed_butadiene(atom, component, sign * step * BOHR)
                    gradients.append(embedding_gradient_kernel(
                        embedding.electrostatic(moved, options), dm))
                reference[:, atom, :, component] = \
                    (gradients[0] - gradients[1]) / (2 * step)
        self.assertGreater(abs(hess).max(), 1e-3)
        self.assertAlmostEqual(abs(hess - reference).max(), 0, 8)

        # both terms couple a nucleus to the sites and not to the other nuclei
        off_diagonal = hess.copy()
        for atom in range(mol.natm):
            off_diagonal[atom, atom] = 0
        self.assertAlmostEqual(abs(off_diagonal).max(), 0, 10)

    def test_hessian_matches_finite_differences(self):
        '''The whole Hessian against finite differences of the gradient.

        Two atoms rather than ten, which is what atmlst is for: the analytic
        Hessian costs the same either way, the finite differences do not. The
        tolerance is set by noise in the gradients being differenced -- the bare
        RHF Hessian of this molecule differs from its own finite differences by
        the same amount at this step, and both shrink as the step grows.
        '''
        potential = make_potential('hess_scf.json', orders=lambda i: i % 3,
                                   depolarize=True)
        mol = butadiene()
        mf = embedding.electrostatic(scf.RHF(mol), potential)
        mf.conv_tol = 1e-13
        mf.kernel()

        atmlst = [0, 1]
        hess = mf.Hessian().kernel(atmlst=atmlst)
        self.assertEqual(hess.shape, (2, 2, 3, 3))
        self.assertAlmostEqual(abs(hess - hess.transpose(1, 0, 3, 2)).max(), 0, 6)

        step = 1e-3
        reference = np.zeros_like(hess)
        for j0, atom in enumerate(atmlst):
            for component in range(3):
                gradients = []
                for sign in (1, -1):
                    moved = perturbed_butadiene(atom, component, sign * step * BOHR)
                    displaced = embedding.electrostatic(scf.RHF(moved), potential)
                    displaced.conv_tol = 1e-13
                    displaced.kernel()
                    gradients.append(displaced.nuc_grad_method().kernel()[atmlst])
                reference[:, j0, :, component] = \
                    (gradients[0] - gradients[1]) / (2 * step)
        self.assertGreater(abs(hess).max(), 1e-2)
        self.assertLess(abs(hess - reference).max(), 1e-5)

    def test_hessian_unrestricted_matches_restricted(self):
        '''A closed-shell UHF Hessian is the RHF one, embedding included.'''
        potential = make_potential('hess_uhf.json', depolarize=True)
        mol = butadiene()
        restricted = embedding.electrostatic(scf.RHF(mol), potential)
        restricted.conv_tol = 1e-12
        restricted.kernel()
        unrestricted = embedding.electrostatic(scf.UHF(mol), potential)
        unrestricted.conv_tol = 1e-12
        unrestricted.kernel()

        atmlst = [0, 2]
        self.assertAlmostEqual(
            abs(restricted.Hessian().kernel(atmlst=atmlst)
                - unrestricted.Hessian().kernel(atmlst=atmlst)).max(), 0, 6)

    def test_hessian_wrappers_and_refusals(self):
        from pyscf.embedding import embedding_hessian
        mol = butadiene()
        mf = embedding.electrostatic(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()

        hess = mf.Hessian()
        self.assertIsInstance(hess, embedding_hessian.EmbeddingHess)
        self.assertIn('ElectrostaticEmbedding', type(hess).__name__)
        bare = hess.undo_embedding()
        self.assertNotIsInstance(bare, embedding_hessian.EmbeddingHess)
        with self.assertRaises(NotImplementedError):
            hess.to_gpu()

        # a frozen potential, for the reason the gradient gives
        guess = scf.RHF(mol).get_init_guess()
        frozen = embedding.electrostatic(scf.RHF(mol), json_file, dm=guess)
        frozen.kernel()
        with self.assertRaises(NotImplementedError):
            frozen.Hessian()

        with self.assertRaises(TypeError):
            embedding_hessian.make_hess_object(scf.RHF(mol))

    def test_charges_only_matches_qmmm(self):
        '''Order-0 sites are plain point charges, polarizabilities and all.

        The potential keeps its polarizabilities here, so this is what says that
        electrostatic embedding really ignores them rather than being handed a
        potential that has none.
        '''
        potential = make_potential('es_charges.json', orders=lambda i: 0)
        coordinates, charges = point_charges()
        mol = butadiene()

        es = embedding.electrostatic(scf.RHF(mol), potential)
        es.conv_tol = 1e-12
        es.kernel()
        mm = qmmm.mm_charge(scf.RHF(mol), coordinates, charges, unit='Bohr')
        mm.conv_tol = 1e-12
        mm.kernel()

        self.assertAlmostEqual(es.e_tot, mm.e_tot, 10)
        self.assertAlmostEqual(
            abs(es.nuc_grad_method().kernel()
                - mm.nuc_grad_method().kernel()).max(), 0, 9)

    def test_no_induction_is_reported(self):
        mol = butadiene()
        mf = embedding.electrostatic(scf.RHF(mol), json_file)
        mf.conv_tol = 1e-12
        mf.kernel()
        with_embedding = mf.with_embedding
        contributions = with_embedding.energy_contributions()
        self.assertFalse(any('Induction' in label for label in contributions))
        self.assertAlmostEqual(with_embedding.e, with_embedding._e_es, 12)
        # the induced dipoles were never solved for
        self.assertAlmostEqual(
            abs(with_embedding.classical_subsystem.induced_dipoles
                .induced_dipoles).max(), 0.0, 12)

    def test_vdw_and_environment_energy_are_unchanged(self):
        '''Everything but the induction is that of polarizable embedding.'''
        potential = make_vdw_potential()
        mol = butadiene()
        es = embedding.electrostatic(mol, {'json_file': potential,
                                           'vdw': {'method': 'LJ'}})
        pe = embedding.polarizable(mol, {'json_file': potential,
                                         'vdw': {'method': 'LJ'}})
        self.assertAlmostEqual(es._e_rep, pe._e_rep, 12)
        self.assertAlmostEqual(es._e_disp, pe._e_disp, 12)
        self.assertAlmostEqual(es.environment_energy(),
                               pe.environment_energy(), 12)

    def test_option_validation(self):
        mol = butadiene()
        with self.assertRaises(ValueError) as caught:
            # the solver of the induced dipoles has nothing to act on
            embedding.electrostatic(mol, {'json_file': json_file,
                                          'induced_dipoles': {'solver': 'jacobi'}})
        self.assertIn('induced_dipoles', str(caught.exception))
        with self.assertRaises(TypeError):
            embedding.electrostatic(scf.RHF(mol))

    def test_the_model_of_the_object_has_to_match(self):
        '''A prebuilt object of the other model must not be attached silently.'''
        mol = butadiene()
        with self.assertRaises(TypeError):
            embedding.electrostatic(scf.RHF(mol),
                                    embedding.PolarizableEmbedding(mol, json_file))
        with self.assertRaises(TypeError):
            embedding.polarizable(scf.RHF(mol),
                                  embedding.ElectrostaticEmbedding(mol, json_file))
        # the matching one goes through
        obj = embedding.ElectrostaticEmbedding(mol, json_file)
        mf = embedding.electrostatic(scf.RHF(mol), obj)
        self.assertIs(mf.with_embedding, obj)


if __name__ == "__main__":
    print("Full Tests for Embedding")
    unittest.main()
