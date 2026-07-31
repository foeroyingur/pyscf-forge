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

import numpy as np
import pyscf
from pyscf import embedding
from pyscf import dft, qmmm, scf

try:
    # Importing the package above does not pull in PyFraME; the submodule does.
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
            from pyscf import mp
            embedding.polarizable(mp.MP2(scf.RHF(mol)), json_file)

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

    def test_quantum_subsystem_size_is_checked(self):
        mismatched = pyscf.M(atom='C 0 0 0; O 0 0 1.2', basis='sto3g', verbose=0)
        with self.assertRaises(ValueError):
            embedding.polarizable(mismatched, json_file)


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
