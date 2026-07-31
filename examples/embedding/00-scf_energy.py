#!/usr/bin/env python
'''
Polarizable and electrostatic embedding for a self-consistent field method.

The environment is described by an MM potential -- a json file holding the
multipoles and polarizabilities of the classical sites, together with the
nuclei of the quantum region the potential was made for. Generating such a
potential is PyFraME's job (see https://gitlab.com/pyframe-project/pyframe).

Requires PyFraME:  pip install pyframe
'''

import os

from pyscf import gto, scf, dft
from pyscf import embedding

mol = gto.M(
    atom='''
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
    ''',
    basis='sto3g',
    verbose=4,
)

# The sample potential shipped with the tests. Replace it with your own; an
# MM potential is a json file describing the classical sites and the
# nuclei of the quantum region.
potential = os.path.join(os.path.dirname(embedding.__file__),
                         'test', 'butadiene_water.json')

#
# The potential can be given as a path, or as a dictionary of options in which
# it is the "json_file" entry. Both restricted and unrestricted references are
# supported, for Hartree-Fock as well as for DFT.
#
mf = embedding.polarizable(scf.RHF(mol), potential)
mf.kernel()

#
# The embedding energy and the potential added to the Fock matrix are kept on
# the with_embedding attribute, and the breakdown by contribution is printed at
# the end of the SCF. Note that the internal energy of the environment is
# reported but is not part of e_tot: it is a constant for a fixed environment.
#
print('Embedding energy  = %.12g' % mf.with_embedding.e)
for label, energy in mf.with_embedding.energy_contributions().items():
    print('  %s = %.12g' % (label.strip(), energy))

#
# The same for DFT.
#
mf = embedding.polarizable(dft.RKS(mol, xc='pbe0'), potential)
mf.kernel()

#
# Options other than the potential itself are passed in the dictionary form.
# "induced_dipoles" controls the iterative solution of the induced dipoles, and
# "vdw" switches on the Lennard-Jones repulsion and dispersion between the
# quantum region and the environment -- the latter requires a potential that
# carries the Lennard-Jones parameters. Its "combination_rule" names how the
# parameters of two sites are combined into a pair coefficient, as
# "<sigma rule>-<epsilon rule>": Lorentz or Good-Hope for sigma, Berthelot or
# Fender-Halsey for epsilon, defaulting to Lorentz-Berthelot. The powers of the
# potential come from the sites of the potential file itself, so a form other
# than 12-6 needs nothing here.
#
options = {
    'json_file': potential,
    'induced_dipoles': {'threshold': 1e-10, 'max_iterations': 200,
                        'solver': 'jacobi'},
    # 'vdw': {'method': 'LJ', 'combination_rule': 'Lorentz-Berthelot'},
}
mf = embedding.polarizable(scf.RHF(mol), options)
mf.kernel()

#
# The same potential can also be used without letting the environment respond
# to the density: embedding.electrostatic ignores the polarizabilities and keeps
# only the permanent multipoles, so there are no induced dipoles to solve for
# and no induction energy. The "induced_dipoles" option is not accepted there,
# the others are.
#
mf = embedding.electrostatic(scf.RHF(mol), potential)
mf.kernel()
print('Electrostatic embedding energy = %.12g' % mf.with_embedding.e)
