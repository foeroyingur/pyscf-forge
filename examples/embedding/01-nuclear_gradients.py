#!/usr/bin/env python
'''
Analytical nuclear gradients and geometry optimization with polarizable
embedding.

The environment is held fixed: the gradient is taken with respect to the nuclei
of the quantum region only. The geometry of the quantum region is taken from the
Mole object rather than from the potential file, so the two need to describe the
same set of atoms in the same order.

Requires PyFraME:  pip install pyframe
'''

import os

from pyscf import gto, scf
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

mf = embedding.polarizable(scf.RHF(mol), potential)
mf.conv_tol = 1e-12
mf.kernel()

#
# The gradient is split into the contribution of the quantum region in vacuum
# and the contribution of its interaction with the environment; both are kept on
# the gradients object.
#
grad = mf.nuc_grad_method()
de = grad.kernel()
print('quantum subsystem  =\n', grad.de_quantum_subsystem)
print('classical subsystem =\n', grad.de_classical_subsystem)

#
# A subset of the atoms can be requested with atmlst, and a density matrix other
# than the converged one with dm.
#
print('first two atoms =\n', mf.nuc_grad_method().kernel(atmlst=[0, 1]))

#
# Geometry optimization. The environment does not move, and the potential is
# read only once: every step reuses it and only re-synchronizes the geometry of
# the quantum region.
#
# from pyscf.geomopt.geometric_solver import optimize
# optimized = optimize(mf, maxsteps=10)
# print('optimized geometry:\n', optimized.atom_coords())
