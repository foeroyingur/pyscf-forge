#!/usr/bin/env python
'''
Polarizable embedding with a potential built by PyFraME in the same script.

PyFraME partitions a structure into a core region and an environment and
builds the embedding potential of the environment. The system it did that on
can be handed to pyscf.embedding directly: its subsystems are built in memory,
with no JSON file written or read in between. The molecule of the calculation
is taken from the same system, i.e. the core region as PyFraME terminates it,
with its basis sets and its charge.

Requires PyFraME 0.5 or newer (https://gitlab.com/pyframe-project/pyframe).
'''

import os
import shutil
import tempfile

import pyframe

from pyscf import gto, scf
from pyscf import embedding
from pyscf.data.elements import ELEMENTS

# Butadiene with three waters around it.
PDB = '''\
HETATM    1 C1   BUT A   1      32.081  29.944  30.110  1.00  0.00           C
HETATM    2 C2   BUT A   1      30.771  30.374  29.820  1.00  0.00           C
HETATM    3 C3   BUT A   1      29.571  29.984  30.590  1.00  0.00           C
HETATM    4 C4   BUT A   1      28.361  30.394  30.239  1.00  0.00           C
HETATM    5 H1   BUT A   1      32.911  30.224  29.460  1.00  0.00           H
HETATM    6 H2   BUT A   1      32.181  29.274  30.950  1.00  0.00           H
HETATM    7 H3   BUT A   1      30.701  31.104  29.020  1.00  0.00           H
HETATM    8 H4   BUT A   1      29.571  29.434  31.540  1.00  0.00           H
HETATM    9 H5   BUT A   1      28.241  31.044  29.380  1.00  0.00           H
HETATM   10 H6   BUT A   1      27.441  30.054  30.720  1.00  0.00           H
HETATM   11 O    WAT A   2      29.651  33.114  32.150  1.00  0.00           O
HETATM   12 H1   WAT A   2      30.031  32.303  32.489  1.00  0.00           H
HETATM   13 H2   WAT A   2      30.131  33.284  31.340  1.00  0.00           H
HETATM   14 O    WAT A   3      30.461  31.074  33.630  1.00  0.00           O
HETATM   15 H1   WAT A   3      29.731  31.054  34.249  1.00  0.00           H
HETATM   16 H2   WAT A   3      31.241  30.914  34.169  1.00  0.00           H
HETATM   17 O    WAT A   4      29.981  26.914  29.420  1.00  0.00           O
HETATM   18 H1   WAT A   4      29.681  27.824  29.370  1.00  0.00           H
HETATM   19 H2   WAT A   4      29.611  26.594  30.250  1.00  0.00           H
END
'''

work_dir = tempfile.mkdtemp(prefix='pyframe-')
pdb = os.path.join(work_dir, 'butadiene_water.pdb')
with open(pdb, 'w') as handle:
    handle.write(PDB)

#
# Partition the structure and build the potential. The waters are described by
# the SEP standard potential, a charge and an isotropic polarizability per atom
# looked up in a table, so no quantum chemistry program is run for them.
#
system = pyframe.MolecularSystem(input_file=pdb, bond_threshold=0.15)
system.set_core_region(system.get_fragments_by_name(names=['BUT']),
                       basis='cc-pvdz')
system.add_region(name='solvent',
                  fragments=system.get_fragments_by_name(names=['WAT']),
                  use_standard_potentials=True, standard_potential_model='SEP')
pyframe.Project(work_dir=work_dir).create_embedding_potential(system)

#
# The molecule is the quantum subsystem of the potential, i.e. the core region
# with the link or capping atoms that terminate any bond the partitioning cuts
# (none here). Its nuclei are in Bohr, and every nucleus carries the basis set
# its region was given. A capping atom would also need its capping potential,
# given as an ECP.
#
_, (quantum_subsystem,), _ = system.to_subsystems()
labels = ['%s%d' % (ELEMENTS[int(nucleus.charge[0])], i)
          for i, nucleus in enumerate(quantum_subsystem.nuclei)]
mol = gto.M(
    atom=[(label, nucleus.coordinate)
          for label, nucleus in zip(labels, quantum_subsystem.nuclei)],
    unit='Bohr',
    basis=dict(zip(labels, quantum_subsystem.basis)),
    charge=quantum_subsystem.charge,
    verbose=4,
)

#
# The system itself is the potential. A JSON file written by PyFraME
# (project.write_potential(system, filetype='json')) gives the same result.
#
mf = embedding.polarizable(scf.RHF(mol), system)
mf.kernel()

shutil.rmtree(work_dir)
