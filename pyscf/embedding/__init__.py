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

'''Polarizable and electrostatic embedding in a molecular mechanics (MM) potential.

The submodules of this package require PyFraME, so they are imported lazily:
importing pyscf.embedding itself has to keep working when PyFraME is absent,
otherwise a bare `from pyscf import embedding` in a test module would fail
before the test could be skipped.
'''

__all__ = ['polarizable', 'electrostatic',
           'PolarizableEmbedding', 'ElectrostaticEmbedding']


def __getattr__(name):
    # PEP 562. Gives `embedding.PolarizableEmbedding` without importing PyFraME
    # at package import time.
    if name in ('EmbeddingBase', 'PolarizableEmbedding',
                'ElectrostaticEmbedding'):
        from pyscf.embedding import embedding
        return getattr(embedding, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def polarizable(method_or_mol, solvent_obj=None, dm=None):
    '''Initialize polarizable embedding model.

    The environment responds to the density: the multipoles of the potential
    are accompanied by the dipoles they induce, solved for at every SCF
    iteration. See electrostatic for the model without that response.

    Args:
        method_or_mol (pyscf method object or gto.Mole object)
            An SCF method, a method built on one (MP2, CISD, CCSD, CASCI,
            CASSCF), or a gto.Mole object, in which case this function returns
            a PolarizableEmbedding object constructed with it.
        solvent_obj (PolarizableEmbedding object, dict of options, or the
            potential itself)
            If solvent_obj is an object of PolarizableEmbedding class, the
            embedding-enabled method will be created using solvent_obj.
            Otherwise a PolarizableEmbedding object will be created first with
            the solvent_obj, on top of which the embedding-enabled method will
            be created. The potential is the path of a PyFraME JSON file, a
            PyFraME MolecularSystem or Snapshot whose embedding potential has
            been created (built in memory, with no file in between), or the
            Subsystems of one; see PolarizableEmbedding for the options.
        dm (ndarray)
            If given, the environment is frozen at this density matrix: the
            potential is computed once and then held fixed, rather than
            responding to the density through the SCF iterations -- or, for a
            correlated method, rather than being relaxed against the correlated
            density.

    Examples:

    >>> options = {"potential": "potential.json"}
    >>> mf = polarizable(scf.RHF(mol), options)
    >>> mf.kernel()

    or, with the potential built by PyFraME in the same session,

    >>> project.create_embedding_potential(system)
    >>> mf = polarizable(scf.RHF(mol), system)
    '''
    from pyscf.embedding import embedding
    return _embedding(embedding.PolarizableEmbedding, method_or_mol, solvent_obj, dm)


def electrostatic(method_or_mol, solvent_obj=None, dm=None):
    '''Initialize electrostatic embedding model.

    Only the permanent multipoles of the potential are used; the
    polarizabilities it may carry are ignored, so the environment does not
    respond to the density. See polarizable for the model that includes the
    induced dipoles.

    Args:
        method_or_mol (pyscf method object or gto.Mole object)
            An SCF method, a method built on one (MP2, CISD, CCSD, CASCI,
            CASSCF), or a gto.Mole object, in which case this function returns
            an ElectrostaticEmbedding object constructed with it.
        solvent_obj (ElectrostaticEmbedding object, dict of options, or the
            potential itself)
            If solvent_obj is an object of ElectrostaticEmbedding class, the
            embedding-enabled method will be created using solvent_obj.
            Otherwise an ElectrostaticEmbedding object will be created first with
            the solvent_obj, on top of which the embedding-enabled method will
            be created. The potential is the path of a PyFraME JSON file, a
            PyFraME MolecularSystem or Snapshot whose embedding potential has
            been created (built in memory, with no file in between), or the
            Subsystems of one; see ElectrostaticEmbedding for the options.
        dm (ndarray)
            If given, the environment is frozen at this density matrix: the
            potential is computed once and then held fixed, rather than
            responding to the density through the SCF iterations -- or, for a
            correlated method, rather than being relaxed against the correlated
            density.

    Examples:

    >>> mf = electrostatic(scf.RHF(mol), "potential.json")
    >>> mf.kernel()
    '''
    from pyscf.embedding import embedding
    return _embedding(embedding.ElectrostaticEmbedding, method_or_mol, solvent_obj, dm)


def _embedding(cls, method_or_mol, solvent_obj, dm=None):
    '''Attach an embedding potential of class cls, or build the object itself.'''
    from pyscf.embedding import embedding
    from pyscf import gto, scf

    if solvent_obj is None and not isinstance(method_or_mol, (gto.mole.Mole,
                                                              scf.hf.SCF)):
        # A method built on an embedded SCF already carries the potential;
        # asking for it again would be asking for what is there. TDSCF is left
        # out: _for_tdscf wants to make its own copy, carrying the excited
        # state's flag rather than the ground state's.
        if not _is_tdscf(method_or_mol):
            solvent_obj = getattr(getattr(method_or_mol, '_scf', None),
                                  'with_embedding', None)
    if solvent_obj is None and not _is_tdscf(method_or_mol):
        raise TypeError(
            f'{cls.__name__} requires an embedding potential: a {cls.__name__} '
            'object, a dictionary of options, the path of a PyFraME JSON file, '
            'or a PyFraME system or Subsystems.')
    if (isinstance(solvent_obj, embedding.ElectrostaticEmbedding)
            and solvent_obj.with_induction != cls.with_induction):
        # Both classes take the same potential, so the model is chosen by the
        # object rather than by the function that is handed it -- silently
        # using the object's model would make the call read as the other one.
        raise TypeError(
            f'A {type(solvent_obj).__name__} object cannot be used to build a '
            f'{cls.__name__} method. Build the embedding object with the model '
            'you want, or pass the potential itself.')

    if isinstance(method_or_mol, gto.mole.Mole):
        if dm is not None:
            # Freezing is a property of the attachment to a method, not of the
            # model: there is no SCF here whose potential could be held fixed.
            raise TypeError(
                'dm can only be given when attaching the embedding to a '
                'method, not when building a %s object.' % cls.__name__)
        return cls(method_or_mol, solvent_obj)
    elif isinstance(method_or_mol, scf.hf.SCF):
        # Before the _scf branch below: a second-order SCF carries an _scf of
        # its own while being an SCF method itself.
        return embedding.embedding_for_scf(method_or_mol, solvent_obj, dm, cls=cls)
    elif _is_casscf(method_or_mol):
        # before CASCI: the two are separate classes, but a CASSCF is what a
        # user means by "the CAS method" and its environment rides on the macro
        # iterations rather than on a loop of its own
        return embedding.embedding_for_casscf(method_or_mol, solvent_obj, dm,
                                              cls=cls)
    elif _is_casci(method_or_mol):
        return embedding.embedding_for_casci(method_or_mol, solvent_obj, dm,
                                             cls=cls)
    elif _is_tdscf(method_or_mol):
        return embedding.embedding_for_tdscf(method_or_mol, solvent_obj, dm,
                                             cls=cls)
    elif getattr(method_or_mol, '_scf', None) is not None:
        return embedding.embedding_for_post_scf(method_or_mol, solvent_obj, dm,
                                                cls=cls)
    raise RuntimeError(f'{cls.__name__} for {method_or_mol} not available')


def _is_casscf(method):
    from pyscf import mcscf
    return isinstance(method, mcscf.mc1step.CASSCF)


def _is_casci(method):
    from pyscf import mcscf
    return isinstance(method, mcscf.casci.CASBase)


def _is_tdscf(method):
    '''TDSCF carries an _scf and would otherwise be taken for a correlated
    method, whose treatment of the environment is not its own.'''
    from pyscf import tdscf
    return isinstance(method, tdscf.rhf.TDBase)
