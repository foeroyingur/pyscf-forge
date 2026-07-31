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


def polarizable(method_or_mol, solvent_obj=None):
    '''Initialize polarizable embedding model.

    The environment responds to the density: the multipoles of the potential
    are accompanied by the dipoles they induce, solved for at every SCF
    iteration. See electrostatic for the model without that response.

    Args:
        method_or_mol (pyscf method object or gto.Mole object)
            If method_or_mol is gto.Mole object, this function returns a
            PolarizableEmbedding object constructed with this Mole object.
        solvent_obj (PolarizableEmbedding object, dict of options, str or
            os.PathLike)
            If solvent_obj is an object of PolarizableEmbedding class, the
            embedding-enabled method will be created using solvent_obj.
            If solvent_obj is a dict, a str or a path, a PolarizableEmbedding
            object will be created first with the solvent_obj, on top of which
            the embedding-enabled method will be created.

    Examples:

    >>> options = {"json_file": "potential.json"}
    >>> mf = polarizable(scf.RHF(mol), options)
    >>> mf.kernel()
    '''
    from pyscf.embedding import embedding
    return _embedding(embedding.PolarizableEmbedding, method_or_mol, solvent_obj)


def electrostatic(method_or_mol, solvent_obj=None):
    '''Initialize electrostatic embedding model.

    Only the permanent multipoles of the potential are used; the
    polarizabilities it may carry are ignored, so the environment does not
    respond to the density. See polarizable for the model that includes the
    induced dipoles.

    Args:
        method_or_mol (pyscf method object or gto.Mole object)
            If method_or_mol is gto.Mole object, this function returns an
            ElectrostaticEmbedding object constructed with this Mole object.
        solvent_obj (ElectrostaticEmbedding object, dict of options, str or
            os.PathLike)
            If solvent_obj is an object of ElectrostaticEmbedding class, the
            embedding-enabled method will be created using solvent_obj.
            If solvent_obj is a dict, a str or a path, an ElectrostaticEmbedding
            object will be created first with the solvent_obj, on top of which
            the embedding-enabled method will be created.

    Examples:

    >>> mf = electrostatic(scf.RHF(mol), "potential.json")
    >>> mf.kernel()
    '''
    from pyscf.embedding import embedding
    return _embedding(embedding.ElectrostaticEmbedding, method_or_mol, solvent_obj)


def _embedding(cls, method_or_mol, solvent_obj):
    '''Attach an embedding potential of class cls, or build the object itself.'''
    from pyscf.embedding import embedding
    from pyscf import gto, scf

    if solvent_obj is None:
        raise TypeError(
            f'{cls.__name__} requires an embedding potential: a {cls.__name__} '
            'object, a dictionary of options, or the path of an MM potential '
            'json file.')
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
        return cls(method_or_mol, solvent_obj)
    elif isinstance(method_or_mol, scf.hf.SCF):
        return embedding.embedding_for_scf(method_or_mol, solvent_obj, cls=cls)
    raise RuntimeError(f'{cls.__name__} for {method_or_mol} not available')
