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

from pyscf import lib
from pyscf.lib import logger
from pyscf import scf
from pyscf.embedding._attach_embedding import _Embedding
from pyscf.grad import rhf as rhf_grad


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

    PyFraME's model sums every term, from the integrals of the model's
    EmbeddingIntegralDriver where the electrons enter. A frozen model
    contributes the gradient of the energy it reports: the environment's own
    at the density it is frozen at, plus the interaction of the difference
    between that density and this one with the potential.
    '''
    return embedding_obj.gradient(dm)
