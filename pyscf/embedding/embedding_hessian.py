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
Analytical nuclear Hessians for electrostatic and polarizable embedding.

The embedding energy enters the Hessian in three places:

    the second derivative at fixed density, d2E/dR_A dR_B, which is what
        kernel returns;
    the derivative of the embedding potential, dv/dR_A, which belongs in the
        first-order Fock matrix the CPHF equations are solved against, and is
        what fock_gradient returns;
    the environment's place in the orbital Hessian itself, which gen_response
        adds and EmbeddingHess.kernel switches on for the duration.

PyFraME's model gives the first two whole, induction included, from the
integrals of the model's EmbeddingIntegralDriver; the response of the induced
dipoles to the nuclei, which both need, is solved once for the two. What is
left here is PySCF's layout and the wrapper.
'''

from __future__ import annotations

import numpy as np

from pyscf import lib
from pyscf import scf
from pyscf.embedding._attach_embedding import _Embedding


def fock_gradient(embedding_obj, atmlst=None, dm=None):
    '''The derivative of the embedding potential with respect to the nuclei.

    Returns one (3, nao, nao) array per atom of atmlst, to be added to the
    first-order Fock matrix that the CPHF equations are solved against.

    Kwargs:
        dm : the density matrix, closed-shell or a pair of spin blocks. Needed
            for induction only, where the potential is that of the dipoles dm
            induces, and those move with a nucleus even at a fixed density.
    '''
    mol = embedding_obj.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    return embedding_obj.fock_derivatives(dm)[list(atmlst)]


def kernel(embedding_obj, dm, atmlst=None, verbose=None):
    '''The second derivative of the embedding energy at a fixed density matrix.

    Args:
        embedding_obj : the ElectrostaticEmbedding or PolarizableEmbedding
            carrying the potential.
        dm : the density matrix, either closed-shell or a pair of spin blocks,
            which is traced over spin.
        atmlst : the atoms the Hessian is computed for. Defaults to all.

    Returns:
        Shape (len(atmlst), len(atmlst), 3, 3), matching pyscf's Hessian layout.

    PyFraME returns the flat (3 natm, 3 natm) matrix over all nuclei, every
    term included; atmlst selects from it.
    '''
    mol = embedding_obj.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = list(atmlst)
    hess = embedding_obj.hessian(dm)
    hess = hess.reshape(mol.natm, 3, mol.natm, 3).transpose(0, 2, 1, 3)
    return hess[np.ix_(atmlst, atmlst)]


def make_hess_object(hess_method):
    '''Create a Hessian object including the embedding contributions.

    Args:
        hess_method : the Hessian object of an embedding-attached method, or
            the embedding-attached method itself.
    '''
    from pyscf.hessian.rhf import HessianBase
    if isinstance(hess_method, HessianBase):
        base_method = hess_method.base
    else:
        # For symmetry with pyscf.solvent, whose make_hess_object takes the
        # method rather than its Hessian object.
        base_method = hess_method
    if not isinstance(base_method, _Embedding):
        raise TypeError('%s does not carry an embedding potential.' % base_method)
    if not isinstance(base_method, scf.hf.SCF):
        raise NotImplementedError(
            'Embedding Hessians only implemented for SCF methods.')

    with_embedding = base_method.with_embedding
    if with_embedding.frozen:
        raise NotImplementedError(
            'Nuclear Hessians are not implemented for a frozen embedding '
            'potential. The energy is stationary, so the gradient is available, '
            'but its second derivative needs how the potential of the frozen '
            'density answers a displacement twice over -- see the notes on '
            'parity with pyscf.solvent.')

    vac_hess = base_method.undo_embedding().Hessian()
    vac_hess.base = base_method
    name = with_embedding.__class__.__name__ + vac_hess.__class__.__name__
    return lib.set_class(EmbeddingHess(vac_hess),
                         (EmbeddingHess, vac_hess.__class__), name)


class EmbeddingHess:
    '''The Hessian of a method in an embedding potential.

    The environment reaches the Hessian three ways, and they are kept apart
    here: its own second derivative at a fixed density (kernel, below), its
    place in the first-order Fock matrix (make_h1), and its place in the orbital
    Hessian that the CPHF equations are solved with (gen_response, switched on
    for the duration by equilibrium_solvation).
    '''

    _keys = {'de_classical_subsystem', 'de_quantum_subsystem'}

    def __init__(self, hess_method):
        self.__dict__.update(hess_method.__dict__)
        self.de_classical_subsystem = None
        self.de_quantum_subsystem = None

    def undo_embedding(self):
        '''Revert to the Hessian of the bare method.'''
        cls = self.__class__
        name_mixin = self.base.with_embedding.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, EmbeddingHess, name_mixin))
        del obj.de_classical_subsystem
        del obj.de_quantum_subsystem
        return obj

    def to_gpu(self):
        # See SCFWithEmbedding.to_gpu.
        raise NotImplementedError(
            'Embedding has no GPU implementation. Call '
            '.undo_embedding().to_gpu() to move the bare Hessian to the GPU.')

    def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        '''The Hessian of the embedded SCF energy.

        Kwargs:
            dm : the density matrix the embedding contributions are evaluated
                with. Defaults to that of the converged base method.
            atmlst : the atoms the Hessian is computed for. Defaults to all.
        '''
        if atmlst is not None:
            self.atmlst = atmlst
        atmlst = self.atmlst

        # The environment relaxes along with the density, so its response is
        # part of the orbital Hessian the CPHF equations are solved with.
        with lib.temporary_env(self.base.with_embedding,
                               equilibrium_solvation=True):
            self.de_quantum_subsystem = super().kernel(*args, atmlst=atmlst,
                                                       **kwargs)

        if dm is None:
            dm = self.base.make_rdm1(ao_repr=True)
        self.de_classical_subsystem = kernel(self.base.with_embedding, dm,
                                             atmlst=atmlst)
        self.de = self.de_quantum_subsystem + self.de_classical_subsystem
        return self.de

    hess = kernel

    def make_h1(self, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
        '''The first-order Fock matrix, with the embedding potential's share.

        The potential is a functional of the density, so displacing a nucleus
        changes it; that change belongs in the operator the CPHF equations are
        solved against, alongside the derivative of the core Hamiltonian.
        '''
        if atmlst is None:
            atmlst = range(self.mol.natm)
        h1ao = super().make_h1(mo_coeff, mo_occ, atmlst=atmlst, verbose=verbose)
        dm = self.base.make_rdm1(mo_coeff, mo_occ)
        dv = fock_gradient(self.base.with_embedding, atmlst, dm=dm)
        if isinstance(self.base, (scf.uhf.UHF, scf.rohf.ROHF)):
            # the environment sees the total density and answers both spin
            # blocks with the same potential
            h1ao_a, h1ao_b = h1ao
            for i0, ia in enumerate(atmlst):
                h1ao_a[ia] += dv[i0]
                h1ao_b[ia] += dv[i0]
            return h1ao_a, h1ao_b
        for i0, ia in enumerate(atmlst):
            h1ao[ia] += dv[i0]
        return h1ao
