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
Attach Embedding to SCF
'''

from pyscf import lib
from pyscf.lib import logger


def _for_scf(mf, embedding_obj):
    '''Add embedding to SCF (HF and DFT) method.
    '''
    if isinstance(mf, _Embedding):
        mf.with_embedding = embedding_obj
        return mf

    sol_mf = SCFWithEmbedding(mf, embedding_obj)
    name = embedding_obj.__class__.__name__ + mf.__class__.__name__
    return lib.set_class(sol_mf, (SCFWithEmbedding, mf.__class__), name)


class _Embedding:
    pass


class SCFWithEmbedding(_Embedding):
    _keys = {'with_embedding'}

    def __init__(self, mf, embedding_obj):
        self.__dict__.update(mf.__dict__)
        self.with_embedding = embedding_obj

    def undo_embedding(self):
        '''Revert to the bare SCF method the embedding was attached to.'''
        cls = self.__class__
        name_mixin = self.with_embedding.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, SCFWithEmbedding, name_mixin))
        del obj.with_embedding
        return obj

    def to_gpu(self):
        # Without this, the generic conversion would hand back a GPU method
        # whose embedding potential is silently absent.
        raise NotImplementedError(
            'Embedding has no GPU implementation. Call '
            '.undo_embedding().to_gpu() to move the bare method to the GPU.')

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_embedding.check_sanity()
        self.with_embedding.dump_flags(verbose)
        return self

    def reset(self, mol=None):
        self.with_embedding.reset(mol)
        return super().reset(mol)

    def get_veff(self, mol=None, dm=None, *args, **kwargs):
        vhf = super().get_veff(mol, dm, *args, **kwargs)
        # kernel stores e and v on the embedding object itself
        e_embedding, v_embedding = self.with_embedding.kernel(dm)

        # NOTE: v_embedding should not be added to vhf in this place. This is
        # because vhf is used as the reference for direct_scf in the next
        # iteration. If v_embedding is added here, it may break direct SCF.
        return lib.tag_array(vhf, e_embedding=e_embedding, v_embedding=v_embedding)

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        with_embedding = self.with_embedding
        if with_embedding.e is not None:
            logger.info(self, '\n******** %s Energy Contributions ********',
                        with_embedding.__class__.__name__)
            for label, energy in with_embedding.energy_contributions().items():
                logger.info(self, '%s = %.15g', label, energy)
            logger.info(self, '\n')
        return super()._finalize()

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1,
                 diis=None, diis_start_cycle=None,
                 level_shift_factor=None, damp_factor=None, fock_last=None):
        if dm is None:
            dm = self.make_rdm1()

        # DIIS was called inside super().get_fock. v_embedding, as a function of
        # dm, should be extrapolated as well. To enable it, v_embedding has to be
        # added to the fock matrix before DIIS was called.
        if getattr(vhf, 'v_embedding', None) is None:
            vhf = self.get_veff(self.mol, dm)
        return super().get_fock(h1e, s1e, vhf + vhf.v_embedding, dm, cycle, diis,
                                diis_start_cycle, level_shift_factor, damp_factor,
                                fock_last)

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None:
            dm = self.make_rdm1()
        if getattr(vhf, 'e_embedding', None) is None:
            vhf = self.get_veff(self.mol, dm)
        e_tot, e_coul = super().energy_elec(dm, h1e, vhf)
        e_tot += vhf.e_embedding
        self.scf_summary['e_embedding'] = vhf.e_embedding.real
        logger.debug(self, 'Embedding Energy = %.15g', vhf.e_embedding)
        return e_tot, e_coul

    def nuc_grad_method(self):
        grad_method = super().nuc_grad_method()
        return self.with_embedding.nuc_grad_method(grad_method)

    Gradients = nuc_grad_method
