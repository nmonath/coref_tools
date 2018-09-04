"""
Copyright (C) 2018 IBM Corporation.
This file is part of "coref_tools"
http://github.com/nmonath/coref_tools
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import numpy as np
import sys

from grinch.xdoccoref.MentEncoder import MentEncoder
from grinch.util.IO import lines

class LookupEncoder(MentEncoder):
    def __init__(self, config, vocab, type_key,output_dim=1):
        super(LookupEncoder,self).__init__(config,vocab,type_key,use_pairwise=config.use_pairwise[type_key],output_dim=output_dim)
        self.cached_vectors = dict()
        self.cached_counts = []
        self.dim = self.config.typed_dims[type_key]

    def __getstate__(self):
        """Delete cached_ft so that we don't have to deal with serialization."""
        state = dict(self.__dict__)
        del state['cached_vectors']
        return state

    def __setstate__(self, state):
        state['cached_ft'] = None
        state['cached_vectors'] = dict()
        state['cached_counts'] = []
        self.__dict__ = state

    def load(self):
        if len(self.cached_vectors) == 0:
            for idx,line in enumerate(lines(self.config.lookup_table)):
                if idx % 1000 == 0:
                    sys.stdout.write("\rRead %s lines of lookup table")
                splt = line.split("\t")
                self.cached_vectors[splt[0]] = np.array([float(x) for x in splt[1:]])

    def get_emb_for_ment(self,entMent):
        self.load()
        return self.cached_vectors[entMent.mid]

    def feat_ments(self, batch):
        """
        :param batch: A list of EntMent Objects
        :return:
        """
        batch_size = len(batch)
        # get ids returns the ngram strings
        embs = np.zeros((batch_size,self.dim),dtype=np.float32)
        for i,entMent in enumerate(batch):
            embs[i] = self.get_emb_for_ment(entMent)

        norm = np.linalg.norm(embs,axis=1,keepdims=True)
        norm[norm==0] = 1
        embs /= norm

        embs = torch.from_numpy(embs)

        if self.config.use_cuda:
            embs = embs.cuda()

        return embs