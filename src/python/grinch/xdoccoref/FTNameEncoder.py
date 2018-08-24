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

from grinch.xdoccoref.MentEncoder import MentEncoder

class FTName(MentEncoder):
    def __init__(self, config, vocab, type_key,output_dim=3):
        super(FTName,self).__init__(config,vocab,type_key,use_pairwise=config.use_pairwise[type_key],output_dim=output_dim)
        from grinch.xdoccoref.PretrainedModels import build_ft
        self.cached_ft = build_ft()
        self.cached_vectors = dict()
        self.cached_counts = []

    def __getstate__(self):
        """Delete cached_ft so that we don't have to deal with serialization."""
        state = dict(self.__dict__)
        del state['cached_ft']
        del state['cached_vectors']
        return state

    def __setstate__(self, state):
        state['cached_ft'] = None
        state['cached_vectors'] = dict()
        state['cached_counts'] = []
        self.__dict__ = state

    def ft(self):
        """Load fasttext model if unloaded and return, otherwise return None."""
        if self.cached_ft:
            return self.cached_ft
        else:
            from grinch.xdoccoref.PretrainedModels import build_ft
            self.cached_ft = build_ft()
            return self.cached_ft

    def get_embedding_for_name(self,name):
        if name not in self.cached_vectors:
            emb = torch.from_numpy(self.ft().get_word_vector(name).astype(np.float32))
            self.cached_vectors[name]= emb
            return emb
        else:
            return self.cached_vectors[name]

    def feat_ments(self, batch):
        """
        :param batch: A list of EntMent Objects
        :return:
        """
        batch_size = len(batch)
        # get ids returns the ngram strings
        names = [ment.name_spelling for ment in batch]

        embs = torch.zeros(batch_size,self.ft().get_dimension())
        for i,name in enumerate(names):
            embs[i] = self.get_embedding_for_name(name)

        if self.config.use_cuda:
            embs = embs.cuda()

        mean_norm = torch.norm(embs, dim=1).unsqueeze(1)
        normed = torch.div(embs, mean_norm)
        return normed

    def score_singletons(self,bm1_vecs,bm2_vecs):
        score = torch.sum(torch.mul(bm1_vecs, bm2_vecs), dim=1)
        incompatible_name_scores = (score < 0.9).type(score.type())
        not_exact_match = (score != 1).type(score.type())
        overall = torch.cat([score.unsqueeze(1),incompatible_name_scores.unsqueeze(1),not_exact_match.unsqueeze(1)],dim=1)
        return overall

    def score_one_to_group(self,ent_vec,groups_vecs):
        score = torch.mm(groups_vecs, ent_vec.transpose(1, 0))
        incompatible_name_scores = (score < 0.9).type(score.type())
        not_exact_match = (score != 1).type(score.type())
        overall = torch.cat([score,incompatible_name_scores,not_exact_match],dim=1)
        return overall

    def function_on_pairs_batch(self,scores,counts,addl=None):
        """

        :param scores: B by K by K2
        :param scores: B by K by K2 mask
        :return: scalar
        """
        score = super(FTName,self).function_on_pairs_batch(scores,counts)
        incompatible_name_scores = (score < 0.9).type(score.type())
        not_exact_match = (score != 1).type(score.type())
        overall = torch.cat([score, incompatible_name_scores, not_exact_match], dim=1)
        return overall

    def function_on_pairs(self,scores,counts):
        """

        :param scores: M by N
        :return: scalar
        """
        score = super(FTName,self).function_on_pairs_batch(scores,counts)
        incompatible_name_scores = (score < 0.9).type(score.type())
        not_exact_match = (score != 1).type(score.type())
        overall = torch.cat([score, incompatible_name_scores, not_exact_match], dim=1)
        return overall