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
from grinch.xdoccoref.Core import EntMent
from grinch.models.Ent import Ent

class MentEncoder(torch.nn.Module):
    def __init__(self, config, vocab, type_key, use_pairwise, output_dim=1):
        super(MentEncoder,self).__init__()
        self.config = config
        self.vocab = vocab
        self.output_dim = output_dim
        self.type_key = type_key
        self.use_pairwise = use_pairwise

    def c_ext_fv(self,ent: Ent, entity_model=None):
        """

        :param ent: an Ent object
        :return: a 1 by dim tensor
        """
        if entity_model:
            return entity_model(ent.typed_centroids[self.type_key])
        else:
            return ent.typed_centroids[self.type_key]

    def c_ext_fv_batch(self, ents, entity_model=None):
        """

        :param ent: an N element Ent list
        :return: a N by dim tensor
        """
        if self.config.use_cuda:
            fv = torch.zeros(len(ents), self.config.typed_dims[self.type_key]).cuda()
        else:
            fv = torch.zeros(len(ents), self.config.typed_dims[self.type_key])
        for idx, m in enumerate(ents):
            fv[idx, :] = m.typed_centroids[self.type_key]
        if entity_model:
            return entity_model(fv)
        else:
            return fv

    def pw_ext_fv(self,ent: Ent, entity_model=None):
        """

        :param ent: an Ent object
        :return: a 1 by dim tensor
        """
        if entity_model:
            return entity_model(ent.typed_mat[self.type_key]),ent.typed_mat_assign[self.type_key]
        else:
            return ent.typed_mat[self.type_key],ent.typed_mat_assign[self.type_key]

    def pw_ext_fv_batch(self, ents, entity_model=None):
        """

        :param ent: an N element Ent list
        :return: a N by dim tensor
        """
        # create a tensor that is Batch by K by D
        K = max([m.typed_mat['name_ft'].size(0) for m in ents])
        if self.config.use_cuda:
            fv = torch.zeros(len(ents), K, self.config.typed_dims[self.type_key]).cuda()
            mask = torch.zeros(len(ents), K).cuda()
        else:
            fv = torch.zeros(len(ents), K, self.config.typed_dims[self.type_key])
            mask = torch.zeros(len(ents), K)
        for idx, m in enumerate(ents):
            fv[idx, 0:m.typed_mat[self.type_key].size(0), :] = m.typed_mat[self.type_key]
            mask[idx, 0:m.typed_mat[self.type_key].size(0)] = m.typed_mat_assign[self.type_key]
        if entity_model:
            return entity_model(fv), mask
        else:
            return fv, mask

    def feat_ment(self, entMent: EntMent):
        return self.feat_ments([entMent])

    def feat_ments(self, batch):
        """ Update the EntMent representations for a batch of mentions.

        :param batch: a list of EntMent objects
        :return:
        """
        pass

    def init_ent(self,ent,ment):
        fv = self.feat_ment(ment).squeeze()
        ent.typed_sums[self.type_key] += fv
        ent.typed_centroids[self.type_key] += fv
        ent.typed_counts[self.type_key] += 1
        if self.use_pairwise:
            ent.typed_mat[self.type_key][0, :] = fv
            ent.typed_mat_assign[self.type_key][0] = 1
        else:
            ent.typed_mat[self.type_key] = None
            ent.typed_mat_assign[self.type_key] = None
        ent.mat = [ment.ment().attributes]

    def score_singletons(self,bm1_vecs,bm2_vecs):
        """ Given singleton entity representations provide a score
        or produce other embedding that would be fed to the entity
        model.

        Base behavior is dot product.

        :param bm1_vecs: N by D
        :param bm2_vecs: N by D
        :return: N by 1 or you choose.
        """
        score = torch.sum(torch.mul(bm1_vecs, bm2_vecs), dim=1).unsqueeze(1)

        if self.config.use_cosine_sim:
            norm1 = torch.norm(bm1_vecs, dim=1).unsqueeze(1)
            norm2 = torch.norm(bm2_vecs, dim=1).unsqueeze(1)
            normalizer = torch.mul(norm1, norm2)
            score = score / normalizer
        return score

    def score_one_to_group(self,ent_vec,groups_vecs):
        """ Score ent_vec against the batch groups_vecs

        :param ent_vec: 1 by D
        :param groups_vecs: N by D
        :return: N by output_dim
        """
        score = torch.mm(groups_vecs, ent_vec.transpose(1, 0))

        if self.config.use_cosine_sim:
            norm1 = torch.norm(ent_vec, dim=1).unsqueeze(1)
            norm2 = torch.norm(groups_vecs, dim=1).unsqueeze(1)
            normalizer = torch.mul(norm1.expand_as(norm2), norm2)
            score = score / normalizer
        return score

    def batch_singleton_scores(self, bm1, bm2, entity_model=None):
        """ Score the similarity for each pair in m1 and m2.

        :param bm1: List of N EntMents m1
        :param bm2: List of N EntMents m2
        :param entity_model: The entity model to be applied
            to each of the singleton subenntities
        :return: N element torch tensor
        """
        M = len(bm1)
        N = len(bm2)
        assert M == N
        # 2N by D torch tensor
        vecs = self.feat_ments(bm1 + bm2)
        if entity_model is not None:
            vecs = entity_model(vecs)
        # N by D torch tensor
        m1vecs = vecs[:N]
        # N by D torch tensor
        m2vecs = vecs[N:]
        score = self.score_singletons(m1vecs,m2vecs)
        return score

    def c_one_against_many_score(self,ent:Ent, other_ents,entity_model=None):
        """ Compare ent to other_ents

        :param ent: single entity
        :param other_ents: N element list
        :return: N by output_dim tensor
        """
        # 1 by D torch tensor
        m1features = self.c_ext_fv_batch([ent],entity_model)
        # N by D torch tensor
        m2features = self.c_ext_fv_batch(other_ents,entity_model)
        # N by output_dim torch tensor
        score = self.score_one_to_group(m1features,m2features)
        return score

    def c_score(self, ent1:Ent, ent2: Ent,entity_model=None):
        return self.c_one_against_many_score(ent1,[ent2],entity_model=entity_model)

    def function_on_pairs(self,scores,counts):
        """

        :param scores: M by N
        :return: scalar
        """
        summed = (scores * counts).sum()
        return summed / counts.sum()

    def function_on_pairs_batch(self,scores,counts,addl=None):
        """

        :param scores: B by K by K2
        :param scores: B by K by K2 mask
        :return: 1 by B
        """
        # average:
        # mask = mask.unsqueeze(1).expand_as(scores)
        wscores = torch.mul(scores,counts)
        summed = wscores.sum(dim=2).sum(dim=1)
        avgd = summed / (counts.sum(dim=2).sum(dim=1))
        if len(avgd.size()) == 1:
            avgd = avgd.unsqueeze(1)
        return avgd

    def pw_one_against_many_score(self,ent:Ent, other_ents,entity_model=None):
        """ Compare ent to other_ents

        :param ent: single entity
        :param other_ents: N element list
        :return: N by output_dim tensor
        """
        # K by D torch tensor, K element count
        m1vecs, m1counts = self.pw_ext_fv(ent,entity_model)
        m1counts = m1counts.unsqueeze(1)

        # N by K2 by D torch tensor, N by K mask
        m2vecs, m2counts = self.pw_ext_fv_batch(other_ents,entity_model)

        # N by K by K2
        scores = torch.matmul(m1vecs, m2vecs.transpose(2, 1))
        counts = torch.matmul(m1counts, m2counts.unsqueeze(2).transpose(2, 1))
        assert scores.size() == counts.size()

        addl = {'m1vecs': m1vecs, 'm2vecs': m2vecs}
        # N
        score = self.function_on_pairs_batch(scores, counts,addl)
        return score

    def pw_score(self, ent1:Ent, ent2: Ent):
        # M by D torch tensor, M element count
        m1vecs, m1counts = self.pw_ext_fv(ent1)
        # N by D torch tensor, N element count
        m2vecs, m2counts = self.pw_ext_fv(ent2)
        # M by N
        scores = torch.matmul(m1vecs, m2vecs.transpose(1, 0))
        counts = m1counts.unsqueeze(1) * m2counts.unsqueeze(0)
        assert scores.size() == counts.size()
        addl = {'m1vecs': m1vecs, 'm2vecs': m2vecs}
        # Score
        score = self.function_on_pairs(scores, counts,addl)
        return score

    def one_against_many_score(self,ent:Ent, other_ents,entity_model=None):
        if self.use_pairwise:
            return self.pw_one_against_many_score(ent,other_ents,entity_model)
        else:
            return self.c_one_against_many_score(ent,other_ents,entity_model)

    def score(self,ent1:Ent, ent2: Ent):
        if self.use_pairwise:
            return self.pw_score(ent1,ent2)
        else:
            return self.c_score(ent1,ent2)

class EntityModel(torch.nn.Module):
    def __init__(self,config,types,typed_vocabs, mention_encoders):
        """

        :param config: Config object
        :param types: list of strings
        :param typed_vocabs: typed vocab object
        :param mention_encoders: dict: name to encoder
        """
        super(EntityModel,self).__init__()
        self.types = types
        self.config = config
        self.typed_vocabs = typed_vocabs
        self.mention_encoders = mention_encoders if mention_encoders else dict()
        if mention_encoders:
            self.setup_encoders()

    def setup_encoders(self):
        self.encoder_order = [k for k in self.mention_encoders.keys()]
        self.ordered_encoders = [self.mention_encoders[k] for k in self.encoder_order]
        self.scoring_layer_dim = sum([self.mention_encoders[k].output_dim for k in self.encoder_order])
        for enc_name,enc in self.mention_encoders.items():
            self.add_module('ment_enc_' + enc_name,enc)

    def score_features(self,features):
        """

        :param features: B by scoring_layer_dim features
        :return: B by 1
        """
        pass

    def pairwise_parameters(self):
        return self.parameters()

    def entity_parameters(self):
        return self.parameters()

    def score(self,m1,m2,entity_model=None):
        """ Score a pair of sub-entities

        :param m1: Ent object
        :param m2: Ent object
        :return:
        """
        return self.one_against_many_score(m1,[m2],entity_model)

    def batch_singleton_scores(self, entMents1, entMents2, entity_model=None):
        """ Score a batch of singleton entity mentions

        :param entMents1: List of entMents
        :param entMents2: List of entMents
        :return:
        """
        feats_per_encoder = []
        for enc in self.ordered_encoders:
            feats_per_encoder.append(enc.batch_singleton_scores(entMents1, entMents2,entity_model))
        fvs = torch.cat(feats_per_encoder, dim=1)
        scores = self.score_features(fvs)
        return scores

    def one_against_many_score(self,ent,other_ents,entity_model=None):
        feats_per_encoder = []
        for enc in self.ordered_encoders:
            feats_per_encoder.append(enc.one_against_many_score(ent, other_ents,entity_model))
        fvs = torch.cat(feats_per_encoder, dim=1)
        scores = self.score_features(fvs)
        return scores

    def detach_and_set_to_none(self,m1s,m2s=None):
        pass

    def detach(self):
        for p in self.parameters():
            p.requires_grad = False
        return self

    def attach(self):
        for p in self.parameters():
            p.requires_grad = True
        return self

    def init_ent(self, ent, ment):
        for enc in self.ordered_encoders:
            enc.init_ent(ent,ment)

class EntityScoringModel(EntityModel):
    def __init__(self,config,types,typed_vocabs, mention_encoders):
        super(EntityScoringModel, self).__init__(config, types, typed_vocabs, mention_encoders)

    def score_features(self,features):
        return self.scoring_layer(features).squeeze(1)

    def setup_scoring_layer(self):
        if self.config.use_mlp_scoring_layer:
            self.scoring_layer = torch.nn.Sequential(torch.nn.Linear(self.scoring_layer_dim, self.scoring_layer_dim, bias=True), torch.nn.Tanh(), torch.nn.Linear(self.scoring_layer_dim, 1, bias=True))
        else:
            self.scoring_layer = torch.nn.Linear(self.scoring_layer_dim, 1, bias=True)

