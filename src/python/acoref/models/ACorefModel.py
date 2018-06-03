"""
Copyright (C) 2018 University of Massachusetts Amherst.
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

import fastText as fasttext
import math
import numpy as np
import torch
import time

from torch.nn import Module
from torch.autograd import Variable

from coref.models.SubModelLoaders import new_pairwise_model
from coref.models.SubModelLoaders import new_entity_model
from coref.models.core.Ment import Ment

from coref.util.dist import _fast_norm_diff
from coref.models.core.AttributeProjection import AttributeProjection


from scipy.special import expit

class ACorefModel(object):
    """Implements the centroid model."""
    def __init__(self, config, torch_model, ment, gnode, num_pts=1, pair_to_pw=None):
        """Set my left child and right child."""
        self.config = config
        self.torch_model = torch_model
        self.ment = ment
        self.num_pts = num_pts
        self.gnode = gnode
        # Not sure I love this will be kept as a pointer on
        # each node
        if pair_to_pw:
            self.pair_to_pw = pair_to_pw
        else:
            self.pair_to_pw = {}

    def pairwise(self, other):
        """
            The pairwise distance between the ment at this node and the ment at the other node. 
             Asserts that both are leaves
        :param other: a ACorefModel 
        :return: numpy float distance
        """
        assert self.num_pts == 1
        assert other.num_pts == 1
        if (self.ment.id, other.ment.id) in self.pair_to_pw:
            return self.pair_to_pw[(self.ment.id, other.ment.id)]
        elif (other.ment.id, self.ment.id) in self.pair_to_pw:
            return self.pair_to_pw[(other.ment.id, self.ment.id)]
        else:
            pw = self.torch_model.pw_score(self.ment.attributes, other.ment.attributes).data.numpy()[0]
            self.pair_to_pw[(self.ment.id, other.ment.id)] = pw
            return pw

    def best_pairwise(self,other):
        if self.config.exact_best_pairwise:
            return self._exact_best_pairwise(other)
        else:
            return self._sampled_best_pairwise(other)

    def _exact_best_pairwise(self,other):
        self_leaves = self.gnode.leaves() if self.gnode else [self]
        other_leaves = other.gnode.leaves() if other.gnode else [other]
        best_pw = None
        best_pw_n1 = None
        best_pw_n2 = None
        end_time_in_leaves = time.time()
        count = 0
        start_time = time.time()
        for n1p in self_leaves:
            for n2p in other_leaves:
                if type(n1p) is ACorefModel:
                    assert n1p.num_pts == 1
                else:
                    n1p = n1p.e_model

                if type(n2p) is ACorefModel:
                    assert n2p.num_pts == 1
                else:
                    n2p = n2p.e_model
                pw = n1p.pairwise(n2p)
                if best_pw is None or best_pw < pw:
                    best_pw = pw
                    best_pw_n1 = n1p
                    best_pw_n2 = n2p
                count += 1
        end_time = time.time()
        return best_pw, best_pw_n1, best_pw_n2

    def _sampled_best_pairwise(self, other):
        start_time_in_leaves = time.time()
        best_pw = None
        best_pw_n1 = None
        best_pw_n2 = None
        budget = self.config.approx_best_pairwise_budget
        self_leaves = self.gnode.leaves()
        other_leaves = other.gnode.leaves()
        end_time_in_leaves = time.time()
        count = 0
        if len(self_leaves) * len(other_leaves) <= budget:
            start_time = time.time()
            for n1p in self_leaves:
                for n2p in other_leaves:
                    if type(n1p) is ACorefModel:
                        assert n1p.num_pts == 1
                    else:
                        n1p = n1p.e_model

                    if type(n2p) is ACorefModel:
                        assert n2p.num_pts == 1
                    else:
                        n2p = n2p.e_model
                    pw = n1p.pairwise(n2p)
                    if best_pw is None or best_pw < pw:
                        best_pw = pw
                        best_pw_n1 = n1p
                        best_pw_n2 = n2p
                    count += 1
            end_time = time.time()
        else:
            start_time = time.time()
            num_n1_leaves_minus_one = len(self_leaves) - 1
            num_n2_leaves_minus_one = len(other_leaves) - 1
            for i in range(budget):
                n1_l = self.config.random.randint(0, num_n1_leaves_minus_one)
                n2_l = self.config.random.randint(0, num_n2_leaves_minus_one)
                n1p = self_leaves[n1_l]
                n2p = other_leaves[n2_l]

                if type(n1p) is ACorefModel:
                    assert n1p.num_pts == 1
                else:
                    n1p = n1p.e_model

                if type(n2p) is ACorefModel:
                    assert n2p.num_pts == 1
                else:
                    n2p = n2p.e_model

                pw = n1p.pairwise(n2p)
                if best_pw is None or best_pw < pw:
                    best_pw = pw
                    best_pw_n1 = n1p
                    best_pw_n2 = n2p
                count += 1
            end_time = time.time()
        return best_pw, best_pw_n1, best_pw_n2


    def hallucinate_merge(self, other):
        """Return the merger of me and other."""
        start_time = time.time()
        ap = AttributeProjection()
        pw_score,_,_ = self.best_pairwise(other)
        ap.update(self.ment.attributes, self.torch_model.sub_ent_model)
        ap.update(other.ment.attributes, self.torch_model.sub_ent_model)

        num_ms = self.num_pts + other.num_pts
        if 'tes' in ap.aproj_sum:
            ap.aproj_sum['tea'] = ap['tes'] / num_ms

        ap.aproj_local['my_pw'] = pw_score
        ap.aproj_local['new_edges'] = self.num_pts * other.num_pts

        self_entity_score = 1.0
        other_entity_score = 1.0

        if self.num_pts > 1 and 'es' in self.ment.attributes.aproj_local:
            self_entity_score = self.ment.attributes.aproj_local['es']
            if self.config.expit_e_score:
                self_entity_score = expit(self_entity_score)
        else:
            assert self.num_pts == 1
        if other.num_pts > 1 and 'es' in other.ment.attributes.aproj_local:
            other_entity_score = other.ment.attributes.aproj_local['es']
            if self.config.expit_e_score:
                other_entity_score = expit(other_entity_score)
        else:
            assert other.num_pts == 1

        if self_entity_score >= other_entity_score:
            ap.aproj_local['child_e_max'] = self_entity_score
            ap.aproj_local['child_e_min'] = other_entity_score
        else:
            ap.aproj_local['child_e_max'] = other_entity_score
            ap.aproj_local['child_e_min'] = self_entity_score
        if self.config.expit_e_score:
            assert ap.aproj_local['child_e_max'] <= 1.0
            assert ap.aproj_local['child_e_min'] <= 1.0
            assert ap.aproj_local['child_e_max'] >= -0.0
            assert ap.aproj_local['child_e_min'] >= -0.0
        end_time = time.time()
        new_score = self.torch_model.e_score(ap).data.numpy()[0]
        new_node = ACorefModel(self.config,self.torch_model,Ment(ap,None),None,self.num_pts+other.num_pts,self.pair_to_pw)
        new_node.ment.attributes.aproj_local['es'] = new_score
        return new_node

    def quick_e_score(self, n1, n2):
        """Return e_score of n1 and n2. Returns a numpy float object."""
        return n1.e_score(n2)

    def e_score(self, other):
        """Compute distance between my mean and other mean."""
        new_node = self.hallucinate_merge(other)
        return new_node.ment.attributes.aproj_local['es']

    def my_e_score(self):
        """Return my score."""
        raise Exception('I don\'t know how to do this.')

    def update(self, other):
        """Update myself with another centroid."""
        b = self.hallucinate_merge(other)
        self.num_pts = b.num_pts

    def new(self, point):
        """Create a new model from a mention."""
        return ACorefModel(self.config,self.torch_model,point,None,1,self.pair_to_pw)