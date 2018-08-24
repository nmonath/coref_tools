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
from grinch.models.Ent import Ent

class XDocEnt(Ent):

    def __init__(self, types, typed_dims, sim_model,
                 k=None, use_cuda=False,
                 typed_sums=None, typed_counts=None,
                 typed_centroids=None, typed_mat=None,
                 gnode=None, init_pt=None, canopies=None,
                 use_avg=True,use_cos_sim=True,use_pairwise=None):
        super(XDocEnt, self).__init__()
        self.k = k
        self.use_cuda = use_cuda
        self.use_pairwise = use_pairwise
        self.types = types
        self.use_avg = use_avg
        self.use_cos_sim = use_cos_sim
        self.typed_sums = typed_sums if typed_sums is not None else dict()
        self.typed_counts = typed_counts if typed_counts is not None else dict()
        self.typed_centroids = typed_centroids if typed_centroids is not None else dict()
        self.typed_dims = typed_dims
        self.sim_model = sim_model
        self.needs_update = False
        self.gnode = gnode
        self.mat = None
        self.typed_mat = typed_mat if typed_mat is not None else dict()
        self.typed_mat_assign = torch.zeros(1)if typed_mat is not None else dict()
        if len(self.typed_sums) == 0:
            self._init_mats()
        if init_pt:
            self.sim_model.init_ent(self,init_pt)
        self.canopies = canopies if canopies is not None else set()

    def cuda(self):
        if self.use_cuda:
            for t in self.types:
                self.typed_sums[t] = self.typed_sums[t].cuda()
                self.typed_centroids[t] = self.typed_centroids[t].cuda()
                if self.use_pairwise[t]:
                    self.typed_mat[t] = self.typed_mat[t].cuda()
                    self.typed_mat_assign[t] = self.typed_mat_assign[t].cuda()

    def _init_mats(self):
        for t in self.types:
            self.typed_sums[t] = torch.zeros(self.typed_dims[t])
            self.typed_centroids[t] = torch.zeros(self.typed_dims[t])
            if self.use_pairwise[t]:
                self.typed_mat[t] = torch.zeros(1, self.typed_dims[t])
                self.typed_mat_assign[t] = torch.ones(1)
            self.typed_counts[t] = 0.0
        self.cuda()

    def _update(self):
        to_update = []
        to_check = [self]
        while to_check:
            curr = to_check.pop(0)
            if curr.needs_update:
                to_update.append(curr)
                if curr.gnode.children:
                    to_check.append(curr.gnode.children[0].ent)
                    to_check.append(curr.gnode.children[1].ent)

        for i in range(len(to_update)-1,-1,-1):
            to_update[i]._single_update()

    def _single_update(self):
        if self.needs_update:
            self.needs_update = False
            c1, c2 = self.gnode.children[0].ent, self.gnode.children[1].ent
            assert not c1.needs_update and not c2.needs_update
            self.canopies = c1.canopies.union(c2.canopies)
            self._init_mats()
            for t in self.types:
                self.typed_sums[t] += c1.typed_sums[t]
                self.typed_sums[t] += c2.typed_sums[t]
                self.typed_counts[t] += c1.typed_counts[t]
                self.typed_counts[t] += c2.typed_counts[t]
                self.typed_centroids[t] = (self.typed_sums[t] / self.typed_counts[t]) if self.use_avg else self.typed_sums[t]
                if self.use_pairwise[t]:
                    self.typed_mat[t] = torch.cat([c1.typed_mat[t], c2.typed_mat[t]], dim=0)
                    self.typed_mat_assign[t] = torch.cat([c1.typed_mat_assign[t], c2.typed_mat_assign[t]], dim=0)

    def copy(self):
        n = XDocEnt(self.types, self.typed_dims, self.sim_model, k=self.k, use_cuda=self.use_cuda,use_cos_sim=self.use_cos_sim,use_avg=self.use_avg,use_pairwise=self.use_pairwise)
        n.needs_update = True
        return n

    def merged_rep(self, other):
        self.needs_update = True
        return self