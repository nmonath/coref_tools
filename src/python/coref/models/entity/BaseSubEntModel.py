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

from torch.nn import Module
import torch
import torch.nn as nn

from coref.router.Utils import activation_from_str

class BaseSubEntModel(Module):
    def __init__(self, config, vocab):
        """A Sub Entity model."""
        super(BaseSubEntModel, self).__init__()
        self.config = config
        self.vocab = vocab
        self.dim = 4
        self.activation = activation_from_str(self.config.activation)
        self.e_mlp = None
        self.e_linear = None
        self.init_e_model()

    def init_e_model(self):
        self.e_mlp = nn.Sequential(nn.Linear(self.dim, self.dim), self.activation, nn.Linear(self.dim, 1))
        self.e_linear = nn.Linear(self.dim, 1)
        # (!) Initialize the weights to be 1-hot with the 1 on the PW feature.
        self.e_linear.weight.data = torch.zeros((1, self.dim))
        self.e_linear.weight.data[0][0] = 1.0
        self.e_linear.bias.data[0] = 0.0

    def my_pw(self, entity):
        """Return the pw cost to construct this subent."""
        return float(entity['my_pw'])

    def new_edges(self, entity):
        """Returns the number of new 'edges' introduced by this merge."""
        # NOTE 4/15/2018 - Removed by nbgm, 'new_edges' is still populated though.
        return 0.0
        # return float(entity['new_edges'])

    def ent_child_min_score(self, entity):
        """entity score of the left child."""
        return float(entity['child_e_min'])

    def ent_child_max_score(self, entity):
        """entity score of the right child"""
        return float(entity['child_e_max'])

    def emb(self, entity):
        """Get all features of entity."""
        fv = []
        fv.append(self.my_pw(entity))
        fv.append(self.new_edges(entity))  # only comes into play for regress.
        fv.append(self.ent_child_min_score(entity))
        fv.append(self.ent_child_max_score(entity))
        # if self.config.debug:
        #     print('== emb ==')
        #     print('==> my_pw: %s' % self.my_pw(entity))
        #     print('==> new_edges: %s' % self.new_edges(entity))
        #     print('==> ent_left: %s' % self.ent_left(entity))
        #     print('==> ent_right: %s' % self.ent_right(entity))
        #     print('== bme ==')
        return fv

    def update(self,self_aproj,other_aproj):
        pass


