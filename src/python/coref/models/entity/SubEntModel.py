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


class SubEntModel(Module):
    def __init__(self, config, vocab):
        """A Sub Entity model."""
        super(SubEntModel, self).__init__()
        self.config = config
        self.vocab = vocab
        self.dim = 7

    def sum_pw(self, entity):
        """1 if there is more than 1 distinct first names."""
        return float(entity['pw'])

    def min_pw(self, entity):
        """Return the minimum pairwise score in this entity."""
        return float(entity['pw_bb'][0])

    def max_pw(self, entity):
        """Return the minimum pairwise score in this entity."""
        return float(entity['pw_bb'][1])

    def avg_pw(self, entity):
        """Return the average pairwise score in this subentity."""
        return float(entity['pw']) / float(entity['count'])

    def my_pw(self, entity):
        """Return the pw cost to construct this subent."""
        return float(entity['my_pw'])

    def avg_e_score(self, entity):
        """Return the average entity score in this subentity."""
        return float(entity['es']) / float(entity['count'])

    def min_e_score(self, entity):
        """Return the average entity score in this subentity."""
        return float(entity['es_bb'][0])

    def max_e_score(self, entity):
        """Return the average entity score in this subentity."""
        return float(entity['es_bb'][1])

    def new_edges(self, entity):
        """Returns the number of new 'edges' introduced by this merge."""
        return float(entity['new_edges'])

    def emb(self, entity):
        """Get all features of entity."""
        fv = []
        # fv.append(self.sum_pw(entity))
        fv.append(self.avg_pw(entity))
        fv.append(self.min_pw(entity))
        fv.append(self.max_pw(entity))
        fv.append(self.my_pw(entity))
        fv.append(self.min_e_score(entity))
        fv.append(self.max_e_score(entity))
        fv.append(self.avg_e_score(entity))
        # fv.append(self.new_edges(entity))  # only comes into play for regress.
        return fv
