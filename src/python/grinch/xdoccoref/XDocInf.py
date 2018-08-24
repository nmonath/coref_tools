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

from grinch.models.nn.NSW import NSW
from grinch.xdoccoref.XDocEnt import XDocEnt
from grinch.xdoccoref.Vocab import TypedVocab
from grinch.xdoccoref.BaseXDocGrinchNode import BaseXDocGrinchNode
from grinch.models.HAC import NSWEHAC

class XDocNSWHAC(NSWEHAC):
    def __init__(self,config,sim_model):
        super(XDocNSWHAC, self).__init__(config,sim_model)
        self.config = config
        self.graft_beam = config.graft_beam
        self.single_elimination = config.single_elimination
        self.use_single_search = config.use_single_search
        self._nn_struct = NSW(self.config.exact_nn,self.config.nn_k,
                              self.config.nsw_r,self.config.random_seed,self.config.use_canopies)
        self.sim_model = sim_model
        self.typed_vocab = TypedVocab(self.config.vocab_file)
        self.typed_dims = config.typed_dims
        self.max_num_leaves = config.max_num_leaves
        self.collapsibles = [] if self.max_num_leaves > 0 else None

    def nn_struct(self):
        return self._nn_struct

    def e_score(self, ent1, ent2):
        if ent1.needs_update:
            ent1._update()
        if ent2.needs_update:
            ent2._update()
        score = self.sim_model.score(ent1,ent2)
        return score

    def e_score_vec(self, ent1, nodes):
        if len(nodes) > 0:
            if ent1.needs_update:
                ent1._update()
            for v in nodes:
                assert not v.ent.needs_update
            ent2mat = [n.ent for n in nodes]
            pw_scores = self.sim_model.one_against_many_score(ent1,ent2mat)
            assert len(nodes) == pw_scores.size(0)
            return pw_scores
        else:
            return []

    def node_from_pt(self, pt):
        from grinch.xdoccoref.PretrainedModels import hash_canopies
        ent = XDocEnt(self.sim_model.types,self.typed_dims,self.sim_model,init_pt=pt,use_pairwise=self.config.use_pairwise,
                                        use_cuda=self.config.use_cuda,k=self.config.typed_k_rep,
                      canopies=frozenset(hash_canopies([x for x in pt.name_character_n_grams if len(x) >=4])))
        n = BaseXDocGrinchNode(ent,
                               lambda x,y: self.e_score(x.ent,y.ent), lambda x,y: self.e_score_vec(x.ent,y),
                               self.config.max_degree)
        n.add_point(pt)
        n.ent.gnode = n
        return n

    def node_from_nodes(self, n1, n2):
        ent = n1.ent.copy()
        ent.merged_rep(n2.ent)
        n = BaseXDocGrinchNode(ent, lambda x, y: self.e_score(x.ent, y.ent), lambda x, y: self.e_score_vec(x.ent, y), self.config.max_degree)
        ent.gnode = n
        n.point_counter = n1.point_counter + n2.point_counter
        return n

    def k(self):
        return self.config.nn_k
