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


import numpy as np

from grinch.models.Centroid import Centroid
from grinch.models.Grinch import Grinch
from grinch.models.GNSWNode import GNSWNode
from grinch.models.nn.NSW import NSW


class CntrdGrinch(Grinch):
    def __init__(self,config):
        super(CntrdGrinch, self).__init__()
        self.config = config
        self._nn_struct = NSW(self.config.exact_nn,self.config.nn_k,
                              self.config.nsw_r,self.config.random_seed)

    def nn_struct(self):
        return self._nn_struct

    def e_score(self, ent1, ent2):
        """Centroid score is negative distance between vectors."""
        return -np.linalg.norm(ent1.centroid - ent2.centroid)

    def node_from_pt(self, pt):
        n = GNSWNode(Centroid(pt[0],1), lambda x,y: self.e_score(x.ent,y.ent), None)
        n.add_point(pt)
        return n

    def node_from_nodes(self, n1, n2):
        ent = n1.ent.copy()
        ent.merged_rep(n2.ent)
        n = GNSWNode(ent,lambda x,y: self.e_score(x.ent,y.ent),None)
        return n


    def k(self):
        return self.config.nn_k

