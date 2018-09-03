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

from grinch.models.GNSWNode import GNSWNode

class BaseXDocGrinchNode(GNSWNode):

    def __init__(self, ent, e_score_fn, e_score_fn_vec, max_degree=None):
        super(BaseXDocGrinchNode, self).__init__(ent, e_score_fn, max_degree)
        self.e_score_fn_vec = e_score_fn_vec
        self.point_counter = 1

    def score_group(self, query, others, offlimits, path):
        batch_size = 100
        query_canopies = query.canopies()
        ok_neighbors = [n for n in others if n not in offlimits and n not in path and query_canopies.isdisjoint(n.canopies())]
        for b in range(0,len(ok_neighbors),batch_size):
            start=b
            end=min(b+batch_size,len(ok_neighbors))
            scores = self.e_score_fn_vec(query,ok_neighbors[b:b+batch_size])
            for b_i,n_i in enumerate(range(start,end)):
                yield scores[b_i],ok_neighbors[n_i]

    def canopies(self):
        if self.ent.needs_update:
            self.ent._update()
        return self.ent.canopies

    def score(self):
        if self.ent.needs_update:
            self.ent._update()
        return super().score()

    def update_from_children(self):
        super().update_from_children()
        self.ent.gnode = self
        self.point_counter = sum([c.point_counter for c in self.children])


