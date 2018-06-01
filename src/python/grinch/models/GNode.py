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

from grinch.models.core.F1Node import F1Node

class GNode(F1Node):
    def __init__(self, ent, e_score_fn):
        super(GNode, self).__init__()
        self._score = None
        self.ent = ent
        self.e_score_fn = e_score_fn

    def score(self):
        if self._score is None:
            self._score = self.e_score_fn(self.children[0],self.children[1])
        return self._score

    def update_from_children(self):
        """Update from the children"""
        self.ent = self.children[0].ent.copy()
        self.ent.merged_rep(self.children[1].ent)
