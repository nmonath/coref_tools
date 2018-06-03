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

from coref.models.entity.BaseSubEntAvgModel import BaseSubEntAvgModel
from coref.models.entity.TitleModel import TitleModel
import numpy as np

class BasePlusTitle(BaseSubEntAvgModel):
    def __init__(self, config, vocab):
        """A Sub Entity model."""
        super(BasePlusTitle, self).__init__(config, vocab)
        self.title_model = TitleModel(config,vocab)
        self.dim = self.dim + self.title_model.dim
        self.init_e_model()

    def emb(self, entity):
        """Get all features of entity."""
        fv = []
        fv.extend(super().emb(entity))
        fv.extend(self.title_model.emb(entity))
        # if self.config.debug:
        #     print('== emb ==')
        #     print('==> name_model: %s' % self.name_model.emb(entity))
        #     print('==> sub_ent_model: %s' % self.sub_ent_model.emb(entity))
        #     print('==> title_model: %s' % self.title_model.emb(entity))
        #     print('== bme ==')
        return fv

    def update(self,self_aproj,other):
        for k in other.aproj.keys():
            self_aproj[k].update(other.aproj[k])

            # print('attr_proj_update')
            # print(self.aproj_sum)
            # print(other.aproj_sum)

        for k in other.aproj_sum.keys():
            if k in self_aproj.aproj_sum:
                self_aproj.aproj_sum[k] += other[k]
            else:
                self_aproj.aproj_sum[k] = np.copy(other[k])

        for k in other.aproj_bb.keys():
            if k in self_aproj.aproj_bb:
                mins = np.min(np.array([self_aproj.aproj_bb[k][0],
                                        other.aproj_bb[k][0]]), axis=0)
                maxs = np.max(np.array([self_aproj.aproj_bb[k][1],
                                        other.aproj_bb[k][1]]), axis=0)
                self_aproj.aproj_bb[k] = (mins, maxs)
            else:
                self_aproj.aproj_bb[k] = (np.copy(other[k][0]), np.copy(other[k][1]))