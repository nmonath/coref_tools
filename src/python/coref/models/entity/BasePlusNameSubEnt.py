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

from coref.models.entity.NameModel import NameModel
from coref.models.entity.BaseSubEntModel import BaseSubEntModel

class BasePlusNameSubEnt(BaseSubEntModel):
    def __init__(self, config, vocab):
        """A Sub Entity model."""
        super(BasePlusNameSubEnt, self).__init__(config,vocab)
        self.name_model = NameModel(config,vocab)
        self.dim = self.name_model.dim + self.dim
        self.init_e_model()

    def emb(self, entity):
        """Get all features of entity."""
        fv = []
        fv.extend(self.name_model.emb(entity))
        fv.extend(super().emb(entity))
        # if self.config.debug:
        #     print('== emb ==')
        #     print('==> name_model: %s' % self.name_model.emb(entity))
        #     print('==> sub_ent_model: %s' % self.sub_ent_model.emb(entity))
        #     print('== bme ==')
        return fv
