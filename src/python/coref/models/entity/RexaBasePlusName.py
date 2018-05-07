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

from coref.models.entity.BasePlusNameSubEnt import BasePlusNameSubEnt
from coref.models.entity.RexaBaseSubEnt import RexaBaseSubEnt

class RexaBasePlusName(RexaBaseSubEnt):
    """A pairwise author model (designed for Rexa)"""

    def __init__(self, config, vocab):
        super(RexaBasePlusName, self).__init__(config,vocab)
        self.sub_ent_model = BasePlusNameSubEnt(config, vocab)