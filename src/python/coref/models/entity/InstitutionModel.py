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


class InstitutionModel(Module):
    def __init__(self, config, vocab):
        """Init."""
        super(InstitutionModel, self).__init__()
        self.config = config
        self.vocab = vocab
        self.dim = 3

    def num_is_over_num_ments(self, entity):
        """(max year - min year) / # menmtions"""
        num_ments = entity['count']
        if entity['i']:
            return len(entity['i']) / float(num_ments)
        else:
            return 0.0

    def lt1_i(self, entity):
        if entity['i'] and len(entity['i']) <= 1:
            return 1.0
        else:
            return 0.0

    def gt1_i(self, entity):
        if entity['i'] and len(entity['i']) > 1:
            return 1.0
        else:
            return 0.0

    def emb(self, entity):
        """

        :param routee:
        :param dest:
        :return: Some kind of vector, you choose
        """
        fv = []
        fv.append(self.num_is_over_num_ments(entity))
        fv.append(self.lt1_i(entity))
        fv.append(self.gt1_i(entity))
        return fv
