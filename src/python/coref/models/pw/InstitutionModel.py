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
        self.dim = 1

    def match(self, m1, m2):
        """(max year - min year) / # menmtions"""
        for i1 in m1['i']:
            if i1 in m2['i']:
                return 1.0
        else:
            return 0.0

    def emb(self, m1, m2):
        """

        :param routee:
        :param dest:
        :return: Some kind of vector, you choose
        """
        fv = []
        fv.append(self.match(m1, m2))
        return fv
