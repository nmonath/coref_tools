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


class EmailModel(Module):
    def __init__(self, config, vocab):
        """ Construct a router model

        :param config: the config to use
        :param vocab: the vocab to use
        """
        super(EmailModel, self).__init__()
        self.config = config
        self.vocab = vocab
        self.dim = 1

    def email_overlap(self, m1, m2):
        """1 if there is more than 1 distinct first names."""
        for e in m1['em']:
            if e in m2['em']:
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
        fv.append(self.email_overlap(m1, m2))
        return fv
