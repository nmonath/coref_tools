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
        self.dim = 3

    def gt1_em(self, entity):
        """1 if there is more than 1 distinct first names."""
        if entity['em']:
            return 1.0 if len(entity['em']) > 1 else 0.0
        else:
            return 0.0

    def lt1_em(self, entity):
        if entity['em']:
            return 1.0 if len(entity['em']) <=1 else 0.0
        else:
            return 0.0

    def emails_over_num_ments(self, entity):
        num_ments = entity['count']
        if entity['em']:
            return len(entity['em']) / float(num_ments)
        else:
            return 0.0

    def emb(self, entity):
        """

        :param routee:
        :param dest:
        :return: Some kind of vector, you choose
        """
        fv = []
        fv.append(self.gt1_em(entity))
        fv.append(self.lt1_em(entity))
        fv.append(self.emails_over_num_ments(entity))
        return fv
