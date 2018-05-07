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

import torch
from torch.autograd import Variable
from torch.nn import Module
from coref.router.Utils import cosine_similarity


class CoAuthorModel(Module):
    def __init__(self, config, vocab):
        """ Construct a router model

        :param config: the config to use
        :param vocab: the vocab to use
        """
        super(CoAuthorModel, self).__init__()
        self.config = config
        self.vocab = vocab
        self.dim = 3

    def coauthor_overlap(self, routee, dest):
        """Returns 1 if routee and dest have at least 1 coauthor in common."""
        for n1 in routee['ca']:
            if n1 in dest['ca']:
                return 1.0
        return 0.0

    def coauthor_last_overlap(self, routee, dest):
        rl = [x.split()[-1] for x in routee['ca']]
        dl = [x.split()[-1] for x in dest['ca']]
        for r in rl:
            if r in dl:
                return 1.0
        return 0.0

    def emb(self, routee, dest):
        """

        :param routee: 
        :param dest: 
        :return: Some kind of vector, you choose 
        """
        cosine = cosine_similarity(routee['ca'], dest['ca'])
        overlap = self.coauthor_overlap(routee, dest)
        clo = self.coauthor_last_overlap(routee, dest)
        return [cosine, overlap, clo]
