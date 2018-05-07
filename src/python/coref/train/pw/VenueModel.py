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


class VenueModel(Module):
    def __init__(self, config, vocab):
        """ Construct a router model

        :param config: the config to use
        :param vocab: the vocab to use
        """
        super(VenueModel, self).__init__()
        self.config = config
        self.vocab = vocab
        self.dim = 1

    def emb(self, routee, dest):
        """

        :param routee: 
        :param dest: 
        :return: Some kind of vector, you choose 
        """
        return [cosine_similarity(routee['v'], dest['v'])]
