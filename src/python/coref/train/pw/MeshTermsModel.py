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
from coref.router.Utils import cosine_similarity

class MeshTermsModel(Module):
    def __init__(self, config, vocab):
        """ Construct a router model

        :param config: the config to use
        :param vocab: the vocab to use
        """
        super(MeshTermsModel, self).__init__()
        self.config = config
        self.vocab = vocab
        self.dim = 1

    def emb(self, routee, dest):
        """

        :param routee: 
        :param dest: 
        :return: Some kind of vector, you choose 
        """
        # cosine = cosine_similarity(routee['mt'], (dest['mt']))
        # if self.config.use_cuda:
        #     return Variable(torch.cuda.FloatTensor([cosine]))
        # else:
        #     return Variable(torch.FloatTensor([cosine]))
        return [cosine_similarity(routee['mt'], dest['mt'])]
