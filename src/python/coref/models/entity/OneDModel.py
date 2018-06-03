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

import fastText as fasttext
import math
import numpy as np
import torch

from torch.nn import Module, Linear
from torch.autograd import Variable

from coref.models.SubModelLoaders import new_pairwise_model
from coref.models.SubModelLoaders import new_entity_model

class OneDModel(Module):
    """Author Coref Model specs from config files"""

    def __init__(self, config, vocab):
        super(OneDModel, self).__init__()
        self.config = config
        self.vocab = vocab

        # self.pw_model = new_pairwise_model(config,vocab)
        # self.sub_ent_model = new_entity_model(config, vocab)
        # self.pw_concat_dim = self.pw_model.pw_concat_dim
        # self.e_concat_dim = self.sub_ent_model.dim

        # to not break print-statements
        # self.pw_output_layer = self.pw_model.pw_linear
        # self.e_output_layer = self.sub_ent_model.e_linear

        if self.config.fasttext:
            self.cached_ft = fasttext.load_model(self.config.fasttext)
        else:
            self.cached_ft = None
        if config.use_cuda:
            self.cuda()

    def __getstate__(self):
        """Delete cached_ft so that we don't have to deal with serialization."""
        state = dict(self.__dict__)
        del state['cached_ft']
        return state

    def __setstate__(self, state):
        state['cached_ft'] = None
        self.__dict__ = state

    def ft(self):
        """Load fasttext model if unloaded and return, otherwise return None."""
        if self.cached_ft:
            return self.cached_ft
        elif self.config.fasttext:
            self.cached_ft = fasttext.load_model(self.config.fasttext)
            return self.cached_ft
        else:
            return None

    def e_extract_features(self, entity):
        fv = []
        pw_score = float(entity['my_pw'])
        new_edges = int(entity['new_edges'])
        ms_count = int(entity['count'])
        size_diff = None
        for i in range(1, ms_count):
            if i * (ms_count - i) == new_edges:
                size_diff = -abs(i - (ms_count - i))
                break
        fv.append(pw_score + size_diff)
        return Variable(torch.FloatTensor(fv))

    def e_score(self, entity):
        """Score an attribute projection.

        :param entity: the thing being routed (mention group)
        :return: torch.variable of the fit
        """
        return self.e_extract_features(entity)

    def pw_extract_features(self, m1, m2):
        fv = []
        d1 = int(list(m1['d'])[0])
        d2 = int(list(m2['d'])[0])
        fv.append(-abs(d1 - d2))
        return Variable(torch.FloatTensor(fv))

    def pw_score(self, m1, m2):
        """Score the similarity of m1 and m2.

        :param m1: Attribute Projection of m1
        :param m2: Attribute Projection of m2
        :return: torch.variable of the fit
        """
        concat = self.pw_extract_features(m1, m2)
        return concat

    def pw_score_mat(self, mat):
        """Score a matrix of pairwise feature vectors.

        :param mat: the matrix of feature vectors.
        :param arch: the type of model to use to score (string)
        :return: torch.variable of the fit
        """
        ones = Linear(1, 1)
        return ones(mat)

    def e_score_mat(self, mat):
        """Score a matrix of entity feature vectors.

        :param mat: the matrix of feature vectors.
        :param arch: the type of model to use to score (string)
        :return: torch.variable of the fit
        """
        ones = Linear(1, 1)
        return ones(mat)

    @staticmethod
    def load(f):
        """Load from file"""
        x = torch.load(f)
        if x.config.fasttext:
            x.cached_ft = fasttext.load_model(x.config.fasttext)
        return x
