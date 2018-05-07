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

import fasttext
import math
import numpy as np
import torch

from torch.nn import Module
from torch.autograd import Variable

from coref.models.SubModelLoaders import new_pairwise_model
from coref.models.SubModelLoaders import new_entity_model

class AuthorCorefModel(Module):
    """Author Coref Model specs from config files"""

    def __init__(self, config, vocab):
        super(AuthorCorefModel, self).__init__()
        self.config = config
        self.vocab = vocab

        self.pw_model = new_pairwise_model(config,vocab)
        self.sub_ent_model = new_entity_model(config, vocab)
        self.pw_concat_dim = self.pw_model.pw_concat_dim
        self.e_concat_dim = self.sub_ent_model.dim

        # to not break print-statements
        self.pw_output_layer = self.pw_model.pw_linear
        self.e_output_layer = self.sub_ent_model.e_linear

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
        fv = self.sub_ent_model.emb(entity)
        return Variable(torch.FloatTensor(fv))

    def e_score(self, entity):
        """Score an attribute projection.

        :param entity: the thing being routed (mention group)
        :return: torch.variable of the fit
        """
        concat = self.e_extract_features(entity)
        # if self.config.debug:
        #     print('[e_score] concat %s ' % concat)
        res = None
        if self.config.e_arch == 'mlp':
            res = self.sub_ent_model.e_mlp(concat)
        else:
            res = self.sub_ent_model.e_linear(concat)
        # if self.config.debug:
        #     print('[e_score] res %s' % res)
        return res

    def pw_extract_features(self, m1, m2):
        fv = self.pw_model.emb(m1,m2)
        return Variable(torch.FloatTensor(fv))

    def pw_score(self, m1, m2):
        """ Score the similarity of m1 and m2.

        :param m1: Attribute Projection of m1
        :param m2: Attribute Projection of m2
        :return: torch.variable of the fit
        """
        concat = self.pw_extract_features(m1, m2)
        if self.config.pw_arch == 'mlp':
            return self.pw_model.pw_mlp(concat)
        else:
            return self.pw_model.pw_linear(concat)

    def pw_score_mat(self, mat):
        """Score a matrix of pairwise feature vectors.

        :param mat: the matrix of feature vectors.
        :param arch: the type of model to use to score (string)
        :return: torch.variable of the fit
        """
        if self.config.pw_arch == 'mlp':
            if type(mat) is Variable:
                return self.pw_model.pw_mlp(mat)
            else:
                return self.pw_model.pw_mlp(Variable(mat))
        else:
            if type(mat) is Variable:
                return self.pw_model.pw_linear(mat)
            else:
                return self.pw_model.pw_linear(Variable(mat))

    def e_score_mat(self, mat):
        """Score a matrix of entity feature vectors.

        :param mat: the matrix of feature vectors.
        :param arch: the type of model to use to score (string)
        :return: torch.variable of the fit
        """
        if self.config.e_arch == 'mlp':
            if type(mat) is Variable:
                return self.sub_ent_model.e_mlp(mat)
            else:
                return self.sub_ent_model.e_mlp(Variable(mat))
        else:
            if type(mat) is Variable:
                return self.sub_ent_model.e_linear(mat)
            else:
                return self.sub_ent_model.e_linear(Variable(mat))

    @staticmethod
    def load(f):
        """Load from file"""
        x = torch.load(f)
        if x.config.fasttext:
            x.cached_ft = fasttext.load_model(x.config.fasttext)
        return x
