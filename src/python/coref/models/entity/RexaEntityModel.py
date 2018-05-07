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
import torch

from torch.nn import Module
import torch.nn as nn
from torch.autograd import Variable
from coref.router.Utils import activation_from_str

from coref.models.entity.NameModel import NameModel as ENameModel
from coref.models.entity.CoauthorModel import CoauthorModel as ECoauthorModel
from coref.models.entity.TitleModel import TitleModel as ETitleModel


class RexaEntityModel(Module):
    """A pairwise author model (designed for Rexa)"""
    def __init__(self, config, vocab):
        super(RexaEntityModel, self).__init__()
        self.config = config
        self.vocab = vocab

        # Entity Model.
        self.name_model = ENameModel(config, vocab)
        self.coauthor_model = ECoauthorModel(config, vocab)
        self.title_model = ETitleModel(config, vocab)
        # self.venue_model = VenueModel(config, vocab)
        # self.institution_model = InstitutionModel(config, vocab)
        self.concat_dim = self.name_model.dim + \
                          self.coauthor_model.dim + \
                          self.title_model.dim
                          # self.venue_model.dim + \
                          # self.institution_model.dim

        self.hidden_layer = nn.Linear(self.concat_dim, self.concat_dim)
        self.output_layer = nn.Linear(self.concat_dim, 1)
        self.activation = activation_from_str(self.config.activation)
        self.config.use_cuda = self.config.use_cuda
        self.loss = torch.nn.BCEWithLogitsLoss()
        if self.config.fasttext:
            self.cached_ft = fasttext.load_model(self.config.fasttext)
        else:
            self.cached_ft = None
        if config.use_cuda:
            self.cuda()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          config.learning_rate,
                                          weight_decay=config.l2penalty)
        if self.config.use_cuda:
            self.fv = Variable(torch.cuda.FloatTensor(self.concat_dim).fill_(0))
        else:
            self.fv = Variable(torch.zeros(self.concat_dim))

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

    def linear(self, input_vec):
        # return self.linear_model(input_vec)

        out = self.output_layer(input_vec)
        # print("out")
        # print(out)
        return out

    def pw_linear(self, input_vec):
        # return self.linear_model(input_vec)

        out = self.pw_output_layer(input_vec)
        # print("out")
        # print(out)
        return out

    def mlp(self, input_vec):
        # print("input_vec")
        # print(input_vec)
        # print('self.hidden_layer.parameters()')
        # print(self.hidden_layer.parameters())
        h0_out = self.hidden_layer(input_vec)
        # print("h0_out")
        # print(h0_out)
        h0_out_nl = self.activation(h0_out)
        out = self.output_layer(h0_out_nl)
        # print("out")
        # print(out)
        return out

    def mlp(self, input_vec):
        # print("input_vec")
        # print(input_vec)
        # print('self.hidden_layer.parameters()')
        # print(self.hidden_layer.parameters())
        h0_out = self.pw_hidden_layer(input_vec)
        # print("h0_out")
        # print(h0_out)
        h0_out_nl = self.activation(h0_out)
        out = self.pw_output_layer(h0_out_nl)
        # print("out")
        # print(out)
        return out

    def extract_features(self, entity):
        name_emb = self.name_model.emb(entity)
        coauthor_emb = self.coauthor_model.emb(entity)
        title_emb = self.title_model.emb(entity)
        # venue_emb = self.venue_model.emb(m1.attributes, m2.attributes)
        # institution_emb = self.institution_model.emb(m1.attributes,
        #                                              m2.attributes)
        fv = []
        next_ind = 0
        # for l in [name_emb, coauthor_emb, title_emb, venue_emb,
        #           institution_emb]:
        for l in [name_emb, coauthor_emb, title_emb]:
            for i, v in enumerate(l):
                self.fv.data[next_ind + i] = v
                fv.append(v)
            next_ind += len(l)
        # return self.fv
        return Variable(torch.FloatTensor(fv))

    def score(self, entity, arch='lin'):
        """Score an attribute projection.

        :param entity: the thing being routed (mention group)
        :param arch: the type of model to use to score m1,m2 (string)
        :return: torch.variable of the fit
        """
        concat = self.extract_features(entity)
        if arch == 'mlp':
            return self.mlp_pw(concat)
        else:
            return self.linear_pw(concat)

    @staticmethod
    def load(f):
        """Load from file"""
        x = torch.load(f)
        if x.config.fasttext:
            x.cached_ft = fasttext.load_model(x.config.fasttext)
        return x
