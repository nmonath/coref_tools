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
import torch.nn as nn
from coref.router.Utils import activation_from_str


from coref.train.pw.NameModel import NameModel as PWNameModel
from coref.train.pw.TitleModel import TitleModel as PWTitleModel
from coref.train.pw.CoAuthorModel import CoAuthorModel as PWCoauthorModel
from coref.models.pw.EmailModel import EmailModel as PWEmailModel
from coref.models.pw.YearModel import YearModel as PWYearModel


class BasePairwiseModel(Module):

    def __init__(self,config, vocab):
        super(BasePairwiseModel, self).__init__()
        self.config = config
        self.vocab = vocab
        self.activation = activation_from_str(self.config.activation)

        self.pw_name_model = PWNameModel(config, vocab)
        self.pw_coauthor_model = PWCoauthorModel(config, vocab)
        self.pw_title_model = PWTitleModel(config, vocab)
        self.pw_email_model = PWEmailModel(config, vocab)
        self.pw_year_model = PWYearModel(config, vocab)
        self.pw_concat_dim = self.pw_name_model.dim + \
                             self.pw_coauthor_model.dim + \
                             self.pw_title_model.dim + \
                             self.pw_year_model.dim + \
                             self.pw_email_model.dim

        self.pw_mlp = nn.Sequential(nn.Linear(self.pw_concat_dim, self.pw_concat_dim), self.activation,
                                    nn.Linear(self.pw_concat_dim, 1))

        self.pw_linear = nn.Linear(self.pw_concat_dim, 1)

    def emb(self, m1, m2):
        name_emb = self.pw_name_model.emb(m1, m2)
        coauthor_emb = self.pw_coauthor_model.emb(m1, m2)
        title_emb = self.pw_title_model.emb(m1, m2)
        email_emb = self.pw_email_model.emb(m1, m2)
        year_emb = self.pw_year_model.emb(m1, m2)
        fv = []
        for l in [name_emb, coauthor_emb, title_emb, year_emb, email_emb]:
            for i, v in enumerate(l):
                fv.append(v)
        return fv
