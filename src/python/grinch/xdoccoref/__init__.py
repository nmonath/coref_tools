"""
Copyright (C) 2018 IBM Corporation.
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

from grinch.util.Config import Config
from grinch.xdoccoref.Vocab import TypedVocab
from nltk.corpus import stopwords as sw
from grinch.xdoccoref.XDocModels import PretrainedNameOnly,BaseCNNScoringModel,AttnCNNScoringModel,LookupModel,DSLookupModel,AttnLookupModel
stopwords = set(sw.words('english'))
import torch

from grinch.xdoccoref.XDocInf import XDocNSWHAC


def build_model(config: Config,typed_vocab: TypedVocab):
    if config.model_name == 'PretrainedNameOnly':
        return PretrainedNameOnly(config)
    elif config.model_name == 'BaseCNNModel':
        return BaseCNNScoringModel(config, typed_vocab)
    elif config.model_name == 'AttnCNNModel':
        return AttnCNNScoringModel(config, typed_vocab)
    elif config.model_name == 'LookupModel':
        return LookupModel(config, typed_vocab)
    elif config.model_name == 'DSLookupModel':
        return DSLookupModel(config, typed_vocab)
    elif config.model_name == 'AttnLookupModel':
        return AttnLookupModel(config, typed_vocab)

def new_grinch(config,best_model=None,eval_mode=True):
    grinch = None

    def load_model(config,best_model):
        if not best_model:
            best_model = torch.load(config.best_model, map_location='cpu')
        if eval_mode:
            best_model = best_model.eval()
        if not config.use_cuda:
            if hasattr(best_model,'name_model'):
                best_model.name_model.config.use_cuda = False
            if hasattr(best_model,'context_model'):
                best_model.context_model.config.use_cuda = False
        else:
            best_model = best_model.cuda()
        return best_model
    if config.clustering_scheme == 'XDocNSWHAC':
        best_model = load_model(config,best_model)
        grinch = XDocNSWHAC(config,best_model)
    return grinch