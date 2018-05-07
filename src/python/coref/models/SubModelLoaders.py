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
from coref.models.entity.BaseSubEntModel import BaseSubEntModel
from coref.models.entity.BasePlusNameSubEnt import BasePlusNameSubEnt
from coref.models.entity.BasePlusNamePlusTitle import BasePlusNamePlusTitle

from coref.models.entity.BasePairwiseModel import BasePairwiseModel
from coref.models.entity.PubMedPairwiseModel import PubMedPairwiseModel

def new_entity_model(config, vocab=None):
    """ Create a new model object based on the model_name field in the config

    :param config: 
    :return: 
    """
    if config.entity_model_name == 'BaseSubEnt' or config.entity_model_name == 'BaseSubEntModel':
        model = BaseSubEntModel(config, vocab)
    elif config.entity_model_name == 'BasePlusName':
        model = BasePlusNameSubEnt(config, vocab)
    elif config.entity_model_name == 'BasePlusNamePlusTitle':
        model = BasePlusNamePlusTitle(config, vocab)
    else:
        raise Exception("Unknown Model: {}".format(config.entity_model_name))
    return model

def new_pairwise_model(config, vocab=None):
    """ Create a new model object based on the model_name field in the config

    :param config: 
    :return: 
    """
    if config.pairwise_model_name == 'BasePairwiseModel':
        model = BasePairwiseModel(config, vocab)
    elif config.pairwise_model_name == 'PubMedPairwiseModel':
        model = PubMedPairwiseModel(config, vocab)
    else:
        raise Exception("Unknown Model: {}".format(config.pairwise_model_name))
    return model