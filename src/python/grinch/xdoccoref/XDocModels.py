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

from grinch.xdoccoref.MentEncoder import EntityScoringModel
from grinch.xdoccoref.Vocab import Vocab
from grinch.xdoccoref.FTNameEncoder import FTName
from grinch.xdoccoref.CNNEncoders import NameCNNEncoder,ContextCNNEncoder
from grinch.xdoccoref.AttentionEncoders import AttnEncoder
from grinch.xdoccoref.LookupEncoder import LookupEncoder

class PretrainedNameOnly(EntityScoringModel):
    def __init__(self,config):
        super(PretrainedNameOnly,self).__init__(config,['name_ft'],{'name_ft': Vocab(max_len=300)},None)
        self.ft_name = FTName(config,self.typed_vocabs['name_ft'],'name_ft')
        self.mention_encoders['name_ft'] = self.ft_name
        self.setup_encoders()
        self.setup_scoring_layer()

class BaseCNNScoringModel(EntityScoringModel):
    def __init__(self,config,typedVocab):
        super(BaseCNNScoringModel, self).__init__(config, ['name_ft', 'name', 'context'], typedVocab, None)
        self.typed_vocabs['name_ft'] = Vocab(max_len=300)
        self.ft_name = FTName(config,self.typed_vocabs['name_ft'],'name_ft')
        self.mention_encoders['name_ft'] = self.ft_name
        self.mention_encoders['name'] = NameCNNEncoder(config, self.typed_vocabs['name'])
        self.mention_encoders['context'] = ContextCNNEncoder(config, self.typed_vocabs['context'])
        self.setup_encoders()
        self.setup_scoring_layer()

class AttnCNNScoringModel(EntityScoringModel):
    def __init__(self,config,typed_vocab, mention_encoders=None):
        super(AttnCNNScoringModel, self).__init__(config, ['name_ft', 'name', 'context'], typed_vocab, mention_encoders)
        self.typed_vocabs['name_ft'] = Vocab(max_len=300)
        self.ft_name = FTName(config,self.typed_vocabs['name_ft'],'name_ft')
        self.mention_encoders['name_ft'] = self.ft_name #if 'name_ft' not in mention_encoders or mention_encoders is None else mention_encoders['name_ft']
        self.mention_encoders['name'] = AttnEncoder(NameCNNEncoder(config, self.typed_vocabs['name'])) # if 'name' not in mention_encoders or mention_encoders is None else mention_encoders['name']
        self.mention_encoders['context'] = AttnEncoder(ContextCNNEncoder(config, self.typed_vocabs['context'])) #if 'context' not in mention_encoders or mention_encoders is None else mention_encoders['context']
        self.setup_encoders()
        self.setup_scoring_layer()

class LookupModel(EntityScoringModel):
    def __init__(self,config,typed_vocab, mention_encoders=None):
        super(LookupModel, self).__init__(config, ['name_ft', 'context'], typed_vocab, mention_encoders)
        self.typed_vocabs['name_ft'] = Vocab(max_len=300)
        self.ft_name = FTName(config,self.typed_vocabs['name_ft'],'name_ft')
        self.mention_encoders['name_ft'] = self.ft_name #if 'name_ft' not in mention_encoders or mention_encoders is None else mention_encoders['name_ft']
        self.mention_encoders['context'] = LookupEncoder(config, self.typed_vocabs['context'],'context') #if 'context' not in mention_encoders or mention_encoders is None else mention_encoders['context']
        self.setup_encoders()
        self.setup_scoring_layer()

