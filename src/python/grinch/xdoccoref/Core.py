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

import json

from grinch.util.Misc import filter_json

class EntMent(object):
    def __init__(self,mid=None,gt=None,doc=None):
        self.mid = mid
        self.gt = gt
        self.name_spelling = None
        self.name_character_n_grams = None
        self.name_character_n_grams_ids = None
        self.name_emb = None
        self.pretrained_name_emb = None
        self.context_char_ids = None
        self.context_emb = None
        self.context_string = None
        self.context_ids = None
        self.sentence_token_offsets = None
        self.sentence_tokens = None
        self.doc = doc

    def cuda(self):
        if self.name_emb is not None: # we allow it to be initially none
            self.name_emb = self.name_emb.cuda()
        if self.pretrained_name_emb is not None: # we allow it to be initially none
            self.pretrained_name_emb = self.pretrained_name_emb.cuda()
        if self.context_emb is not None:
            self.context_emb = self.context_emb.cuda()

    @staticmethod
    def new_from_json(jobj):
        newEntMent = EntMent()
        newEntMent.from_json(jobj)
        return newEntMent

    def from_json(self,jobj):
        self.__dict__ = jobj
        # TODO: Handle the name representations

    def to_json(self):
        return json.dumps(filter_json(self.__dict__),skipkeys=True,sort_keys=True)