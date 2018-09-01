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

class Vocab(object):
    def __init__(self,filename=None,self_dict=None,max_len=20,pad=False,OOV=True):
        self.size = 0
        self.w2id = dict()
        self.id2w = dict()
        self.OOV_INDEX = 1 if OOV else -1
        self.pad = pad
        self.OOV = OOV
        self.OOV_TOKEN = "<OOV>"
        self.padding_index = 0 if pad else -1
        self.max_len = max_len
        if filename:
            self.__dict__.update(json.load(open(filename)))
        elif self_dict:
            self.__dict__.update(self_dict)
        if self.OOV:
            self.w2id[self.OOV_TOKEN] = self.OOV_INDEX
            self.id2w[self.OOV_INDEX] = self.OOV_TOKEN
        if '1' not in self.id2w and not 1 in self.id2w:
            self.id2w['1'] = self.OOV_TOKEN
        print('OOV_INDEX = %s' % self.OOV_INDEX)
        print('padding_index = %s' % self.padding_index)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        if item in self.w2id:
            return self.w2id[item]
        else:
            return self.OOV_INDEX

class TypedVocab(object):

    def __init__(self,filename=None):
        if filename:
            jobj = json.load(open(filename))
            for t in jobj:
                print('Loading %s vocab' % t)
                self.__dict__[t] = Vocab(self_dict=jobj[t])

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def to_json(self):
        state = dict()
        for t in self.__dict__.keys():
            state[t] = filter_json(self.__dict__[t].__dict__)
        return json.dumps(filter_json(state),sort_keys=True)