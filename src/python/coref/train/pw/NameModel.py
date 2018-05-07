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
import torch
from torch.autograd import Variable
from torch.nn import Module
from coref.router.Utils import cosine_similarity


class NameModel(Module):
    def __init__(self, config, vocab):
        """ Construct a router model

        :param config: the config to use
        :param vocab: the vocab to use
        """
        super(NameModel, self).__init__()
        self.config = config
        self.vocab = vocab
        self.dim = 8

    def fn_match(self,routee,dest):
        """ 1 if the first name appears in both and is not an initial"""
        for rname in routee['fn']:
            if len(rname) > 1:
                if rname in dest['fn']:
                    return 1.0
        return 0.0

    def fn_initial_match(self, routee, dest):
        """First names match on initial and are not different full names."""
        for rname in routee['fn']:
            if len(rname) == 1 and rname in dest['fn']:
                return 1.0
        for dname in dest['fn']:
            if len(dname) == 1 and dname in routee['fn']:
                return 1.0
        return 0.0

    # def fn_cosine_feature(self, routee, dest):
    #     return cosine_similarity(routee['fn'], dest['fn'])

    def fn_incompatible_feature(self, routee, dest):
        for rname in routee['fn']:
            if len(rname) > 1:
                for dname in dest['fn']:
                    if len(dname) > 1:
                        if rname != dname:
                            return 1.0
        return 0.0

    def mn_cosine_feature(self, routee, dest):
        return cosine_similarity(routee['mn'], dest['mn'])

    def mn_incompatible_feature(self, routee, dest):
        for rname in routee['mn']:
            if len(rname) > 1:
                for dname in dest['mn']:
                    if len(dname) > 1:
                        if rname != dname:
                            return 1.0
        return 0.0

    def mi_incompatible(self, m1, m2):
        intersection_size = len(m1['mn'].intersection(m2['mn']))
        if len(m1['mn']) != len(m2['mn']) or intersection_size != len(m1['mn']):
            return 1.0
        else:
            return 0.0

    def mi_match(self, m1, m2):
        # if m1['mn'] or m2['mn']:
        #     print("m1['mn'], m2['mn'], m1['mn'] == m2['mn']")
        #     print(m1['mn'], m2['mn'], m1['mn'] == m2['mn'])
        if len(m1['mn']) == 1 and len(m2['mn']) == 1 and list(m1['mn'])[0][0] == list(m2['mn'])[0][0]:
            return 1.0
        else:
            return 0.0

    def one_mn_missing(self, routee, dest):
        if len(routee['mn']) >= 1 and len(dest['mn']) == 0:
            return 1.0
        elif len(routee['mn']) == 0 and len(dest['mn']) >= 1:
            return 1.0
        else:
            return 0.0

    def both_mn_missing(self, m1, m2):
        if len(m1['mn']) == 0 and len(m2['mn']) == 0:
            return 1.0
        else:
            return 0.0

    def emb(self, routee, dest):
        """

        :param routee: 
        :param dest: 
        :return: Some kind of vector, you choose 
        """
        fv = []
        fv.append(self.fn_match(routee, dest))
        fv.append(self.fn_initial_match(routee, dest))
        fv.append(self.fn_incompatible_feature(routee, dest))
        fv.append(self.mn_incompatible_feature(routee, dest))
        fv.append(self.mi_incompatible(routee, dest))
        fv.append(self.mi_match(routee, dest))
        fv.append(self.one_mn_missing(routee, dest))
        fv.append(self.both_mn_missing(routee, dest))
        return fv
