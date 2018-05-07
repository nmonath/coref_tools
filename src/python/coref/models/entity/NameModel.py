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


class NameModel(Module):
    def __init__(self, config, vocab):
        """ Construct a router model

        :param config: the config to use
        :param vocab: the vocab to use
        """
        super(NameModel, self).__init__()
        self.config = config
        self.vocab = vocab
        self.dim = 4

    def gt1_fn(self, entity):
        """1 if there is more than 1 distinct first names."""
        names = {}
        for rname in entity['fn']:
            if len(rname) > 1:
                if rname not in names:
                    names[rname] = True
            # If entity has two different first names -> incompatible.
            if len(names.keys()) > 1:
                return 1.0
        return 0.0

    def num_fn_over_num_ments(self, entity):
        """1 if there is more than 1 distinct first names."""
        num_ments = entity['count']
        num_names = len(entity['fn'])
        return num_names / num_ments

    def num_fi_over_num_ments(self, entity):
        """1 if there is more than 1 distinct first names."""
        num_ments = entity['count']
        num_names = len(set([x[0].lower() for x in entity['fn'] if len(x) > 0]))
        return num_names / num_ments

    def one_fn(self, entity):
        """If there is 1 distinct first name."""
        names = {}
        for rname in entity['fn']:
            if len(rname) > 1:
                if rname not in names:
                    names[rname] = True
            # If entity has two different first names -> incompatible.
            if len(names.keys()) > 1:
                return 0.0
        # assert len(names.keys()) < 1
        return 1.0

    def gt1_mn(self, entity):
        """1 if there is more than 1 distinct middle names."""
        names = {}
        for rname in entity['mn']:
            if len(rname) > 1:
                if rname not in names:
                    names[rname] = True
            # If entity has two different first names -> incompatible.
            if len(names.keys()) > 1:
                return 1.0
        return 0.0

    def num_ments_over_uniq_names(self, entity):
        """1 if there is more than 1 distinct middle initial."""
        num_ments = entity['count']
        num_names = len(entity['fn'])
        return num_ments / num_names

    def gt1_fi(self, entity):
        """1 if the entity contains an initial that is not the start of a fn."""
        names = {}
        count = 0
        for rname in entity['fn']:
            if len(rname) > 0:
                if rname[0].lower() not in names:
                    names[rname[0].lower()] = True
                    count += 1
            if count > 1:
                return 1.0
        return 0.0

    def gt1_fn(self, entity):
        """1 if more than 1 first name."""
        names = {}
        count = 0
        for rname in entity['fn']:
            if len(rname) > 1:
                if rname[0].lower() not in names:
                    names[rname[0].lower()] = True
                    count += 1
            if count > 1:
                return 1.0
        return 0.0

    def gt1_mi(self, entity):
        """1 if the entity contains an initial that is not the start of a fn."""
        names = {}
        count = 0
        for rname in entity['mn']:
            if len(rname) > 0:
                if rname[0].lower() not in names:
                    names[rname[0].lower()] = True
                    count += 1
            if count > 1:
                return 1.0
        return 0.0

    def emb(self, entity):
        """

        :param routee:
        :param dest:
        :return: Some kind of vector, you choose
        """
        fv = []
        fv.append(self.one_fn(entity))
        fv.append(self.gt1_fn(entity))
        # fv.append(self.num_fn_over_num_ments(entity))
        # fv.append(self.num_fi_over_num_ments(entity))
        fv.append(self.gt1_mn(entity))
        fv.append(self.gt1_mi(entity))
        # fv.append(self.num_ments_over_uniq_names(entity))
        return fv
