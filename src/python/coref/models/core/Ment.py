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

import json
import uuid
from collections import defaultdict

import numpy as np

from coref.models.core.AttributeProjection import AttributeProjection


class Ment(object):
    def __init__(self, attributes, pack, edge_f=None, gt=None,mid=None):
        """Represents a mention, the base data object.

        For non-edit-mentions the attributes and pack(aging) are the same.

        Args:
            attributes - frozenset of attributes.
            pack - frozenset of attributes.
            edge_f - a function that scores this ment against future ments.
            gt - the id of the ground-truth entity this mention belongs to.
        """
        self.id = str(uuid.uuid4()) if mid is None else mid
        self.mid = mid
        self.attributes = attributes
        self.pack = pack
        self.edge_f = edge_f
        self.gt = gt
        self.is_feedback = False

    def to_json(self):
        """Return this mention id and attributes as json."""
        return json.dumps({'id': self.id,
                           'attrs': self.attributes.to_fmt_list(),
                           'pack': list(self.pack),
                           'gt': str(self.gt)})

    def from_json(self, s):
        """Modify this mention from a json record."""
        js = json.loads(s)
        self.id = js['id']
        self.attributes = Ment.attribute_proj_from_list(js['attrs'])
        self.pack = frozenset(js['pack']) if 'pack' in js else set()
        self.gt = js['gt'] if 'gt' in js else js['gt-dblp']
        self.mid = js['id']
        # print('gt')
        # print(self.gt)
        # temp debug
        # self.attributes.aproj['gt'].add(self.gt)
        return self

    def get_title_embs(self, ft, stop_words=None, idfs=None,use_new_ft_model_format=False):
        def word_emb(w):
            if use_new_ft_model_format:
                return ft.get_word_vector(w)
            else:
                return ft[w]

        # tea == title embedding average
        # tes == title embedding sum
        # if 'tea' not in self.attributes and ft is not None:
        #     assert('tes' not in self.attributes)
        if ft is not None:
            words = [y.lower() for x in self.attributes['t'] for y in x.split()]
            if stop_words is not None:
                words = [word for word in words if word not in stop_words]
            if idfs is not None:
                title_emb_sum = np.sum([np.array(word_emb(w)) * 1.0 / idfs[w]
                                        for w in words], axis=0)
            else:
                title_emb_sum = np.sum([np.array(word_emb(w)) for w in words], axis=0)
            if len(words) > 0:
                title_emb_avg = title_emb_sum / len(self.attributes['t'])
                title_emb_avg /= np.linalg.norm(title_emb_avg)
            else:
                # TODO(AK): is this value broad cast? Shouldn't it be the emb dim?
                title_emb_avg = np.zeros(ft.get_dimension())
                title_emb_sum = np.zeros(ft.get_dimension())
            self.attributes.aproj_sum['tes'] = title_emb_sum / np.linalg.norm(title_emb_sum) if len(words) > 0 else title_emb_sum
            self.attributes.aproj_sum['tea'] = title_emb_avg
            self.attributes.aproj_bb['tbb'] = (title_emb_avg, title_emb_avg)

    @staticmethod
    def attribute_proj_from_list(list_of_attrs):
        aproj = defaultdict(set)
        for attr in list_of_attrs:
            splt = attr.split(":", 1)
            if len(splt) == 2 and len(splt[1]) > 0:
                aproj[splt[0]].add(splt[1])
            else:
                print("Couldn't parse attr: {}".format(attr))
        return AttributeProjection(aproj)

    @staticmethod
    def load_ments(filename, model=None,skip_unlabeled=False):
        stop_words = set()
        with open('resources/stopwords-nltk.txt', 'r') as fin:
            for line in fin:
                stop_words.add(line.lower().strip())

        idfs = defaultdict(lambda: 1.0)
        with open('resources/idfs.tsv', 'r') as fin:
            for line in fin:
                splits = line.split('\t')
                idfs[splits[0]] = float(splits[1])

        with open(filename, 'r') as f:
            for line in f:
                m = Ment(-1, -1).from_json(line)
                if model is not None:
                    m.get_title_embs(model.ft(), stop_words=stop_words,
                                     idfs=idfs,use_new_ft_model_format=model.config.use_new_ft)
                m.attributes.aproj_sum['count'] = 1.0
                # subentity features
                m.attributes.aproj_sum['pw'] = 0.0
                m.attributes.aproj_bb['pw_bb'] = (0.0, 0.0)
                m.attributes.aproj_local['my_bb'] = 0.0

                # This conflicts with the name of the 'es' score in the aproj_local dictionary.
                # I don't think that this aproj_sum feature is used anymore.
                # if so we should rename the feature?
                # We've also I think changed the e model code to directly access the aproj_local.
                #m.attributes.aproj_sum['es'] = 0.0
                m.attributes.aproj_bb['es_bb'] = (0.0, 0.0)
                m.attributes.aproj_local['my_es'] = 0.0
                if m.gt != "None" or not skip_unlabeled:
                    yield ((m, m.gt, m.id))

class MentLoader(object):
    """The same as the prev Ment.load_ments but it persists the ids and stopwords so we don't need to load them
    all the time."""

    def __init__(self):
        self.stop_words = set()
        with open('resources/stopwords-nltk.txt', 'r') as fin:
            for line in fin:
                self.stop_words.add(line.lower().strip())

        self.idfs = defaultdict(lambda: 1.0)
        with open('resources/idfs.tsv', 'r') as fin:
            for line in fin:
                splits = line.split('\t')
                self.idfs[splits[0]] = float(splits[1])

    def load_ment(self,line, model=None,skip_unlabeled=False):
        m = Ment(-1, -1).from_json(line)
        if model is not None:
            m.get_title_embs(model.ft(), stop_words=self.stop_words,
                             idfs=self.idfs, use_new_ft_model_format=model.config.use_new_ft)
        m.attributes.aproj_sum['count'] = 1.0
        # subentity features
        m.attributes.aproj_sum['pw'] = 0.0
        m.attributes.aproj_bb['pw_bb'] = (0.0, 0.0)
        m.attributes.aproj_local['my_bb'] = 0.0

        # This conflicts with the name of the 'es' score in the aproj_local dictionary.
        # I don't think that this aproj_sum feature is used anymore.
        # if so we should rename the feature?
        # We've also I think changed the e model code to directly access the aproj_local.
        # m.attributes.aproj_sum['es'] = 0.0
        m.attributes.aproj_bb['es_bb'] = (0.0, 0.0)
        m.attributes.aproj_local['my_es'] = 0.0
        if m.gt != "None" or not skip_unlabeled:
            return ((m, m.gt, m.id))
        else:
            return None