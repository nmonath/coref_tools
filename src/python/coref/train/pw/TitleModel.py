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

import numpy as np

from torch.nn import Module

from coref.router.Utils import cosine_similarity
from coref.util.dist import _fast_max_to_box, _fast_min_to_box


class TitleModel(Module):
    def __init__(self, config, vocab):
        """ Construct a router model

        :param config: the config to use
        :param vocab: the vocab to use
        :param ft: a fasttext loaded model
        """
        super(TitleModel, self).__init__()
        self.config = config
        self.vocab = vocab
        self.dim = 2

    def embbeded_sim(self, routee, dest):
        """Consine similarity between averaged word embeddings."""
        r_avg_emb = routee['tes']  # title embedding avg
        d_avg_emb = dest['tes']
        # r_avg_emb = routee['tes']  # title embedding sum
        # d_avg_emb = dest['tes']
        if type(r_avg_emb) == set or type(d_avg_emb) == set:
            return 0.0
        else:
            denom = np.linalg.norm(r_avg_emb) * np.linalg.norm(d_avg_emb)
            if denom == 0.0:
                return 0.0
            else:
                cs = np.dot(r_avg_emb, d_avg_emb) / denom
                return cs

    # def one_over_emb_min_d(self, routee, dest):
    #     """Min distance between embedding bounding box."""
    #     rbb = routee['tbb']   # title embedding avg
    #     dbb = dest['tbb']
    #     if type(rbb) == set or type(dbb) == set:
    #         return 0.0
    #     else:
    #         min_d1 = _fast_min_to_box(dbb[0], dbb[1], rbb[0])
    #         min_d2 = _fast_min_to_box(dbb[0], dbb[1], rbb[1])
    #         mind_d = min(min_d1, min_d2)
    #         return 1.0 / np.exp(mind_d)
    #
    # def one_over_emb_max_d(self, routee, dest):
    #     """Max distance between embedding bounding box."""
    #     rbb = routee['tbb']  # title embedding avg
    #     dbb = dest['tbb']
    #     if type(rbb) == set or type(dbb) == set:
    #         return 0.0
    #     else:
    #         max_d1 = _fast_max_to_box(dbb[0], dbb[1], rbb[0])
    #         max_d2 = _fast_max_to_box(dbb[0], dbb[1], rbb[1])
    #         max_d = max(max_d1, max_d2)
    #         return 1.0 / np.exp(max_d)

    def word_sim(self, routee, dest):
        """

        :param routee: 
        :param dest: 
        :return: Some kind of vector, you choose 
        """
        cosine = cosine_similarity(routee['t'], (dest['t']))
        return cosine

    def emb(self, routee, dest):
        """

        :param routee:
        :param dest:
        :return: Some kind of vector, you choose
        """
        fv = []
        fv.append(float(self.word_sim(routee, dest)))
        fv.append(float(self.embbeded_sim(routee, dest)))
        # fv.append(self.one_over_emb_max_d(routee, dest))
        # fv.append(self.one_over_emb_min_d(routee, dest))
        return fv
