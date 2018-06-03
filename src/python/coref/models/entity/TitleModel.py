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

from coref.util.dist import _fast_norm_diff


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
        self.dim = 1

    def norm_diff_min_avg(self, entity):
        """Norm of difference between min and average title emedding."""
        avg_emb = entity['tea']   # title embedding avg
        bb_emb = entity['tbb']
        if type(avg_emb) == set or type(bb_emb) == set:
            return 0.0
        else:
            return _fast_norm_diff(bb_emb[0], avg_emb)

    def norm_diff_max_avg(self, entity):
        """Norm of difference between max and average title emedding."""
        avg_emb = entity['tea']  # title embedding avg
        bb_emb = entity['tbb']
        if type(avg_emb) == set or type(bb_emb) == set:
            return 0.0
        else:
            return _fast_norm_diff(bb_emb[1], avg_emb)

    def norm_diff_max_min(self, entity):
        """Norm of difference between min and max title emedding."""
        bb_emb = entity['tbb']
        if type(bb_emb) == set:
            return 0.0
        else:
            return _fast_norm_diff(bb_emb[0], bb_emb[1])

    def emb(self, entity):
        """

        :param routee:
        :param dest:
        :return: Some kind of vector, you choose
        """
        fv = []
        # fv.append(self.norm_diff_max_avg(entity))
        # fv.append(self.norm_diff_min_avg(entity))
        fv.append(self.norm_diff_max_min(entity))
        return fv
