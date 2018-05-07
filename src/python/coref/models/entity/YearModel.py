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


class YearModel(Module):
    def __init__(self, config, vocab):
        """Init."""
        super(YearModel, self).__init__()
        self.config = config
        self.vocab = vocab
        self.dim = 1

    def year_span_over_num_ments(self, entity):
        """(max year - min year) / # menmtions"""
        num_ments = entity['count']
        if entity['y']:
            years = [int(y) for y in entity['y']]
            return (max(years) - min(years)) / float(num_ments)
        else:
            return 0.0

    def emb(self, entity):
        """

        :param routee:
        :param dest:
        :return: Some kind of vector, you choose
        """
        fv = []
        fv.append(self.year_span_over_num_ments(entity))
        return fv
