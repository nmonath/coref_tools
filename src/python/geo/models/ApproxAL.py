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
from scipy.spatial.distance import cdist

from coref.util.dist import _fast_norm_diff


class ApproxAL(object):
    """Implements the approximate average linkage model."""
    def __init__(self, l_child, r_child):
        """Set my left child and right child."""
        self.l_child = l_child
        self.r_child = r_child
        self.l_mat = np.array(self.l_child)
        self.r_mat = np.array(self.r_child)
        self.my_mat = np.array(self. l_child + self.r_child)

    def hallucinate_merge(self, other):
        """Return the merger of me and other."""
        l_child = self.l_child + self.r_child
        r_child = other.l_child + other.r_child
        return ApproxAL(l_child, r_child)

    def exact_e_score(self, other):
        """Pass in an AvgLink and return negative average distance."""
        dist = cdist(other.l_mat, other.r_mat)
        return -np.mean(dist)

    def quick_e_score(self, n1, n2, k=100):
        """Pass in an AvgLink and return negative average distance."""
        # print('Im HERE!')
        if len(n1.l_child) * len(n2.r_child) > k:
            return n1.approx_e_score(n2, k)
        else:
            dist = cdist(n1.my_mat, n2.my_mat)
            return -np.mean(dist)

        # sum_distances = 0.0
        # num_dists = 0.0
        # for c1 in other.l_child:
        #     for c2 in other.r_child:
        #         sum_distances += _fast_norm_diff(c1, c2)
        #         num_dists += 1.0
        #
        # return -sum_distances / num_dists

    def approx_e_score(self, other, k):
        """Pass in an AvgLink and return approx negative average distance."""
        sum_distances = 0.0
        num_dists = float(k)
        p1 = np.random.choice(np.arange(len(other.l_child)), k)
        p2 = np.random.choice(np.arange(len(other.r_child)), k)
        for i in range(k):
            sum_distances += _fast_norm_diff(other.l_child[p1[i]],
                                             other.r_child[p2[i]])
        return -sum_distances / num_dists

    def approx_e_score_new(self, other, k):
        """Pass in an AvgLink and return approx negative average distance."""
        sum_distances = 0.0
        num_dists = float(k)
        p1 = np.random.choice(np.arange(len(other.l_child)), k)
        p2 = np.random.choice(np.arange(len(other.r_child)), k)
        dim = len(other.l_child[0])
        p1_mat = np.zeros((k, len(other.l_child[0])))
        p2_mat = np.zeros((k, len(other.r_child[0])))
        for i in range(k):
            p1_mat[i] = other.l_child[p1[i]]
            p2_mat[i] = other.r_child[p2[i]]
            # sum_distances += _fast_norm_diff(other.l_child[p1[i]],
            #                                  other.r_child[p2[i]])
        sum_distances = np.mean(np.power(p1_mat - p2_mat, 2)) * dim
        return -sum_distances / num_dists

    def e_score(self, other, k=100):
        """Choose to either do approx or exact scoring."""
        if len(other.l_child) * len(other.r_child) > k:
            return self.approx_e_score(other, k)
        else:
            return self.exact_e_score(other)

    def my_e_score(self):
        """Return my score."""
        return self.e_score(self)

    def update(self, other):
        """Update myself with another box."""
        b = self.hallucinate_merge(other)
        self.l_child = b.l_child
        self.r_child = b.r_child
        self.l_mat = np.array(self.l_child)
        self.r_mat = np.array(self.r_child)
        self.my_mat = np.array(b.l_child + b.r_child)

    def new(self, point):
        """Create a new box around point."""
        return ApproxAL([point], [point])