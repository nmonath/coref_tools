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

from coref.util.dist import _fast_norm_diff


class AvgLink(object):
    """Implements the average linkage model."""
    def __init__(self, l_child, r_child):
        """Set my left child and right child."""
        self.l_child = l_child
        self.r_child = r_child

    def hallucinate_merge(self, other):
        """Return the merger of me and other."""
        l_child = self.l_child + self.r_child
        r_child = other.l_child + other.r_child
        return AvgLink(l_child, r_child)

    def e_score(self, other):
        """Pass in an AvgLink and return negative average distance."""
        sum_distances = 0.0
        num_dists = 0.0
        for c1 in other.l_child:
            for c2 in other.r_child:
                sum_distances += _fast_norm_diff(c1, c2)
                num_dists += 1.0

        return -sum_distances / num_dists

    def my_e_score(self):
        """Return my score."""
        return self.e_score(self)

    def update(self, other):
        """Update myself with another box."""
        b = self.hallucinate_merge(other)
        self.l_child = b.l_child
        self.r_child = b.r_child

    def new(self, point):
        """Create a new box around point."""
        return AvgLink([point], [point])