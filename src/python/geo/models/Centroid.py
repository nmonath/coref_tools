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

from coref.util.dist import _fast_norm_diff


class Centroid(object):
    """Implements the centroid model."""
    def __init__(self, _sum, num_pts):
        """Set my left child and right child."""
        self._sum = _sum
        self.num_pts = num_pts
        self.centroid = _sum / num_pts
        self.gnode = None

    def hallucinate_merge(self, other):
        """Return the merger of me and other."""
        return Centroid(self._sum + other._sum, self.num_pts + other.num_pts)

    def quick_e_score(self, n1, n2):
        """Return e_score of n1 and n2."""
        return n1.e_score(n2)

    def e_score(self, other):
        """Compute distance between my mean and other mean."""
        self.gnode.grinch.num_e_scores += 1
        return -_fast_norm_diff(self.centroid, other.centroid)

    def my_e_score(self):
        """Return my score."""
        raise Exception('I don\'t know how to do this.')

    def update(self, other):
        """Update myself with another centroid."""
        b = self.hallucinate_merge(other)
        self._sum = b._sum
        self.num_pts = b.num_pts

    def new(self, point,ment_id=None):
        """Create a new box around point."""
        return Centroid(point, 1)

    def as_vec(self):
        return self.centroid[np.newaxis, :]
