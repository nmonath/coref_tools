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


class BoxModel(object):
    """Implements a bounding box in n-dims."""
    def __init__(self, mns, mxs):
        """Set my mins and maxes."""
        self.mns = mns
        self.mxs = mxs

    def hallucinate_merge(self, other):
        """Return the merger of me and other."""
        mins = np.min(np.array([self.mns, other.mns]), axis=0)
        maxes = np.max(np.array([self.mxs, other.mxs]), axis=0)
        return BoxModel(mins, maxes)

    def e_score(self, box):
        """Pass in a BoxModel and return its negative log volumne."""
        return -np.sum(np.log(np.abs(box.mns - box.mxs)))

    def my_e_score(self):
        """Return my score."""
        return self.e_score(self)

    def update(self, other):
        """Update myself with another box."""
        b = self.hallucinate_merge(other)
        self.mns = b.mns
        self.mxs = b.mxs

    def new(self, point):
        """Create a new box around point."""
        return BoxModel(point, point)