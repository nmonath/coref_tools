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
import ot
from geo.models.ApproxALVec import ApproxALVec


class Wasserstein(ApproxALVec):
    def __init__(self, mat, mask, max_num_samples):
        super(Wasserstein,self).__init__(mat,mask,max_num_samples)
        self.sinkhorn_reg = 2
        self.max_sinkhorn_iter = 100

    def quick_e_score(self, n1, n2):
        """Pass in an AvgLink and return negative average distance."""
        if n1.needs_update:
            n1._update()
        if n2.needs_update:
            n2._update()
        # rows i want 1 is num_samples by dim
        rows_i_want_1 = n1.mat
        # rows i want 2 is num_samples by dim
        rows_i_want_2 = n2.mat
        # compute the point cloud wasserstein distance between the normalized
        # distributions.
        M = cdist(rows_i_want_1, rows_i_want_2)
        a = np.ones(rows_i_want_1.shape[0]) / rows_i_want_1.shape[0]
        b = np.ones(rows_i_want_2.shape[0]) / rows_i_want_2.shape[0]
        dist = ot.sinkhorn2(a,b,M,self.sinkhorn_reg,method='sinkhorn_stabilized',numItermax=self.max_sinkhorn_iter)
        return -dist[0]

    def new(self, point, ment_id=None):
        """Create a new box around point."""
        mat = np.zeros((1, np.size(point)))
        mat[0] = point
        return Wasserstein(mat, None, self.max_num_samples)

    def hallucinate_merge(self, other):
        """Return the merger of me and other."""
        res = Wasserstein(None, None, self.max_num_samples)
        res.needs_update = True
        return res