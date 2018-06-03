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


from geo.models.ApproxALVec import ApproxALVec

import numpy as np
from scipy.spatial.distance import cdist
from scipy.misc import logsumexp

class CompleteVec(ApproxALVec):
    def __init__(self, mat, mask, max_num_samples):
        super(CompleteVec,self).__init__(mat,mask,max_num_samples)

    def quick_e_score(self, n1, n2):
        """Pass in an AvgLink and return negative average distance."""
        if n1.needs_update:
            n1._update()
        if n2.needs_update:
            n2._update()
        dists = cdist(n1.mat, n2.mat)
        return -np.max(dists)

    def new(self, point,ment_id=None):
        """Create a new box around point."""
        mat = np.zeros((1, np.size(point)))
        mat[0] = point
        return CompleteVec(mat, None,self.max_num_samples)

    def hallucinate_merge(self, other):
        """Return the merger of me and other."""
        res = CompleteVec(None,None,self.max_num_samples)
        res.needs_update = True
        return res


class CosineCompleteVec(ApproxALVec):
    def __init__(self, mat, mask, max_num_samples):
        super(CosineCompleteVec,self).__init__(mat,mask,max_num_samples)

    def quick_e_score(self, n1, n2):
        """Pass in an AvgLink and return negative average distance."""
        if n1.needs_update:
            n1._update()
        if n2.needs_update:
            n2._update()
        dists = cdist(n1.mat, n2.mat,'cosine')
        return -np.max(dists)

    def new(self, point,ment_id=None):
        """Create a new box around point."""
        mat = np.zeros((1, np.size(point)))
        mat[0] = point
        return CosineCompleteVec(mat, None,self.max_num_samples)

    def hallucinate_merge(self, other):
        """Return the merger of me and other."""
        res = CosineCompleteVec(None,None,self.max_num_samples)
        res.needs_update = True
        return res



class SingleVec(ApproxALVec):
    def __init__(self, mat, mask, max_num_samples):
        super(SingleVec,self).__init__(mat,mask,max_num_samples)

    def quick_e_score(self, n1, n2):
        """Pass in an AvgLink and return negative average distance."""
        if n1.needs_update:
            n1._update()
        if n2.needs_update:
            n2._update()
        dists = cdist(n1.mat, n2.mat)
        return -np.min(dists)

class LSEVec(ApproxALVec):
    def __init__(self, mat, mask, max_num_samples):
        super(LSEVec,self).__init__(mat,mask,max_num_samples)

    def quick_e_score(self, n1, n2):
        """Pass in an AvgLink and return negative average distance."""
        if n1.needs_update:
            n1._update()
        if n2.needs_update:
            n2._update()
        dists = -cdist(n1.mat, n2.mat)
        return logsumexp(dists)
