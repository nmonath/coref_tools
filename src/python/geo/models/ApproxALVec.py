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


class ApproxALVec(object):
    """Implements the approximate average linkage model."""

    def __init__(self, mat, mask, max_num_samples):
        """Set my left child and right child."""
        self.mat = mat
        self.dim = np.size(self.mat, axis=1) if self.mat is not None else None
        self.samples = np.size(self.mat, axis=0) if self.mat is not None else None
        self.mask = mask
        self.ind = self.samples if self.mat is not None else None
        self.gnode = None
        self.needs_update = False
        self.max_num_samples = max_num_samples

    def _single_update(self):
        if self.needs_update:
            self.needs_update = False
            c1,c2 = self.gnode.children[0].e_model,self.gnode.children[1].e_model
            if c1.mat.shape[0] + c2.mat.shape[0] < self.max_num_samples:
                self.mat = np.vstack((c1.mat,c2.mat))
                self.samples = np.size(self.mat, axis=0)
                self.ind = self.samples
                self.dim = np.size(self.mat, axis=1)
            else:
                # Here we have to samples some guys from both sides and store them.
                mat = np.zeros((self.max_num_samples, c1.dim))
                ps = np.random.choice(np.arange(c1.ind + c2.ind), c1.samples,replace=False)
                for i, p in enumerate(ps):
                    if p < c1.ind:
                        mat[i] = c1.mat[i]
                    else:
                        assert p - c1.ind >= 0
                        mat[i] = c2.mat[p - c1.ind]
                self.mat = mat
                self.samples = np.size(self.mat, axis=0)
                self.ind = self.samples
                self.dim = np.size(self.mat, axis=1)

    def _update(self):
        to_update = []
        to_check = [self]
        while to_check:
            curr = to_check.pop(0)
            if curr.needs_update:
                to_update.append(curr)
                if curr.gnode.children:
                    to_check.append(curr.gnode.children[0].e_model)
                    to_check.append(curr.gnode.children[1].e_model)

        for i in range(len(to_update)-1,-1,-1):
            to_update[i]._single_update()


    def hallucinate_merge(self, other):
        """Return the merger of me and other."""
        res = ApproxALVec(None,None,self.max_num_samples)
        res.needs_update = True
        return res
        # if self.ind + other.ind < self.samples:
        #     mat = np.copy(self.mat)
        #     rows_i_want = other.mat[other.mask.astype(bool), :]
        #     start = self.ind
        #     end = start + other.ind
        #     mat[start:end] = rows_i_want
        #     mask = np.zeros(self.samples)
        #     mask[:end] = 1.0
        #     return ApproxALVec(mat, mask)
        # else:
        #     # Here we have to samples some guys from both sides and store them.
        #     mat = np.zeros((self.samples, self.dim))
        #     mask = np.ones(self.samples)
        #     ps = np.random.choice(np.arange(self.ind + other.ind), self.samples)
        #     for i, p in enumerate(ps):
        #         if p < self.ind:
        #             mat[i] = self.mat[i]
        #         else:
        #             assert p - self.ind >= 0
        #             mat[i] = other.mat[p - self.ind]
        #     return ApproxALVec(mat, mask)

    def quick_e_score(self, n1, n2):
        """Pass in an AvgLink and return negative average distance."""
        if n1.needs_update:
            n1._update()
        if n2.needs_update:
            n2._update()
        dists = -cdist(n1.mat, n2.mat)
        return np.mean(dists)

    def e_score(self, other):
        if self.needs_update:
            self._update()
        if other.needs_update:
            other._update()
        self.gnode.grinch.num_e_scores += 1
        return self.quick_e_score(self, other)

    def my_e_score(self):
        """Return my score."""
        raise Exception('This function don\'t work \'round here')

    def update(self, other):
        """Update myself with another box."""
        raise Exception('This function don\'t work \'round here')

    def new(self, point,ment_id=None):
        """Create a new box around point."""
        mat = np.zeros((1, np.size(point)))
        mat[0] = point
        return ApproxALVec(mat, None,self.max_num_samples)

    def as_vec(self):
        return self.mat


class CosineApproxALVec(ApproxALVec):
    def __init__(self, mat, mask, max_num_samples):
        super(CosineApproxALVec,self).__init__(mat,mask,max_num_samples)

    def quick_e_score(self, n1, n2):
        """Pass in an AvgLink and return negative average distance."""
        if n1.needs_update:
            n1._update()
        if n2.needs_update:
            n2._update()
        dists = cdist(n1.mat, n2.mat,'cosine')
        return -np.mean(dists)

    def new(self, point, ment_id=None):
        """Create a new box around point."""
        mat = np.zeros((1, np.size(point)))
        mat[0] = point
        return CosineApproxALVec(mat, None, self.max_num_samples)

    def hallucinate_merge(self, other):
        """Return the merger of me and other."""
        res = CosineApproxALVec(None, None, self.max_num_samples)
        res.needs_update = True
        return res