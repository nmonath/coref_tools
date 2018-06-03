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

"""Implements a navigable small world graph (Malkov, 2014)."""
import numpy as np

from heapq import heappush, heappop, heappushpop
from scipy.spatial.distance import cdist
from coref.models.core.Node import Node


class VecNSWNode(Node):
    def __init__(self, k, score_fn, v, nsw=None):
        """Init a NSWnode.

        Args:
            k - (int) max number of children.
            score_fn - (func) takes two pts returns a score (higher is better).
            v - (obj) the type of object that this NSWNode takes.

        Returns:
            New BNode.
        """
        super().__init__()
        self.k = k
        self.score_fn = score_fn
        self.v = v
        self.vec = v.e_model.as_vec()[0, :]
        self.neighbors = list()
        self.nsw = nsw
        self.max_size = 200
        self.accepting_neighbors = True
        self.mat_size = int(1.5*k)
        self.dim = np.size(self.vec)
        self.neighbor_mats = np.zeros((self.mat_size, np.size(self.vec)))
        self.mask = np.zeros(self.mat_size)

    def expand_neighbor_mat(self):
        new_size = min(self.max_size,len(self.neighbors) + min(1000,len(self.neighbors)))
        if new_size == len(self.neighbors):
            self.accepting_neighbors = False
        else:
            print('#expanding\t%s\t%s' % (len(self.neighbors), new_size))
            new_neighbor_mats = np.zeros((new_size,self.dim))
            new_neighbor_mats[0:len(self.neighbors),:] = self.neighbor_mats
            self.neighbor_mats = new_neighbor_mats
            new_mask = np.zeros(new_size)
            new_mask[0:len(self.neighbors)] = self.mask
            self.mask = new_mask
            self.mat_size = new_size

    def nsw_parent(self):
        """Returns the NSW node of the parent of this node in the tree."""
        return [self.v.parent.nsw_node] if self.v.parent else []

    def nsw_children(self):
        """Returns the NSW nodes of the children of this node in the tree."""
        return [c.nsw_node for c in self.v.children]

    # def delete(self):
    #     for n in self.neighbors:
    #         n.neighbors.remove(self)
    #     self.neighbors = []
    #     self.nsw.nodes.remove(self)

    def add_link(self, other):
        """Add an edge between this and other."""
        my_num_neighs = len(self.neighbors)
        other_num_neighs = len(other.neighbors)
        if self.accepting_neighbors:
            if other not in self.neighbors:
                self.neighbor_mats[my_num_neighs, :] = other.vec
                self.neighbors.append(other)
                if len(self.neighbors) == self.mat_size:
                    self.expand_neighbor_mat()

        if other.accepting_neighbors:
            if self not in other.neighbors:
                other.neighbor_mats[other_num_neighs, :] = self.vec
                other.neighbors.append(self)
                if len(other.neighbors) == other.mat_size:
                    other.expand_neighbor_mat()

    def knn_and_score_vec(self, v, k=1, offlimits=None,sim_cache=None,path=set()):
        """Return approx k nearest neighbors of v and corresponding scores."""
        local_path = set()
        knn_not_offlimits = []
        curr = self
        score = curr.score_fn(v, curr.v)
        best = (score, curr)

        v_as_vec = v.e_model.as_vec()
        num_under_v = v_as_vec.shape[0]
        assert curr not in offlimits
        heappush(knn_not_offlimits, (score, curr))
        while True:   # We must always break out early.
            if curr in path:
                return None,None
            local_path.add(curr)
            curr.mask *= 0.0
            norder = []
            for i, c in enumerate(curr.neighbors):
                if c not in offlimits:
                    curr.mask[i] = 1.0
                    norder.append(c)
                    c.v.grinch.num_e_scores += 1
                else:
                    curr.mask[i] = 0.0

            # Here comes the select and cdist.
            rows_i_want = curr.neighbor_mats[curr.mask.astype(bool), :]
            dists = -np.mean(cdist(rows_i_want, v_as_vec), keepdims=False, axis=1)
            # dists = -np.sum(cdist(rows_i_want, v_as_vec), keepdims=False, axis=1) * (num_under_v + 1)
            # print('dists')
            # print(dists)
            # print('v_as_vec')
            # print(v_as_vec)
            # print('rows_i_want')
            # print(rows_i_want)
            # print('dists.shape %s rows_i_want.shape %s v_as_vec.shape %s ' % (dists.shape,rows_i_want.shape,v_as_vec.shape))
            for i in range(dists.shape[0]):
                score = dists[i]
                c = norder[i]
                # print('score %s c %s' % (score,c))
                if best[0] is None or best[0] < score or ( best[0] == score and best[1] < c):
                    best = (score, c)
                if len(knn_not_offlimits) == k:
                    heappushpop(knn_not_offlimits, (score, c))
                else:
                    heappush(knn_not_offlimits, (score, c))

            while len(knn_not_offlimits) > k:
                heappop(knn_not_offlimits)
            if best[1] == curr:
                print('#searchscore end')
                path.update(local_path)
                return knn_not_offlimits, 0
            else:
                curr = best[1]
                print('#searchscore\t%s' % best[0])
