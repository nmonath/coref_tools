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
import random
import logging

from heapq import heapify, heappush, heappop, heappushpop

from grinch.models.nn.NSWNode import NSWNode
from collections import defaultdict

class NSW(object):
    """Builds a navigable small world nearest neighbor structure."""
    def __init__(self,  exact_nn, k, r, seed=1451,use_canopies=False):
        """Build the NSW graph.

        Args:
            dataset - a list of triples (record, label, id)
            score_fn - a fun to compute the score between two data points.
            k - (int) max number of initial connections for a new node.
            r - (int) number of nodes from which to start searches.

        Returns:
            None - but sets the root to be the root of the boundary tree.
        """
        self.exact_nn = exact_nn
        self.k = k
        self.r = r
        self.num_neighbor_edges = 0
        self.nodes = set()
        self.approx_max_degree = 0
        self.random = random.Random(seed)
        self.logger = logging.getLogger('NSW')
        self.logger.setLevel(logging.INFO)
        self.use_canopies = use_canopies
        self.canopy_map = defaultdict(list)


    def num_edges(self):
        return self.num_neighbor_edges + len(self.nodes)-1

    def __del__(self):
        """Delete all nodes from the NSW."""
        self.nodes = None

    def _knn(self, v, offlimits, k=1):
        """Returns the approximate nearest neighbor of v.

        Args:
            v - the query object.
            offlimits - a set of nodes that cannot be returned.
            k - (int) number of neighbors to retrieve.
            r - (int) the number of nodes to start the search from.

        Returns:
            A min heap of (score, node).
        """
        if self.use_canopies:
            nodes = set([node for c in v.canopies() for node in self.canopy_map[c]])
        else:
            nodes = self.nodes

        allowable = nodes.difference(offlimits)

        if self.use_canopies:
            offlimits = offlimits.union(self.nodes.difference(allowable))

        if len(allowable) == 0 \
                or k * self.r * np.log(len(allowable)) > len(allowable) or self.exact_nn:
            self.logger.debug('[_knn] Using exact NN.')
            scores_and_nodes, num_score_fn = self.exact_knn(v, offlimits, k)
        else:
            self.logger.debug('[_knn] Using approximate NN.')
            knn = set()
            num_score_fn = 0
            sim_cache = dict()
            if len(allowable) < self.r:
                roots = allowable
            else:
                roots = self.random.sample(allowable, self.r)
            path = set()
            for root in roots:
                logging.debug('[_knn] search starting from root %s' % root.id)
                knn_res, num_score_fn_root = root.cknn(v, k, offlimits, sim_cache,path)
                # print('#simcache\t%s' % len(sim_cache.keys()))
                if knn_res:
                    num_score_fn += num_score_fn_root
                    knn.update(knn_res)
            scores_and_nodes = list(knn)
            heapify(scores_and_nodes)
        for i in range(max(0, len(scores_and_nodes) - k)):
            heappop(scores_and_nodes)
        assert k >= len(scores_and_nodes)
        return scores_and_nodes, num_score_fn

    def cknn(self, v, offlimits, k=1):
        """Returns the approximate nearest neighbor of v.

        Args:
            v - the query object.
            offlimits - a set of nodes that cannot be returned.
            k - (int) number of neighbors to retrieve.
            r - (int) the number of nodes to start the search from.

        Returns:
            A sorted (descending) list of (score, node).
        """
        scores_and_nodes, num_score_fn = self._knn(v, offlimits, k)
        self.logger.debug('[cknn]\tnum_score_fn=%s\tbest_score=%s\tbest_node=%s\tv=%s\tlen(offlimits)=%s\tk=%s' % (
            num_score_fn,
            scores_and_nodes[0][0] if len(scores_and_nodes) > 0 else 'None',
            scores_and_nodes[0][1].id if len(scores_and_nodes) > 0 else 'None',
            v.id, len(offlimits), k))
        return sorted(scores_and_nodes, key=lambda x: (x[0],x[1]),reverse=True)

    def cknn_and_insert(self, v, offlimits, k=1):
        """Adds v to the NSW and returns its approximate 1-nearest neighbor.

        Args:
            v - the query object.
            offlimits - a set of nodes that cannot be returned.
            k - (int) number of neighbors to retrieve.
            r - (int) the number of nodes to start the search from.

        Returns:
            A list of no more than k nodes and their scores.
        """
        self.logger.debug('[cknn_and_insert]\tv=%s\tlen(offlimits)=%s\tk=%s' % (v.id,len(offlimits),k))
        if len(self.nodes) == 0:
            self.nodes.add(v)
            self.logger.debug('[cknn_and_insert]\tadded first node\tv=%s\tlen(offlimits)=%s\tk=%s' % (v.id, len(offlimits), k))
            return None
        else:
            scores_and_nodes, num_score_fn = self._knn(v, offlimits, k)
            best = (None, None)
            self.nodes.add(v)
            if self.use_canopies:
                for canopy in v.canopies():
                    self.canopy_map[canopy].append(v)
            for score, node in scores_and_nodes:
                v.add_link(node)
                if best[0] is None or score > best[0] or (score == best[0] and best[1] < node):
                    best = (score, node)
            if best[0] is None:
                self.logger.info('[cknn_and_insert]\tno edges added\tv.id=%s\tlen(v.canopies())=%s' % (v.id,len(v.canopies())))
                return []
            else:
                sorted_nodes = sorted(scores_and_nodes, key=lambda x: (x[0], x[1]), reverse=True)
                self.logger.debug('[cknn_and_insert]\tbest\tbest_score=%s\tbest_node=%s\tv=%s\tlen(offlimits)=%s\tk=%s' % (best[0],best[1].id, v.id, len(offlimits), k))
                assert best == sorted_nodes[0]
                return sorted_nodes

    def exact_knn(self, v, offlimits,  k=1):
        """Returns the exact knn of v.

        Args:
            v - the query object.
            offlimits - a set of nodes that cannot be returned.
            k - (int) number of neighbors to retrieve.

        Returns:
            A minheap of no more than k (score,node).
        """
        if self.use_canopies:
            nodes = [node for c in v.canopies() for node in self.canopy_map[c]]
        else:
            nodes = self.nodes
        knn = []
        num_score_fn = 0
        for n in nodes:
            if n not in offlimits:
                num_score_fn += 1
                if len(knn) == k:
                    heappushpop(knn, (n.e_score_fn(v, n), n))
                else:
                    heappush(knn, (n.e_score_fn(v, n), n))
        return sorted(knn, key=lambda x: (x[0], x[1]), reverse=True), num_score_fn

