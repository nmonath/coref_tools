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

from heapq import heapify, heappush, heappop, heappushpop

from geo.models.VecNSWNode import VecNSWNode
from torch.autograd import Variable

import time
import os


class VecNSWExact(object):
    """Builds a navigable small world nearest neighbor structure."""
    def __init__(self, config, dataset, score_fn, k=5, r=5, seed=1451):
        """Build the NSW graph.

        Args:
            dataset - a list of triples (record, label, id)
            score_fn - a fun to compute the score between two data points.
            k - (int) max number of initial connections for a new node.
            r - (int) number of nodes from which to start searches.

        Returns:
            None - but sets the root to be the root of the boundary tree.
        """
        self.config = config
        ordering_and_nn = self.config.ordering
        # Load the nn map
        self.pid2node = dict()
        self.pid2nn =  dict()

        with open(ordering_and_nn) as fin:
            for line in fin:
                splt = line.strip().split('\t')
                if splt[1].lower() != "none":
                    self.pid2nn[splt[0]] = splt[1]
        self.pdict = dict()

        self.exact_nn = self.config.exact_nn
        self.dataset = dataset
        self.k = k
        self.r = r
        self.num_neighbor_edges = 0
        self.score_fn = score_fn
        self.nodes = set([VecNSWNode(self.k, self.score_fn, dataset[0], self)]) if len(dataset) > 0 else set()
        self.approx_max_degree = 0
        self.random = random.Random(seed)
        for d in dataset[1:]:
            new_node = self.insert(d)

    def insert(self, v):
        """Insert a new node into the graph.

        Args:
            v - (obj) the object we're trying to insert.

        Returns:
            Return the new node that was inserted.
        """
        assert len(self.nodes) == 0
        print('Inserting the point %s into the nsw' % v.id)
        new_node = VecNSWNode(self.k, self.score_fn, v, self)
        v.nsw_node = new_node
        self.nodes.add(new_node)
        this_pid= v.pts[0][2]
        self.pid2node[this_pid] = new_node
        print('This pid %s' % (this_pid))
        num_score_fn = 0
        return num_score_fn

    def num_edges(self):
        return self.num_neighbor_edges + len(self.nodes)-1

    def __del__(self):
        """Delete all nodes from the NSW."""
        self.nodes = None

    def _knn(self, v, offlimits, k=1, r=1):
        """Returns the approximate nearest neighbor of v.

        Args:
            v - the query object.
            offlimits - a set of nodes that cannot be returned.
            k - (int) number of neighbors to retrieve.
            r - (int) the number of nodes to start the search from.

        Returns:
            A min heap of (score, node).
        """
        allowable = self.nodes.difference(offlimits)
        if len(allowable) == 0 or k * r * np.log(len(allowable)) > len(
                allowable) or self.exact_nn:
            scores_and_nodes, num_score_fn = self.exact_knn(v, offlimits, k)
        else:
            knn = set()
            num_score_fn = 0
            sim_cache = dict()
            if len(allowable) < r:
                roots = allowable
            else:
                roots = self.random.sample(allowable, r)
            path = set()
            for root in roots:
                knn_res, num_score_fn_root = \
                    root.knn_and_score_vec(v, k, offlimits, sim_cache,path)
                # print('#simcache\t%s' % len(sim_cache.keys()))
                if knn_res:
                    num_score_fn += num_score_fn_root
                    knn.update(knn_res)
            scores_and_nodes = list(knn)
            heapify(scores_and_nodes)
        for i in range(max(0, len(scores_and_nodes) - k)):
            heappop(scores_and_nodes)
        assert k >= len(scores_and_nodes)
        # print('#sizeof scores_and_nodes\t%s' % len(scores_and_nodes))
        return scores_and_nodes, num_score_fn

    def knn(self, v, offlimits, k=1, r=1):
        """Returns the approximate nearest neighbor of v.

        Args:
            v - the query object.
            offlimits - a set of nodes that cannot be returned.
            k - (int) number of neighbors to retrieve.
            r - (int) the number of nodes to start the search from.

        Returns:
            A sorted (descending) list of (score, node).
        """
        scores_and_nodes, num_score_fn = self._knn(v, offlimits, k, r)
        return sorted(scores_and_nodes, key=lambda x: (x[0],x[1]),reverse=True), num_score_fn

    def knn_and_insert(self, v, offlimits, k=1, r=1):
        """Adds v to the NSW and returns its approximate 1-nearest neighbor.

        Args:
            v - the query object.
            offlimits - a set of nodes that cannot be returned.
            k - (int) number of neighbors to retrieve.
            r - (int) the number of nodes to start the search from.

        Returns:
            A list of no more than k nodes and their scores.
        """
        scores_and_nodes, num_score_fn = self._knn(v, offlimits, k, r)
        new_node = VecNSWNode(self.k, self.score_fn, v, self)
        v.nsw_node = new_node
        this_pid = v.pts[0][2]
        self.pid2node[this_pid] = new_node
        this_guys_nn = self.pid2nn[this_pid]
        print('this_pid\t%sthis_guys_nn\t%s' % (this_pid,this_guys_nn))
        best = self.pid2node[this_guys_nn]
        best_score = best.score_fn(v,new_node.v)
        print('best_score\t%s' % best_score)
        self.nodes.add(new_node)
        new_node.add_link(best)
        scores_and_nodes_to_return = []
        for score, node in scores_and_nodes:
            if node != best:
                new_node.add_link(node)
                scores_and_nodes_to_return.append((scores_and_nodes,node))
        sorted_nodes = [(best_score,best)] + sorted(scores_and_nodes_to_return, key=lambda x: (x[0], x[1]), reverse=True)
        return sorted_nodes, num_score_fn

    def exact_knn(self, v, offlimits,  k=1):
        """Returns the exact knn of v.

        Args:
            v - the query object.
            offlimits - a set of nodes that cannot be returned.
            k - (int) number of neighbors to retrieve.

        Returns:
            A minheap of no more than k (score,node).
        """
        knn = []
        num_score_fn = 0
        for n in self.nodes:
            if n not in offlimits:
                num_score_fn += 1
                if len(knn) == k:
                    heappushpop(knn, (n.score_fn(v, n.v), n))
                else:
                    heappush(knn, (n.score_fn(v, n.v), n))
        return sorted(knn, key=lambda x: (x[0],x[1]),reverse=True), num_score_fn
