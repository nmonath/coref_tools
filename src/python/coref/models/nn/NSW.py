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

from coref.models.nn.NSWNode import NSWNode
from torch.autograd import Variable

import time


class NSW(object):
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
        self.exact_nn = self.config.exact_nn
        self.dataset = dataset
        self.k = k
        self.r = r
        self.num_neighbor_edges = 0
        self.score_fn = score_fn
        self.nodes = set([NSWNode(self.k, self.score_fn, dataset[0],self)]) if len(dataset) > 0 else set()
        self.tree_leaves = set()
        self.non_leaves = set()
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
        new_node = NSWNode(self.k, self.score_fn, v,self)
        v.nsw_node = new_node
        self.nodes.add(new_node)
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
                    root.knn_and_score_not_deleted_or_offlimits(
                        v, k, offlimits, sim_cache,path)
                # print('#simcache\t%s' % len(sim_cache.keys()))
                if knn_res:
                    num_score_fn += num_score_fn_root
                    knn.update(knn_res)
            scores_and_nodes = list(knn)
            # print("lens %s\t%s\t%s" %(len(set([x[0] for x in scores_and_nodes])),len(set([x[1] for x in scores_and_nodes])),len(set([x[0] for x in scores_and_nodes]))==len(set([x[1] for x in scores_and_nodes]))))
            # print('scores {}'.format(set([x[0] for x in scores_and_nodes])))
            # print('nodes {}'.format(set([x[1] for x in scores_and_nodes])))
            # nodes_only = [x[1] for x in scores_and_nodes]
            # assert len(nodes_only) == len(set(nodes_only))
            # scores_only = [x[0] for x in scores_and_nodes]
            # assert len(scores_only) == len(set(scores_only))
            heapify(scores_and_nodes)
            # print(
            #     '#knnsearch\tnum_nodes=%s\tnum_edges=%s\tnum_explored=%s\tnum_offlimits=%s' % (
            #     len(self.nodes), self.num_edges(), num_score_fn,
            #     len(offlimits)))
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
        new_node = NSWNode(self.k, self.score_fn, v, self)
        v.nsw_node = new_node
        best = (None, None)
        self.nodes.add(new_node)
        for score, node in scores_and_nodes:
            new_node.add_link(node)
            if best[0] is None or score > best[0] or (score == best[0] and best[1] < node):
                best = (score, node)
        sorted_nodes = sorted(scores_and_nodes, key=lambda x: (x[0], x[1]), reverse=True)
        assert best == sorted_nodes[0]
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

        # knn = set()
        # num_score_fn = 0
        # for root in self.nodes:
        #     if root not in offlimits and not root.v.deleted:
        #         knn_res, num_score_fn_root = set([(root.score_fn(v,root.v),root)]),1
        #         num_score_fn += num_score_fn_root
        #         knn.update(knn_res)
        ## print('#exact_knn num_fn_calls %s' % num_score_fn)
        # knn = sorted(list(knn), key=lambda x: ( (-x[0].data.numpy()[0] if type(x[0]) is Variable else -x[0] ),-x[1].v.depth()))
        ## knn = sorted(list(knn), key=lambda x: -x[0].data.numpy()[0] if type(x[0]) is Variable else -x[0])
        # print(
        #     '#knn_and_score\tnum_nodes=%s\tnum_edges=%s\tnum_explored=%s\tnum_offlimits=%s' % (
        #     len(self.nodes), self.num_edges(), num_score_fn, len(offlimits)))
        # return [x for x in knn[:min(k, len(knn))] if not x[1].v.deleted],\
        #        num_score_fn

