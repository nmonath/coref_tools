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

import random

from coref.models.nn.NSWNode import NSWNode
from torch.autograd import Variable

class NSW(object):
    """Builds a navigable small world nearest neighbor structure."""
    def __init__(self, dataset, score_fn, k=5, r=5, seed=1451):
        """Build the NSW graph.

        Args:
            dataset - a list of triples (record, label, id)
            score_fn - a fun to compute the score between two data points.
            k - (int) max number of initial connections for a new node.
            r - (int) number of nodes from which to start searches.

        Returns:
            None - but sets the root to be the root of the boundary tree.
        """
        self.dataset = dataset
        self.k = k
        self.r = r
        self.num_edges = 0
        self.score_fn = score_fn
        self.nodes = set([NSWNode(self.k, self.score_fn, dataset[0],self)]) if len(dataset) > 0 else set()
        self.tree_leaves = set()
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
        new_node = NSWNode(self.k, self.score_fn, v,self)
        v.nsw_node = new_node
        if len(self.nodes) == 0:
            self.nodes.add(new_node)
            if v.point_counter <= 1:
                print("Adding to tree leaves")
                self.tree_leaves.add(new_node)
        else:
            knn = self.knn(v, self.k, self.r)
            for n in knn:
                new_node.add_link(n)
                self.num_edges += 1
            self.nodes.add(new_node)
            if v.point_counter <= 1:
                print("Adding to tree leaves")
                self.tree_leaves.add(new_node)
        self.add_tree_edges(v)

    def __del__(self):
        """Delete all nodes from the NSW."""
        self.nodes = None

    def knn(self, v, k=1, r=1):
        """Returns the approximate nearest neighbor of v.

        Args:
            v - the query object.
            k - (int) number of neighbors to retrieve.
            r - (int) the number of nodes to start the search from.

        Returns:
            A list of no more than k nodes.
        """
        return [x[1] for x in self.knn_and_score_offlimits(v, [], k, r)[0]]

    def knn_and_score(self, v, k=1, r=1):
        """Returns the approximate nearest neighbor of v.

        Args:
            v - the query object.
            k - (int) number of neighbors to retrieve.
            r - (int) the number of nodes to start the search from.

        Returns:
            A list of no more than k nodes and their scores.
        """
        knn = set()
        num_score_fn = 0
        if len(self.tree_leaves) < r:
            roots = self.tree_leaves
        else:
            roots = self.random.sample(self.tree_leaves, r)
        for root in roots:
            knn_res, num_score_fn_root = root.knn_and_score_not_deleted(v, k)
            num_score_fn += num_score_fn_root
            knn.update(knn_res)
        knn = sorted(list(knn), key=lambda x: ( (-x[0].data.numpy()[0] if type(x[0]) is Variable else -x[0] ),-x[1].v.depth()))
        print('#knn_and_score num_fn_calls %s' % num_score_fn)
        return [x for x in knn[:min(k, len(knn))] if not x[1].v.deleted]

    def knn_and_score_filter(self, v, fil,  k=1, r=1):
        """Returns the approximate nearest neighbor of v.

        Args:
            v - the query object.
            fil- (func) node -> boolean, only return things matching the filter.
            k - (int) number of neighbors to retrieve.
            r - (int) the number of nodes to start the search from.

        Returns:
            A list of no more than k nodes and their scores.
        """
        knn = set()
        num_score_fn = 0
        if len(self.nodes) < r:
            roots = self.nodes
        else:
            roots = self.random.sample(self.tree_leaves, r)
        for root in roots:
            knn_res, num_score_fn_root = root.knn_and_score_not_deleted_or_offlimits(v, k,offlimits=[])
            num_score_fn += num_score_fn_root
            knn.update(knn_res)
        knn = sorted(list(knn), key=lambda x: ( (-x[0].data.numpy()[0] if type(x[0]) is Variable else -x[0] ),-x[1].v.depth()))
        # knn = sorted(list(knn), key=lambda x: -x[0].data.numpy()[0] if type(x[0]) is Variable else -x[0])
        print('#knn_and_score num_fn_calls %s' % num_score_fn)
        return [x for x in knn[:min(k, len(knn))] if fil(x[1].v)]

    def knn_and_score_offlimits(self, v, offlimits,  k=1, r=1):
        """Returns the approximate nearest neighbor of v.

        Args:
            v - the query object.
            fil- (func) node -> boolean, only return things matching the filter.
            k - (int) number of neighbors to retrieve.
            r - (int) the number of nodes to start the search from.

        Returns:
            A list of no more than k nodes and their scores.
        """
        if k*r > len(self.nodes):
            return self.exact_knn(v,offlimits,k,r)
        else:
            knn = set()
            num_score_fn = 0
            allowable = self.nodes.difference(offlimits)
            sim_cache = dict()
            if len(allowable) < r:
                roots = allowable
            else:
                roots = self.random.sample(allowable, r)
            print('#kso v=%s len(offlimits)=%s len(allowable)=%s len(roots)=%s k=%s r=%s' % (v.id, len(offlimits), len(allowable), len(roots), k, r))
            for root in roots:
                knn_res, num_score_fn_root = root.knn_and_score_not_deleted_or_offlimits(v, k,offlimits,sim_cache)
                num_score_fn += num_score_fn_root
                knn.update(knn_res)
            knn = sorted(list(knn), key=lambda x: ( (-x[0].data.numpy()[0] if type(x[0]) is Variable else -x[0] ),-x[1].v.depth()))
            # knn = sorted(list(knn), key=lambda x: -x[0].data.numpy()[0] if type(x[0]) is Variable else -x[0])
            print('#knn_and_score num_fn_calls %s size of offlimits %s' % (num_score_fn,len(offlimits)))
            return [x for x in knn[:min(k, len(knn))] if not x[1].v.deleted],num_score_fn

    def exact_knn(self, v, offlimits,  k=1, r=1):
        """Returns the approximate nearest neighbor of v.

        Args:
            v - the query object.
            fil- (func) node -> boolean, only return things matching the filter.
            k - (int) number of neighbors to retrieve.
            r - (int) the number of nodes to start the search from.

        Returns:
            A list of no more than k nodes and their scores.
        """
        knn = set()
        num_score_fn = 0
        for root in self.nodes:
            if root not in offlimits and not root.v.deleted:
                knn_res, num_score_fn_root = set([(root.score_fn(v,root.v),root)]),1
                num_score_fn += num_score_fn_root
                knn.update(knn_res)
        print('#exact_knn num_fn_calls %s' % num_score_fn)
        knn = sorted(list(knn), key=lambda x: ( (-x[0].data.numpy()[0] if type(x[0]) is Variable else -x[0] ),-x[1].v.depth()))
        # knn = sorted(list(knn), key=lambda x: -x[0].data.numpy()[0] if type(x[0]) is Variable else -x[0])
        return [x for x in knn[:min(k, len(knn))] if not x[1].v.deleted],num_score_fn

    def add_tree_edges(self,erch_node):
        """Add the edges connecting erch_node to its parents/children 
        """
        # parent
        if erch_node.parent:
            erch_node.parent.nsw_node.add_link(erch_node.nsw_node)
        # children
        if erch_node.children:
            erch_node.children[0].nsw_node.add_link(erch_node.nsw_node)
            erch_node.children[1].nsw_node.add_link(erch_node.nsw_node)