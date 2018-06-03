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

from heapq import heappush, heappop
from coref.models.core.Node import Node


class NSWNode(Node):
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
        self.neighbors = list()
        self.nsw = nsw

    def nsw_parent(self):
        """Returns the NSW node of the parent of this node in the tree."""
        return [self.v.parent.nsw_node] if self.v.parent else []

    def nsw_children(self):
        """Returns the NSW nodes of the children of this node in the tree."""
        return [c.nsw_node for c in self.v.children]

    def delete(self):
        for n in self.neighbors:
            n.neighbors.remove(self)
        self.neighbors = []
        self.nsw.nodes.remove(self)

    def add_link(self, other):
        """Add an edge between this and other."""
        if other not in self.neighbors:
            self.neighbors.append(other)
        if self not in other.neighbors:
            other.neighbors.append(self)

    def knn_and_score_not_deleted_or_offlimits(self, v, k=1, offlimits=None,sim_cache=None,path=set()):
        """Return approx k nearest neighbors of v and corresponding scores."""
        # print('#ksndo searching from %s for v %s k=%s len(offlimits)=%s' % (self.id,v.id,k,",".join([x.v.id for x in offlimits])))
        num_score_fn = 0

        if sim_cache is None:
            sim_cache = dict()

        def compute_score(v1,v2):
            # we always call this with the same order
            if (v1,v2) in sim_cache:
                return sim_cache[(v1,v2)],0
            else:
                score = curr.score_fn(v1, v2)
                sim_cache[(v1,v2)] = score
                return score,1

        knn_not_offlimits = []
        local_path = set()
        curr = self
        visited = set()
        score, miss = compute_score(v,curr.v)
        # print('#ksndo visit %s (%s,%s) %s %s CacheMiss=%s' % (curr.v.id, curr.v.deleted, curr in offlimits, v.id, score,miss))

        # num_score_fn += miss
        num_score_fn += 1
        best = (score, curr)
        visited.add(curr)
        assert not curr.v.deleted
        if curr not in offlimits:
            # print('#ksndo heappush %s (%s,%s) %s %s' % (curr.v.id, curr.v.deleted, curr in offlimits, v.id, score))
            heappush(knn_not_offlimits, (score, curr))
        while True:   # We must always break out early.
            if curr in path:
                return None,None
            local_path.add(curr)
            candidates = curr.neighbors #+ curr.nsw_parent() + curr.nsw_children()
            for c in candidates:
                assert not c.v.deleted
                if c not in visited and c not in offlimits:
                    visited.add(c)
                    score, miss = compute_score(v, c.v)
                    # num_score_fn += miss
                    num_score_fn += 1
                    # print('#ksndo visit %s (%s,%s) %s %s CacheMiss=%s' % (c.v.id, c.v.deleted, c in offlimits, v.id, score,miss))
                    if best[0] is None or best[0] < score or ( best[0] == score and best[1] < c):
                        # print('#ksndo new_best %s (%s,%s) %s %s prev_best %s %s' % (c.v.id, c.v.deleted, c in offlimits, v.id, score,best[1].v.id,best[0]))
                        best = (score, c)
                    # print('#ksndo heappush %s (%s,%s) %s %s' % (c.v.id, c.v.deleted, c in offlimits, v.id, score))
                    heappush(knn_not_offlimits, (score, c))
            while len(knn_not_offlimits) > k:
                heappop(knn_not_offlimits)
            if best[1] == curr:
                # print('#ksndo returning %s (%s,%s) %s %s' % (
                # best[1].v.id, best[1].v.deleted, best[1] in offlimits, best[1].id, best[0]))
                print('#searchscore end')
                path.update(local_path)
                return knn_not_offlimits, num_score_fn
            else:
                # print('#ksndo setting_curr %s (%s,%s) %s %s' % (
                # best[1].v.id, best[1].v.deleted, best[1] in offlimits, best[1].id, best[0]))
                curr = best[1]
                print('#searchscore\t%s' % best[0])
