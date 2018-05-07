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
from heapq import heappush, heappop
from coref.models.core.Node import Node


class NSWNode(Node):
    def __init__(self, k, score_fn, v,nsw=None):
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

    def knn(self, v, k=1):
        """Return the approximate k nearest neighbors of v."""
        knn = []
        curr = self
        while True:   # We must always break out early.
            candidates = [curr] + [n for n in curr.neighbors]
            best = (None, None)
            for c in candidates:
                score = c.score_fn(v, c.v)
                if best[0] is None or best[0] < score:
                    best = (score, c)
                heappush(knn, (score, c))
            while len(knn) > k:
                heappop(knn)
            if best[1] == curr:
                return [x.v for _, x in knn]
            else:
                curr = best[1]

    def knn_and_score(self, v, k=1, offlimits=None):
        """Return approx k nearest neighbors of v and corresponding scores."""
        knn = []
        curr = self
        num_score_fn = 0
        visited = set()
        if offlimits:
            visited.update(offlimits)
        score = curr.score_fn(v, curr.v)
        num_score_fn += 1
        best = (score, curr)

        visited.add(curr)
        heappush(knn, (score, curr))
        while True:   # We must always break out early.
            candidates = curr.neighbors
            for c in candidates:
                if c not in visited:
                    visited.add(c)
                    score = c.score_fn(v, c.v)
                    num_score_fn += 1
                    if best[0] is None or best[0] < score:
                        best = (score, c)
                    heappush(knn, (score, c))
            while len(knn) > k:
                heappop(knn)
            if best[1] == curr:
                return knn, num_score_fn
            else:
                curr = best[1]

    def knn_and_score_not_deleted(self, v, k=1, offlimits=None):
        """Return approx k nearest neighbors of v and corresponding scores."""
        knn = []
        knn_not_deleted = []

        curr = self
        num_score_fn = 0
        visited = set()
        if offlimits:
            visited.update(offlimits)
        score = curr.score_fn(v, curr.v)
        num_score_fn += 1
        best = (score, curr)

        visited.add(curr)
        heappush(knn, (score, curr))
        print('curr.v %s curr.v.deleted %s' %(curr.v,curr.v.deleted))
        if not curr.v.deleted:
            print("heap push")
            heappush(knn_not_deleted, (score, curr))
        while True:   # We must always break out early.
            candidates = curr.neighbors
            for c in candidates:
                if c not in visited:
                    visited.add(c)
                    score = c.score_fn(v, c.v)
                    num_score_fn += 1
                    print('#NSW\t%s\t%s\t%s\t%s' % (c.id, c, c.v.id, score))
                    if best[0] is None or best[0] < score:
                        print("#NSW\tNewBest=%s\tScore=%s" % (best[0],score))
                        best = (score, c)
                    if not c.v.deleted:
                        heappush(knn_not_deleted, (score, c))
                    heappush(knn, (score, c))
            while len(knn) > k:
                heappop(knn)
            while len(knn_not_deleted) > k:
                heappop(knn_not_deleted)
            if best[1] == curr:
                return knn_not_deleted, num_score_fn
            else:
                curr = best[1]

    def knn_and_score_not_deleted_or_offlimits(self, v, k=1, offlimits=None,sim_cache=None):
        """Return approx k nearest neighbors of v and corresponding scores."""
        print('#ksndo searching from %s for v %s k=%s len(offlimits)=%s' % (self.id,v.id,k,",".join([x.v.id for x in offlimits])))
        num_score_fn = 0

        if sim_cache is None:
            sim_cache = dict()

        def compute_score(v1,v2):
            # we always call this with the same
            if (v1,v2) in sim_cache:
                return sim_cache[(v1,v2)],0
            else:
                score = curr.score_fn(v1, v2)
                sim_cache[(v1,v2)] = score
                return score,1

        knn = []
        knn_not_deleted = []

        curr = self
        visited = set()
        score, miss = compute_score(v,curr.v)
        print('#ksndo visit %s (%s,%s) %s %s CacheMiss=%s' % (curr.v.id, curr.v.deleted, curr in offlimits, v.id, score,miss))
        num_score_fn += miss
        best = (score, curr)
        visited.add(curr)
        heappush(knn, (score, curr))
        if not curr.v.deleted and curr not in offlimits:
            print('#ksndo heappush %s (%s,%s) %s %s' % (curr.v.id, curr.v.deleted, curr in offlimits, v.id, score))
            heappush(knn_not_deleted, (score, curr))
        while True:   # We must always break out early.
            candidates = curr.neighbors
            for c in candidates:
                if c not in visited and not c.v.deleted and c not in offlimits:
                    visited.add(c)
                    score, miss = compute_score(v, c.v)
                    num_score_fn += miss
                    print('#ksndo visit %s (%s,%s) %s %s CacheMiss=%s' % (c.v.id, c.v.deleted, c in offlimits, v.id, score,miss))
                    if best[0] is None or best[0] < score:
                        print('#ksndo new_best %s (%s,%s) %s %s prev_best %s %s' % (c.v.id, c.v.deleted, c in offlimits, v.id, score,best[1].v.id,best[0]))
                        best = (score, c)
                    if not c.v.deleted and c not in offlimits:
                        print('#ksndo heappush %s (%s,%s) %s %s' % (c.v.id, c.v.deleted, c in offlimits, v.id, score))
                        heappush(knn_not_deleted, (score, c))
                    heappush(knn, (score, c))
            while len(knn) > k:
                heappop(knn)
            while len(knn_not_deleted) > k:
                heappop(knn_not_deleted)
            if best[1] == curr:
                print('#ksndo returning %s (%s,%s) %s %s' % (
                best[1].v.id, best[1].v.deleted, best[1] in offlimits, best[1].id, best[0]))
                return knn_not_deleted, num_score_fn
            else:
                print('#ksndo setting_curr %s (%s,%s) %s %s' % (
                best[1].v.id, best[1].v.deleted, best[1] in offlimits, best[1].id, best[0]))
                curr = best[1]