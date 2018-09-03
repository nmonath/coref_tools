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
import logging
import string
from grinch.models.core.F1Node import F1Node
from heapq import heappush,heappushpop,heappop
from collections import defaultdict

class NSWNode(object):

    def __init__(self,e_score_fn,max_degree=None):
        super(NSWNode,self).__init__()
        self.id = "id" + ''.join(random.choice(
            string.ascii_uppercase + string.digits) for _ in range(12))
        self._neighbors = set()
        self.accepting_neighbors = True
        self.max_degree = max_degree
        self.e_score_fn = e_score_fn
        self.logger = logging.getLogger("NSWNode")
        self.logger.setLevel(logging.INFO)

    def neighbors(self):
        return self._neighbors

    def num_neighbors(self):
        return len(self._neighbors)

    def add_neighbor(self,other):
        """Use this to handle what happens when a new neighbor is added."""
        self._neighbors.add(other)
        self.logger.debug('self.add_neighbor(other)\tself=%s\tother=%s\tlen(self._neighbors)=%s' % (self.id,other.id,len(self._neighbors)))

    def add_link(self,other):
        """Add an edge between this and other."""
        self_neighbors = self.neighbors()

        if self.accepting_neighbors:
            if other not in self_neighbors:
                self.add_neighbor(other)
                if self.max_degree is not None and self.num_neighbors() == self.max_degree:
                    self.accepting_neighbors = False

        other_neighbors = other.neighbors()
        if other.accepting_neighbors:
            if self not in other_neighbors:
                other.add_neighbor(self)
                if other.max_degree is not None and other.num_neighbors() == other.max_degree:
                    other.accepting_neighbors = False

    def score_neighbors(self,query,offlimits):
        """ Compute the score between the query and all of the neighbors of this node
        
        :param query: 
        :param offlimits:
        :return: generator of (score,node)
        """
        logging.debug('[cknn] #inScoreNeighbors')
        for score,n in self.score_group(query,self.neighbors(),offlimits):
            yield score,n

    def score_group(self, query, others, offlimits):
        for n in self.neighbors():
            if n not in offlimits:
                yield self.e_score_fn(n,query),n

    def cknn(self, query, k=1, offlimits=None, sim_cache=None, path=set()):
        """Return approx k nearest neighbors of v and corresponding scores.
        
        :param query - the query node
        :param k - number of nearest neighbors to compute (default 1)
        :param offlimits - the knn among the nodes
        """

        local_path = set()
        knn_not_offlimits = []
        curr = self
        score = self.e_score_fn(curr,query)
        best = (score, curr)

        assert curr not in offlimits
        heappush(knn_not_offlimits, (score, curr))
        while True:  # We must always break out early.
            if curr in path:
                self.logger.debug('[cknn] Returning None because curr=%s was on path' % curr.id)
                return None, None
            local_path.add(curr)
            scored_neighbors = curr.score_neighbors(query,offlimits)
            for score,c in scored_neighbors:
                # self.logger.debug('[cknn] #ScoreNeighbors\tscore=%s\tc=%s' % (score, c.id))
                if best[0] is None or best[0] < score or (best[0] == score and best[1] < c):
                    self.logger.debug('[cknn] #NewBest\tscore=%s\tc=%s' % (score,c.id))
                    best = (score, c)
                if len(knn_not_offlimits) == k:
                    heappushpop(knn_not_offlimits, (score, c))
                else:
                    heappush(knn_not_offlimits, (score, c))

            while len(knn_not_offlimits) > k:
                heappop(knn_not_offlimits)
            if best[1] == curr:
                self.logger.debug('[cknn] #SearchEnd\t%s\t%s' % (best[0],best[1]))
                path.update(local_path)
                return knn_not_offlimits, 0
            else:
                curr = best[1]
                self.logger.debug('[cknn] #SearchContinues\t%s\t%s' % (best[0],best[1]))
