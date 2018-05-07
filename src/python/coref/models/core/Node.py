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
import string

from collections import defaultdict
from queue import Queue


class Node(object):
    """A generic node with reusable functions."""

    def __init__(self):
        """Init node."""
        self.id = "id" + ''.join(random.choice(
            string.ascii_uppercase + string.digits) for _ in range(12))
        self.children = []
        self.parent = None
        self.pts = []
        self.collapsed_leaves = None
        self.is_collapsed = False
        self.deleted = False
        self.cluster_marker = False
        # Note the confusing difference between is_cluster_root and cluster_marker
        # is_cluster_root is ONLY ever set by the predicted threshold in approx_inf_hac
        self.is_cluster_root = False
        self.agglom = (0.0,0.0) # pw_score, e_score

    def __lt__(self, other):
        """An arbitrary way to determine an order when comparing 2 nodes."""
        return self.id < other.id

    def purity(self, cluster=None):
        """Compute the purity of this node.

        To compute purity, count the number of points in this node of each
        cluster label. Find the label with the most number of points and divide
        bythe total number of points under this node.

        Args:
        cluster - (optional) str, compute purity with respect to this cluster.

        Returns:
        A float representing the purity of this node.
        """
        if cluster:
            pts = [p for l in self.leaves() for p in l.pts]
            return float(len([pt for pt in pts
                              if pt[1] == cluster])) / len(pts)
        else:
            label_to_count = self.class_counts()
        return max(label_to_count.values()) / sum(label_to_count.values())

    def class_counts(self):
        """Produce a map from label to the # of descendant points with label."""
        label_to_count = defaultdict(float)
        pts = [p for l in self.leaves() for p in l.pts]
        for x in pts:
            if len(x) == 3:
                p, l, id = x
            else:
                p, l = x
            label_to_count[l] += 1.0
        return label_to_count

    def pure_class(self):
        """If this node has purity 1.0, return its label; else return None."""
        cc = self.class_counts()
        if len(cc) == 1:
            return list(cc.keys())[0]
        else:
            return None

    def siblings(self):
        """Return a list of my siblings."""
        if self.parent:
            return [child for child in self.parent.children if child != self]
        else:
            return []

    def aunts(self):
        """Return a list of all of my aunts."""
        if self.parent and self.parent.parent:
            return [child for child in self.parent.parent.children
                    if child != self.parent]
        else:
            return []

    def _ancestors(self):
        """Return all of this nodes ancestors in order to the root."""
        anc = []
        curr = self
        while curr.parent:
            anc.append(curr.parent)
            curr = curr.parent
        return anc

    def depth(self):
        """Return the number of ancestors on the root to leaf path."""
        return len(self._ancestors())

    def height(self):
        """Return the height of this node."""
        return max([l.depth() for l in self.leaves()]) - self.depth()

    def descendants(self):
        """Return all descendants of the current node."""
        d = []
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            n = queue.get()
            d.append(n)
            if n.children:
                for c in n.children:
                    queue.put(c)
        return d

    def leaves_old(self):
        """Return the list of leaves under this node."""
        lvs = []
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            n = queue.get()
            if n.children:
                for c in n.children:
                    queue.put(c)
            elif n.collapsed_leaves:
                lvs.extend(n.collapsed_leaves)
            else:
                lvs.append(n)
        return lvs

    def leaves(self):
        """Return the list of leaves under this node."""
        lvs = []
        queue = [self]
        while queue:
            n = queue.pop(0)
            if n.children:
                for c in n.children:
                    queue.append(c)
            else:
                lvs.append(n)
        return lvs

    def lca(self, other):
        """Compute the lowest common ancestor between this node and other.

        The lowest common ancestor between two nodes is the lowest node
        (furthest distances from the root) that is an ancestor of both nodes.

        Args:
        other - a node in the tree.

        Returns:
        A node in the tree that is the lowest common ancestor between self and
        other.
        """

        if self == other:
            return self

        if self.root() == self:
            print('self.root() %s other.root() %s' % (self.root().id,other.root().id))
            assert(other.root() == self)
            return self

        ancestors = [self] + self._ancestors()
        curr_node = other
        while curr_node not in set(ancestors):
            curr_node = curr_node.parent
        return curr_node

    def root(self):
        """Return the root of the tree."""
        curr_node = self
        while curr_node.parent:
            curr_node = curr_node.parent
        return curr_node

    def is_leaf(self):
        """Returns true if self is a leaf, else false."""
        return len(self.children) == 0

    def is_internal(self):
        """Returns false if self is a leaf, else true."""
        return not self.is_leaf()

    def serialize(self, filename):
        """Write out tree to filename."""
        with open(filename, 'w') as fout:
            frontier = [self]
            while frontier:
                n = frontier.pop(0)
                n_id = n.pts[0][2] if n.is_leaf() else n.id
                par_id = n.parent.id if n.parent else "None"
                gt = n.pts[0][1] if n.is_leaf() else "None"
                fout.write("%s\t%s\t%s\n" % (n_id, par_id, gt))
                for c in n.children:
                    frontier.append(c)
                # if n.children:
                    # assert (n.children[0])   # usually this is correct but
                    # assert (n.children[1])   # not for MHACNode.
                    # for c in n.children:
                    #     frontier.append(n.children[0])
                    #     frontier.append(n.children[1])
