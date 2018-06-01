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

import logging
import numpy as np

class Grinch(object):
    def __init__(self):
        self.root = None # type GNode
        self.single_elimination = False
        self.perform_graft = True
        self.logger = logging.getLogger('Grinch')
        self.logger.setLevel(logging.DEBUG)
        # todo other flags

    def node_from_nodes(self, n1,n2):
        raise NotImplementedError('Abstract method.')

    def node_from_pt(self,pt):
        raise NotImplementedError('Abstract method.')

    def e_score(self, ent1, ent2):
        """Compute the score between two ents."""
        raise NotImplementedError('Abstract method.')

    def nn_struct(self):
        raise NotImplementedError('Abstract method.')

    def k(self):
        raise NotImplementedError('Abstract method')

    def knn_and_insert(self, ent, k):
        return self.cknn_and_insert(ent, k, [])

    def knn(self, ent, k):
        return self.cknn(ent, k, [])

    def cknn_and_insert(self, ent, k, offlimits):
        return self.nn_struct().cknn_and_insert(ent, offlimits,k=k)

    def cknn(self, ent, k, offlimits):
        return self.nn_struct().cknn(ent, offlimits,k=k)

    def add_to_nn_struct(self,node):
        self.nn_struct().insert(node)

    def ent_from_pt(self, pt):
        """Returns an ent created from pt."""
        raise NotImplementedError('Abstract method.')

    def graft(self, gnode):
        """The entire grafting procedure (recursion and all)"""
        logging.debug('[graft] Grafting from %s' % gnode.id)
        curr = gnode
        offlimits = set([x for x in (
                curr.siblings() + curr.leaves() + [curr])])
        logging.debug('[graft] Offlimits size %s' % len(offlimits))
        # TODO: check sizes, check beam, single search
        while curr and curr.parent:
            self.logger.debug('[graft] Grafting curr = %s curr.parent = %s ' % (curr.id,curr.parent.id))
            performed_graft = False
            prev_curr = curr
            nn = self.cknn(curr,1,offlimits)
            if len(nn) > 0:
                nn_score, nn = nn[0]
                self.logger.debug('[graft] Found Nearest Neighbor %s with score %s' %(nn.id,nn_score))
                lca = curr.lca(nn)
                self.logger.debug('[graft] lca(curr=%s,nn=%s) = %s' % (curr.id,nn.id,lca.id))
                while curr != lca and nn != lca and curr not in nn.siblings():
                    # Graft if score is better than both of the parents scores.

                    #  - if you don't like me, then go to your parent.
                    #  - if you like me, but I don't like you, go to my parent.
                    #  - if we like each other, then graft and do another search.
                    #  - if either of us gets to our lca, then stop.
                    score_if_grafted = self.e_score(curr.ent, nn.ent)
                    curr_parent_score = curr.parent.score()
                    nn_parent_score = nn.parent.score()
                    i_like_you = score_if_grafted > curr_parent_score
                    you_like_me = score_if_grafted > nn_parent_score
                    self.logger.debug('[graft] i_like_you=%s\tyou_like_me=%s\tscore_if_grafted=%s\tcurr_parent_score=%s\tnn_parent_score=%s' % (i_like_you,you_like_me,score_if_grafted, curr_parent_score, nn_parent_score))

                    if not you_like_me:
                        nn = nn.parent
                        self.logger.debug('[graft] you_like_me=%s\tnn_is_now=%s' % (you_like_me,nn.id))
                    elif you_like_me and not i_like_you:
                        curr = curr.parent
                        self.logger.debug('[graft] you_like_me=%s\ti_like_you=%s\tcurr_is_now=%s' % (you_like_me,i_like_you,curr.id))
                    else:
                        assert you_like_me and i_like_you
                        # This can be none if nn is a child of the root
                        nn_grandparent = nn.parent.parent
                        performed_graft = True
                        parent = self.node_from_nodes(curr,nn)
                        curr.make_sibling(nn,parent)

                        # Update nn_gp -> root
                        # Update curr -> root
                        for curr_update in [nn_grandparent,curr.parent]:
                            while curr_update:
                                curr_update.update_from_children()
                                curr_update = curr_update.parent

                        if lca == self.root:
                            self.root = curr.root()
                            self.logger.debug('[graft] setting new root %s ' % self.root)
                        break
                # if you graft you should be the parent otherwise you should be the lca
                if performed_graft:
                    curr = curr.parent
                    self.logger.debug('[graft] PERFORMED GRAFT. curr=%s ' % curr.id)
                else:
                    curr = lca
                    self.logger.debug('[graft] NO GRAFT (LCA). curr=%s ' % curr.id)
            else:
                curr = curr.parent
                self.logger.debug('[graft] NO GRAFT (NO NN). curr=%s ' % curr.id)
            offlimits.update(set([x for x in (
                curr.leaves_excluding([prev_curr]))]))
            self.logger.debug('[graft] len(offlimits)=%s ' % len(offlimits))

    def _find_insert(self, gnode, sib):
        """Find the place to insert gnode.

        A gnode gets inserted next to it's nearest neighbor and then rotated. In
        this function, we effectively perform the rotations before adding gnode.

        Args:
            gnode - the GNode being insert.
            sib - the GNode sibling that was the initial target of the insert.

        Returns:
            The node gnode should be made a sibling of.
        """
        self.logger.debug('[_find_insert] gnode=%s\tsib=%s' % (gnode.id, sib.id))
        curr = sib
        score = self.e_score(gnode.ent, curr.ent)
        curr_parent_score = curr.parent.score() if curr.parent else -np.inf
        while curr.parent and score < curr_parent_score:
            self.logger.debug('[_find_insert] PERFORMED_ROTATE score=%s\tcurr.parent=%s\tcurr.parent.score=%s' % (score,curr.parent.id,curr_parent_score))
            curr = curr.parent
            curr_parent_score = curr.parent.score() if curr.parent else -np.inf
            score = self.e_score(gnode.ent, curr.ent)
        self.logger.debug('[_find_insert] RETURN score=%s\tcurr=%s\tcurr.parent.score=%s' % (
            score, curr.id, curr_parent_score))
        return curr

    def insert(self, pt):
        """Add the point or gnode to the tree.

        In Grinch, this is done by:
          1) find the nearest neighbor of the incoming point.
          2) add new point as a sibling of its nearest neighbor.
          3) perform rotations while neighbor is better with aunt.
          4) after rotations try grafting from the new point's parent.
          5) add to the NN structure the new node.
        Args:
            pt - a 3-tuple of (data, label, id)

        Returns:
            Nothing - adds pt in new GNode to the tree.
        """
        new_node = self.node_from_pt(pt) # GNode(self.ent_from_pt(pt), self.e_score)
        self.logger.debug('[insert] new_node=%s pt=(%s,%s)' % (new_node.id,pt[1],pt[2]))
        nn = self.knn_and_insert(new_node,self.k())
        if nn is None:
            assert self.root is None
            self.root = new_node
        else:
            self.logger.debug('[insert] nn=%s\tnn_score=%s' % (nn[0][1].id, nn[0][0]))
            sib = self._find_insert(new_node, nn[0][1]) # 1 best, format : (score, node)
            parent = self.node_from_nodes(sib,new_node) # create a new node from nn and sib
            self.logger.debug('[insert] new_parent_node=%s' % parent.id)
            sib.make_sibling(new_node, parent)
            self.root = self.root.root()
            curr_update = parent
            while curr_update:
                curr_update.update_from_children()
                curr_update = curr_update.parent
            self.graft(parent)
            # TODO speed up?
            self.root = self.root.root()

    def build_dendrogram(self, dataset):
        """Construct a tree on a dataset.

        Args:
            dataset - a list or generator that yields tuples: (data, label, id).

        Returns:
            Nothing -- builds tree.
        """
        for idx,d in enumerate(dataset):
            if idx % 100 == 0:
                logging.info('[build_dendrogram] Inserting pt %s' % idx)
            logging.debug('[build_dendrogram] Inserting pt %s' % idx)
            self.insert(d)

    def dendrogram_purity(self):
        raise NotImplementedError('Not implemented yet.')

    def pairwise_prec_rec_f1(self):
        raise NotImplementedError('Not implemented yet.')