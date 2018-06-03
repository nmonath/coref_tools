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

import os
import numpy as np
import time
import sys

from coref.models.nn import new_nn_structure
from coref.util.Graphviz import Graphviz
from coref.util.GraphvizNSW import GraphvizNSW

from geo.models.GNode import GNode


class GraftMetaDataRecorder(object):
    def __init__(self):
        self.records = []

    def count(self,field_name):
        c = 0
        for r in self.records:
            if getattr(r,field_name) or  getattr(r,field_name)  == 1 :
                c += 1
        return c

    def sum(self,field_name):
        c = 0.0
        for r in self.records:
            if type(getattr(r,field_name)) == int:
                c += getattr(r,field_name)
            elif type(getattr(r,field_name)) == bool:
                if getattr(r,field_name):
                    c += 1
        return c

    def report(self):
        num_accepted = 0
        num_allowed = 0.0
        none_available = 0.0
        siblings = 0
        desc = 0
        anc = 0
        graft_from_root = 0
        for r in self.records:
            print(r.tsv())
            if r.accepted:
                num_accepted += 1
            if r.allowed:
                num_allowed += 1
            if r.none_available:
                none_available += 1
            if r.is_descendant:
                desc += 1
            if r.is_ancestor:
                anc += 1
            if r.are_siblings:
                siblings += 1
            if r.graft_from_root:
                graft_from_root += 1

        print('#GRAFT NUM RECORDS\t%s' % len(self.records))
        print('#GRAFT NUM ACCEPTED\t%s' % num_accepted)
        print('#GRAFT NUM ALLOWED\t%s' % num_allowed)
        print('#GRAFT NUM NOT AVAILABLE\t%s' % none_available)
        print('#GRAFT NUM SIBLINGS\t%s' % siblings)
        print('#GRAFT NUM DESCENDANTS\t%s' % desc)
        print('#GRAFT NUM ANCESTORS\t%s' % anc)
        print('#GRAFT NUM FROM ROOT\t%s' % graft_from_root)


class GraftMetaData(object):
    def __init__(self, grinch,graft_from,graft_to,accepted,allowed,none_available):
        self.grinch = grinch
        self.graft_from = graft_from
        self.graft_to = graft_to
        self.none_available = none_available
        self.graft_from_is_leaf = len(graft_from.children) == 0 if graft_from else None
        # add the number of leaves()
        # add the children's score in the graft.
        self.graft_to_num_leaves = len(graft_to.leaves())
        self.graft_from_num_leaves = len(graft_from.leaves()) if graft_from else -1
        self.graft_to_is_pure = graft_to.purity() == 1.0

        if self.none_available:
            self.classes_involved = set([x.pts[0][1] for x in graft_to.leaves()])
            self.num_classes_involved = len(self.classes_involved)
            self.pure = self.num_classes_involved == 1
            self.accepted = accepted
            self.allowed = allowed
            self.are_siblings = False
            # from is desc of to
            self.is_descendant = False
            # from is ancestor of to
            self.is_ancestor = False
            self.graft_from_root = False
        else:
            self.classes_involved = set([x.pts[0][1] for x in graft_from.leaves()]).union(set([x.pts[0][1] for x in graft_to.leaves()]))
            self.num_classes_involved = len(self.classes_involved)
            self.pure = self.num_classes_involved == 1
            self.accepted = accepted
            self.allowed = allowed
            self.are_siblings = self.graft_from.siblings()[0] == self.graft_to
            self.lca = self.graft_from.lca(self.graft_to)
            # from is desc of to
            self.is_descendant = self.lca == self.graft_to
            # from is ancestor of to
            self.is_ancestor = self.lca == self.graft_from
            self.graft_from_root = self.graft_from.parent == None

    def tsv(self):
        return "#graft\tfrom=%s\tto=%s\tclasses=%s\tpure=%s\tis_leaf=%s\taccept=%s\tallowed=%s\tnot_avail=%s\tsibs=%s\tdesc=%s\tanc=%s\troot=%s" % (
        (self.graft_from.id if self.graft_from else "None"), self.graft_to.id, self.num_classes_involved, self.pure,
        self.graft_from_is_leaf, self.accepted, self.allowed, self.none_available, self.are_siblings,
        self.is_descendant, self.is_ancestor, self.graft_from_root)


class Grinch(object):
    def __init__(self, config, dataset, model,perform_rotation=True,perform_graft=True):
        self.config = config
        self.dataset = dataset
        self.model = model
        self.root = None  # GNode
        self.perform_rotation = perform_rotation
        self.perform_graft = perform_graft
        self.nn_structure = new_nn_structure(self.config.nn_structure,
                                             self.config,
                                             self.score_function_np)
        self.graft_recorder = GraftMetaDataRecorder()
        self.pair_to_pw = {}
        self.observed_classes =set()
        self.nn_k = self.config.nn_k if not self.config.exact_nn else np.inf
        self.nsw_r = self.config.nsw_r
        self.num_computations = 0
        self.my_score_f = lambda x: x.children[0].e_model.e_score(
            x.children[1].e_model)
        self.beam = config.beam
        self.num_e_scores = 0
        print('#using beam = %s' % self.beam)
        self.max_node_graft_size = self.config.max_node_graft_size

    def hallucinate_merge(self,n1, n2, pw_score=None, debug_pw_score=None):
        """Return the merge of n1 and n2.

        Args:
            n1 - a GNode.
            n2 - a GNode.

        Returns:
            The merger of n1.e_model and n2.e_model.
        """
        return n1.e_model.hallucinate_merge(n2.e_model)

    def score_function_np(self, n1, n2):
        """Compute the score between two GNodes."""
        # e_score = self.model.e_score(self.hallucinate_merge(n1, n2))
        e_score = self.model.quick_e_score(n1.e_model, n2.e_model)
        return e_score

    def find_insert(self, leaf_node, new_node, pw_score=None, debug=False):
        """Find the the node to split down from.

        Starting from leaf_node compute score of merging new_node with leaf
        recursively up the tree (with its parents) until you find a locally
        optimal place to add yourself. This essentially simulates rotations.

        Args:
            leaf_node - a node.
            new_node - the node being inserted.

        Returns:
            The node to merge with, the resulting entity model, the score of
            the entity model, start time and end time.
        """
        # print('find_insert(%s,%s,%s) ' % (leaf_node.id, new_node.id, pw_score))
        curr = leaf_node
        # ap = self.hallucinate_merge(curr, new_node)
        # new_score = ap.my_e_score()
        new_score = self.score_function_np(curr, new_node)

        # rotation means you need to increase the number of e_scores
        self.num_computations += 1

        time_before_rotation = time.time()
        # print('\tcurr %s' % curr.id)
        # print('\tcurr.parent %s' % curr.parent.id if curr.parent else "None")
        # print('\tnew_score %s' % new_score)
        if self.perform_rotation:
            while curr.parent is not None:
                if self.config.max_node_rotate_size is not None and \
                             curr.parent.point_counter > self.config.max_node_rotate_size:
                    if self.config.debug:
                        print(
                            '#FIND_INSERT-STOP_BY_SIZE\tnew_score=%s\tcurr.parent.score=%s\tcurr_id=%s\tparent_id=%s\tcurr_num_pts=%s\tparent_num_pts=%s' % (
                            new_score, curr.parent.lazy_my_score(), curr.id, curr.parent.id, curr.point_counter, curr.parent.point_counter))
                    break
                # assert curr.parent.my_score == curr.parent.e_model.my_e_score()
                if new_score > curr.parent.lazy_my_score():
                    if debug:
                        print('#FIND_INSERT-RETURN\tnew_score=%s\tcurr.parent.score=%s\tcurr_id=%s\tparent_id=%s\tcurr_num_pts=%s\tparent_num_pts=%s' % (
                            new_score, curr.parent.lazy_my_score(), curr.id, curr.parent.id, curr.point_counter, curr.parent.point_counter))
                    break
                else:
                    if debug:
                        print('#FIND_INSERT-ROTATE\tnew_score=%s\tcurr.parent.score=%s\tcurr_id=%s\tparent_id=%s\tcurr_num_pts=%s\tparent_num_pts=%s' % (
                            new_score, curr.parent.lazy_my_score(), curr.id, curr.parent.id, curr.point_counter, curr.parent.point_counter))
                    curr = curr.parent
                    # ap = self.hallucinate_merge(curr, new_node, None)
                    # new_score = ap.my_e_score()
                    new_score = self.score_function_np(curr, new_node)
                    self.num_computations += 1
                    # print('\tcurr %s' % curr.id)
                    # print('\tcurr.parent %s' % (curr.parent.id if curr.parent else "None"))
                    # print('\tnew_score %s' % new_score)

        time_after_rotation = time.time()
        ap = self.hallucinate_merge(curr, new_node, None)
        return curr, ap, new_score, time_before_rotation,time_after_rotation

    def _try_graft(self, curr, offlimits, gnode, p_idx, graft_index, start_time):
        """Try to find a graft for curr.

        Look through the NSW leaves FIRST for the closest non-offlimits node
        for curr. Compute the score and check for a merge. If the score is
        better than curr.parent.my_score and the other.parent.my_score,
        perform the merge and update. If the merge score is better than
        curr.parent.my_score but not others parent score, then try to merge with
        other's parent. If the merge score is worse than curr's parent score,
        return nothing. This function also does a bunch of logging.

        Args:
            curr - the node to initiate grafting from.
            offlimits - the nodes in the NSW that cannot be grafted.
            gnode - newly created node with new point.
            p_idx - the point index (int)
            graft_index - number of times grafted so far
            start_time - time we started insert

        Returns:
            Nothing
        """
        if self.config.debug:
            print('#tryGraft trying to graft \t%s' % (curr.id))
        # First do a search for the closest leaf in the NSW.
        knn_and_score, num_searched_approx = \
            self.nn_structure.knn(
                curr, offlimits, k=self.nn_k, r=self.nsw_r)
        # knn_and_score, num_searched_approx = \
        #     self.nn_structure.knn_and_score_offlimits(
        #         curr, offlimits, k=self.nn_k, r=self.nsw_r)

        self.num_computations += num_searched_approx

        # If there aren't enough nodes to explore just do nothing.
        if knn_and_score:
            other, other_score = knn_and_score[0][1].v, knn_and_score[0][0]
            if self.config.debug:
                print('#tryGraft found nn\t%s\t%s' % (curr.id, other.id))
        else:
            return

        our_lca = curr.lca(other)

        while curr != our_lca and other != our_lca and curr not in other.siblings() and (self.max_node_graft_size is None or other.point_counter < self.max_node_graft_size):
            if self.config.debug:
                print('#tryGraft trying new parent\t%s\t%s' % (curr.id, other.id))
                sys.stdout.flush()
            # Trying to speed up grafting:
            #  - if you don't like me, then go to your parent
            #  - if you like me, but I don't like you, go to my parent
            #  - if we both like each other, then graft and do another search.
            #  - if either of us gets to our lca, then stop, we shouldn't graft

            # Check if graft score is better than both of the parents scores.
            other_score = self.model.quick_e_score(curr.e_model,
                                                   other.e_model)
            i_like_you = other_score > curr.parent.lazy_my_score()
            you_like_me = other_score > other.parent.lazy_my_score()

            if self.config.debug:
                print('#i_like_you and you_like me\t%s\t%s\t%s\t%s\t%s' % (i_like_you,
                                                           you_like_me,other_score,curr.parent.lazy_my_score(),other.parent.lazy_my_score()))

            if not you_like_me:
                other = other.parent
            elif you_like_me and not i_like_you:
                curr = curr.parent
            else:
                assert you_like_me and i_like_you
                print('#doingGraft')
                # We're going to graft.
                # [LOGGING] Write the tree before the graft
                if self.config.write_every_tree:
                    Graphviz.write_tree(
                        os.path.join(
                            self.config.canopy_out,
                            'tree_%s_before_graft_%s.gv' % (
                                p_idx, graft_index)),
                        self.root, [other.id, curr.id], [gnode.id])

                # Do the graft.
                assert other.parent
                prev_gp = other.parent.parent
                # new_ap_graft = self.hallucinate_merge(curr, other, None)
                new_graft_internal = curr.graft_to_me(
                    other,
                    new_aproj=None,
                    new_my_score=None)  # We don't want a pw guy here.

                # Update from new_graft_internal to the root.
                before_update_time = time.time()
                curr_update = new_graft_internal
                while curr_update:
                    curr_update.update_from_children()
                    curr_update = curr_update.parent
                after_update_time = time.time()

                print('#TimeForUpdateInGraft\t%s\t%s' % (
                    after_update_time - before_update_time,
                    after_update_time - start_time))

                # Update from previous parent to root.
                if prev_gp:
                    before_update_time = time.time()
                    curr_update = prev_gp
                    while curr_update:
                        curr_update.update_from_children()
                        curr_update = curr_update.parent
                    after_update_time = time.time()
                    print('#TimeForUpdateInPrevGPGraft\t%s\t%s' % (
                        after_update_time - before_update_time,
                        after_update_time - start_time))

                # Add new graft internal to the nn-struct.
                # self.nn_structure.insert(new_graft_internal)
                # TODO AK: doe we need this?
                self.root = new_graft_internal.root()

                # Write some trees.
                if self.config.write_every_tree:
                    Graphviz.write_tree(
                        os.path.join(self.config.canopy_out,
                                     'tree_%s_post_graft_%s.gv' % (
                                         p_idx, graft_index)),
                        self.root,
                        [other.id, curr.id],
                        [gnode.id])

                # Update offlimits.
                # offlimits.update({other})
                # offlimits.update(other.descendants())
                # return offlimits
                return new_graft_internal
        return None   # No graft found.

    def insert(self, p, p_idx):
        """Incrementally add p to the tree.

        Based on my parameters, either apply rotations and/or grafting. Steps:
        1) Find the closest node to p in the tree (optionally rotate).
        2) Add p to the nn-structure.
        3) Add p to the tree.
        4) Update the nodes on the path from the new internal node to the root.
        5) Add new internal node to the nn-structure.
        6) Try grafting:
        6.1)   Construct offlimits.
        6.2)   Find nearest non-offlimits leaf.
        6.3)   Compute scores to check if graft should be done. If yes:
        6.3.1)     Do the graft.
        6.3.2)     update nodes on path from new internal to root
        6.3.3)     update nodes on path from previous parent to root
        6.3.4)     Add new graft parent to nn-structure.
        6.4)   If No because leaf likes where it is:
        6.4.1)     Try to graft its parent.
        6.5)   Otherwise do not graft.


        Args:
            p - (np.array, str, str)
            p_idx - int index
        """

        # TODO (AK): next line should be cleaner.
        print(self.my_score_f)
        gnode = GNode([p], self.model.new(p[0],ment_id=p[2]), my_score_f=self.my_score_f,grinch=self)
        start_time = time.time()
        print('Inserting p (%s,%s) into tree ' % (p[1], p[2]))
        graft_index = 0
        if self.root is None:
            self.root = gnode
            self.nn_structure.insert(gnode)
        else:
            # If add_to_mention is True, then internal nodes are offlimits.
            if self.config.add_to_mention:
                offlimits = set(
                    [d.nsw_node for d in self.root.descendants() if
                     d.point_counter > 1 if d.nsw_node])
            else:
                offlimits = set()

            # Find the k-nn to gnode.
            knn_and_score, num_searched_approx = \
                self.nn_structure.knn_and_insert(
                    gnode, offlimits, k=self.nn_k, r=self.nsw_r)
            self.num_computations += num_searched_approx

            print('#num computations local and total\t%s\t%s' % (
                num_searched_approx, self.num_computations))

            approx_closest, approx_closest_score = \
            knn_and_score[0][1].v, knn_and_score[0][0]

            if self.config.debug:
                print('#approx_closest\t%s\t%s' % (
                    p[1], [x[1] for x in approx_closest.pts]))

            # 1) Find where to be added / rotate
            insert_node, new_ap, new_score, time_before_rotation, \
            time_finish_placement = self.find_insert(
                approx_closest, gnode, debug=self.config.debug)

            # 2) Add gnode to the nn structure.
            # self.nn_structure.insert(gnode)

            # 3) Add gnode to the tree.
            new_internal_node = insert_node.split_down(gnode, new_ap, new_score)

            # assert gnode.root() == insert_node.root(), "p_ment.root() %s == insert_node.root() %s" % (
            #     gnode.root(), insert_node.root())
            # assert gnode.lca(
            #     insert_node) == new_internal_node, "p_ment.lca(insert_node) %s == new_internal_node %s" % (
            #     gnode.lca(insert_node), new_internal_node)

            # 4) Update on the path from new_internal node to the root.
            if new_internal_node.parent:
                # Comment out the following check if e_score is random.
                # assert new_internal_node.my_score == \
                #        new_internal_node.e_model.my_e_score()
                new_internal_node.parent.update_aps(gnode.e_model, None)

            # 5) Add new internal node to the nn structure.
            # !! NO LONGER ADDING INTERNL NODES TO NSW !!

            # self.nn_structure.insert(new_internal_node)
            self.root = self.root.root()

            # 6) Try Grafting.
            if self.perform_graft and (self.max_node_graft_size is None or new_internal_node.point_counter < self.max_node_graft_size):
                time_before_graft = time.time()
                curr = new_internal_node
                # 6.1) Find offlimits
                # offlimits = set(
                #     [x.nsw_node for x in (
                #             curr.siblings() + curr.descendants() +
                #             curr._ancestors() + [curr] + [self.root])]
                # )
                offlimits = set([x.nsw_node for x in (
                        curr.siblings() + curr.leaves() + [curr])]
                )

                graft_attempt = 0
                while curr and curr.parent and (self.max_node_graft_size is None or curr.point_counter < self.max_node_graft_size):
                    time_before_this_graft = time.time()
                    prev_curr = curr
                    # Try to graft and update offlimits when successful.
                    if (self.config.fast_graft and graft_attempt == 0) or self.config.fast_grafts_only:
                        curr_new, knn_and_score = self._try_graft_fast(curr, knn_and_score, offlimits, gnode, p_idx,
                                               graft_index, start_time)
                    elif not self.config.fast_grafts_only:
                        curr_new = self._try_graft(curr, offlimits, gnode, p_idx,
                                                   graft_index, start_time)

                    # sib = curr.siblings()[0]
                    # curr = curr.parent

                    # Only take the time to update offlimits if necessary.
                    # NM: If curr is not none, then the only new things to add to
                    # offlimits are the leaves of the node which was grafted to be
                    # the sibling of "curr"
                    if curr_new and curr_new.parent:
                        if self.config.debug:
                            print('#successfulGraftAttempt\t%s' % graft_attempt)
                        graft_index += 1
                        graft_attempt += 1
                        offlimits.update(set([x.nsw_node for x in (
                            curr_new.leaves_excluding([prev_curr]))]))

                    elif graft_attempt < self.beam:
                        graft_attempt += 1
                        if curr.parent:
                            offlimits.update(set([x.nsw_node for x in (
                                curr.parent.leaves_excluding([curr]))]))
                        curr = curr.parent
                    else:
                        graft_attempt += 1
                        curr = curr_new
                        # either you didn't graft or this is the root of the tree?
                        # assert curr is None or curr.parent is None

                    time_after_this_graft = time.time()
                    print("#TimeAfterThisGraftProposal\t%s\t%s" % (
                        time_after_this_graft - time_before_this_graft,
                        time_after_this_graft - start_time))

                end_time = time.time()
                print("#TimeAfterAllGrafts\t%s\t%s" % (
                    end_time - time_before_graft,
                    end_time - start_time))
        end_time = time.time()
        print('Done Inserting p (%s,%s) into tree in %s seconds  ' % (
            p[1], p[2], end_time - start_time))
        print('#numgrafts\t%s' % graft_index)

        # Clean up and log.
        self.observed_classes.add(p[1])
        sys.stdout.flush()
        if self.config.write_every_tree:
            if len(self.config.canopy_out) > 0:
                Graphviz.write_tree(os.path.join(self.config.canopy_out,
                                                 'tree_%s.gv' % p_idx),
                                    self.root, [], [gnode.id])
                if self.config.nn_structure == 'nsw':
                    GraphvizNSW.write_nsw(
                        os.path.join(self.config.canopy_out,
                                     'nsw_%s.gv' % p_idx),
                        self.nn_structure)

    def pure_subtree_with(self, new_pt, nearest):
        for l in nearest.leaves():
            if new_pt.pts[0][1] != l.pts[0][1]:
                return False
        return True

    def build_dendrogram(self):
        idx = 0
        start_time = time.time()
        for p in self.dataset:
            print(
                '[build dendrogram] Inserting pt number %s into tree' % idx)
            self.insert(p, idx)
            print('Time so far %s' % (time.time()-start_time))
            idx += 1 #                                   2            3                    4             5          6            7
            print("#NUMCOMP\t%s\t%s\t%s\t%s\t%s\t%s" % (idx,self.num_computations,self.num_e_scores,np.log(idx),np.log2(idx),np.sqrt(idx)) )
        self.graft_recorder.report()
        return self.root.root()

    def _try_graft_fast(self, curr, knn_and_score, offlimits, gnode, p_idx, graft_index, start_time):
        """Try to find a graft for curr, faster than the speed of sound, faster than we thought we'd go

        Look through the knn_and_score for the closest leaf in the tree that is NOT OFFLIMITS.
        If there is such a leaf, try to graft it as before. 
        If you graft then try grafting from your parent.
        
        Grafts follow the same logic as before:  Compute the score and check for a merge. 
        If the score is
        better than curr.parent.my_score and the other.parent.my_score,
        perform the merge and update. If the merge score is better than
        curr.parent.my_score but not others parent score, then try to merge with
        other's parent. If the merge score is worse than curr's parent score,
        return nothing. This function also does a bunch of logging.

        Args:
            curr - the node to initiate grafting from.
            knn_and_score - the results from the nn search that added curr to the NSW. 
            offlimits - the nodes in the NSW that cannot be grafted.
            gnode - newly created node with new point.
            p_idx - the point index (int)
            graft_index - number of times grafted so far
            start_time - time we started insert

        Returns:
            Nothing
        """
        if self.config.debug:
            print('#tryGraftFast trying to graft \t%s' % (curr.id))


        # First do a search for the closest leaf in the NSW.
        knn_and_score_valid = []
        for score,node in knn_and_score:
            if node not in offlimits:
                knn_and_score_valid.append((score,node))

        if self.config.debug:
            print('#tryGraftFast number of nns %s, num valid \t%s' % (len(knn_and_score),len(knn_and_score_valid)))

        # If there aren't enough nodes to explore just do nothing.
        if knn_and_score_valid:
            other, other_score = knn_and_score_valid[0][1].v, knn_and_score_valid[0][0]
            if self.config.debug:
                print('#tryGraftFast found nn\t%s\t%s' % (curr.id, other.id))
        else:
            if self.config.debug:
                print('#tryGraftFast no nn found for \t%s' % (curr.id))
            return None, knn_and_score_valid

        our_lca = curr.lca(other)

        while curr != our_lca and other != our_lca and curr not in other.siblings():

            if self.max_node_graft_size is not None:
                if other.point_counter > self.max_node_graft_size or curr.point_counter > self.max_node_graft_size:
                    if self.config.debug:
                        print('#tryGraftFast Breaking because of sizes\t%s\t%s\t%s\t%s' % (curr.id, other.id,
                                                                                   curr.point_counter,
                                                                                   other.point_counter))
                    break


            if self.config.debug:
                print('#tryGraftFast trying new parent\t%s\t%s' % (curr.id, other.id))
                sys.stdout.flush()
            # Trying to speed up grafting:
            #  - if you don't like me, then go to your parent
            #  - if you like me, but I don't like you, go to my parent
            #  - if we both like each other, then graft and do another search.
            #  - if either of us gets to our lca, then stop, we shouldn't graft

            # Check if graft score is better than both of the parents scores.
            other_score = self.model.quick_e_score(curr.e_model,
                                                   other.e_model)
            i_like_you = other_score > curr.parent.lazy_my_score()
            you_like_me = other_score > other.parent.lazy_my_score()

            if self.config.debug:
                print('#i_like_you and you_like me\t%s\t%s\t%s\t%s\t%s' % (i_like_you,
                                                           you_like_me,other_score,curr.parent.lazy_my_score(),other.parent.lazy_my_score()))

            if self.config.aggressive_rejection_stop and not you_like_me and not you_like_me:
                if self.config.debug:
                    print('#tryGraftFast aggressive stop you dont like me and i dont like you')
                break

            if not you_like_me:
                other = other.parent
            elif you_like_me and not i_like_you:
                curr = curr.parent
            else:
                assert you_like_me and i_like_you
                print('#doingGraft')
                # We're going to graft.
                # [LOGGING] Write the tree before the graft
                if self.config.write_every_tree:
                    Graphviz.write_tree(
                        os.path.join(
                            self.config.canopy_out,
                            'tree_%s_before_graft_%s.gv' % (
                                p_idx, graft_index)),
                        self.root, [other.id, curr.id], [gnode.id])

                # Do the graft.
                assert other.parent
                prev_gp = other.parent.parent
                # new_ap_graft = self.hallucinate_merge(curr, other, None)
                new_graft_internal = curr.graft_to_me(
                    other,
                    new_aproj=None,
                    new_my_score=None)  # We don't want a pw guy here.

                # Update from new_graft_internal to the root.
                before_update_time = time.time()
                curr_update = new_graft_internal
                while curr_update:
                    curr_update.update_from_children()
                    curr_update = curr_update.parent
                after_update_time = time.time()

                print('#TimeForUpdateInGraft\t%s\t%s' % (
                    after_update_time - before_update_time,
                    after_update_time - start_time))

                # Update from previous parent to root.
                if prev_gp:
                    before_update_time = time.time()
                    curr_update = prev_gp
                    while curr_update:
                        curr_update.update_from_children()
                        curr_update = curr_update.parent
                    after_update_time = time.time()
                    print('#TimeForUpdateInPrevGPGraft\t%s\t%s' % (
                        after_update_time - before_update_time,
                        after_update_time - start_time))

                # Add new graft internal to the nn-struct.
                # self.nn_structure.insert(new_graft_internal)
                # TODO AK: doe we need this?
                self.root = new_graft_internal.root()

                # Write some trees.
                if self.config.write_every_tree:
                    Graphviz.write_tree(
                        os.path.join(self.config.canopy_out,
                                     'tree_%s_post_graft_%s.gv' % (
                                         p_idx, graft_index)),
                        self.root,
                        [other.id, curr.id],
                        [gnode.id])

                # Update offlimits.
                # offlimits.update({other})
                # offlimits.update(other.descendants())
                # return offlimits
                return new_graft_internal,knn_and_score_valid
        return None,knn_and_score_valid   # No graft found.


class Perch(Grinch):
    def __init__(self, config, dataset, model):
        super(Perch, self).__init__(config, dataset, model,
                                    perform_rotation=True,
                                    perform_graft=False)


class Greedy(Grinch):
    def __init__(self, config, dataset, model):
        super(Greedy, self).__init__(config, dataset, model,
                                     perform_rotation=False,
                                     perform_graft=False)