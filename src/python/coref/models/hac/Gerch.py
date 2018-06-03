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

import torch
import os
import numpy as np
import time
import sys
from scipy.special import expit
import random

from coref.models.core.AttributeProjection import AttributeProjection
from coref.models.core.MentNode import MentNode
from coref.models.nn import new_nn_structure
from coref.util.Graphviz import Graphviz
from coref.util.GraphvizNSW import GraphvizNSW
from heapq import heappush, heappop


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
    def __init__(self,gerch,graft_from,graft_to,accepted,allowed,none_available):
        self.gerch = gerch
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

class Gerch(object):
    def __init__(self, config, dataset, model,perform_rotation=True,perform_graft=True):
        self.config = config
        self.dataset = dataset
        self.model = model
        self.root = None  # type MentNode
        self.perform_rotation = perform_rotation
        self.perform_graft = perform_graft
        self.nn_structure = new_nn_structure(self.config.nn_structure,self.config,
                                             self.score_function_np)
        self.graft_recorder = GraftMetaDataRecorder()
        self.pair_to_pw = {}
        self.observed_classes =set()
        self.nn_k = self.config.nn_k if not self.config.exact_nn else np.inf
        self.nsw_r = self.config.nsw_r
        self.num_computations = 0
        # debugging the best pw cost
        self.time_in_best_pw = [0, 0, 0, 0, 0]
        self.time_in_leaves_best_pw = [0, 0]
        self.time_in_hallucinate_merge = [0,0]
        self.insert_time = [0,0]
        self.insert_comps = [0, 0]
        self.rotation_time = [0, 0]
        self.rotation_comps = [0, 0]
        self.grafting_time = [0, 0]
        self.grafting_comps = [0,0]
        self.e_hac_comps = [0,0]
        self.random = random.Random(self.config.random_seed)

    def reset_time_stats(self):
        self.time_in_best_pw[0] = 0
        self.time_in_leaves_best_pw[0] = 0
        self.time_in_best_pw[2] = 0
        self.time_in_hallucinate_merge[0] = 0
        self.insert_time[0] = 0
        self.rotation_time[0] = 0
        self.grafting_time[0] = 0
        self.insert_comps[0] = 0
        self.rotation_comps[0] = 0
        self.grafting_comps[0] = 0

    def hallucinate_merge(self,n1, n2, pw_score,debug_pw_score=None):
        start_time = time.time()
        ap = AttributeProjection()
        ap.update(n1.as_ment.attributes,self.model.sub_ent_model)
        ap.update(n2.as_ment.attributes,self.model.sub_ent_model)
        num_ms = n1.point_counter + n2.point_counter
        if 'tes' in ap.aproj_sum:
            ap.aproj_sum['tea'] = ap['tes'] / num_ms
        new_ap = AttributeProjection()
        new_ap.aproj_sum['pw'] = pw_score
        new_ap.aproj_bb['pw_bb'] = (pw_score, pw_score)
        ap.update(new_ap,self.model.sub_ent_model)
        ap.aproj_local['my_pw'] = pw_score
        ap.aproj_local['new_edges'] = n1.point_counter * n2.point_counter
        if debug_pw_score:
            ap.aproj_local_debug['my_pw'] = debug_pw_score

        left_child_entity_score = 1.0
        right_child_entity_score = 1.0

        if len(n1.children) > 0 and 'es' in n1.as_ment.attributes.aproj_local:
            left_child_entity_score = n1.as_ment.attributes.aproj_local['es']
            if self.config.expit_e_score:
                left_child_entity_score = expit(left_child_entity_score)
        if len(n2.children) > 0 and 'es' in n2.as_ment.attributes.aproj_local:
            right_child_entity_score = n2.as_ment.attributes.aproj_local['es']
            if self.config.expit_e_score:
                right_child_entity_score = expit(right_child_entity_score)

        if left_child_entity_score >= right_child_entity_score:
            ap.aproj_local['child_e_max'] = left_child_entity_score
            ap.aproj_local['child_e_min'] = right_child_entity_score
        else:
            ap.aproj_local['child_e_max'] = right_child_entity_score
            ap.aproj_local['child_e_min'] = left_child_entity_score
        assert ap.aproj_local['child_e_max'] <= 1.0
        assert ap.aproj_local['child_e_min'] <= 1.0
        assert ap.aproj_local['child_e_max'] >= 0.0
        assert ap.aproj_local['child_e_min'] >= 0.0
        end_time = time.time()
        self.time_in_hallucinate_merge[0] += end_time - start_time
        self.time_in_hallucinate_merge[1] += end_time - start_time
        return ap

    def pw_score(self,n1,n2):
        if (n1.id,n2.id) in self.pair_to_pw:
            return self.pair_to_pw[(n1.id, n2.id)]
        elif (n2.id,n1.id) in self.pair_to_pw:
            return self.pair_to_pw[(n2.id, n1.id)]
        else:
            pw = self.model.pw_score(n1.as_ment.attributes, n2.as_ment.attributes)
            self.pair_to_pw[(n1.id,n2.id)] = pw
            return pw

    def best_pairwise(self,n1,n2):
        if self.config.exact_best_pairwise:
            return self.exact_best_pairwise(n1, n2)
        else:
            return self.approx_best_pairwise_budget(n1, n2)

    def exact_best_pairwise(self,n1, n2):
        start_time_in_leaves = time.time()
        best_pw = None
        best_pw_n1 = None
        best_pw_n2 = None
        n1_leaves = n1.leaves()
        n2_leaves = n2.leaves()
        end_time_in_leaves = time.time()
        count = 0
        start_time = time.time()
        for n1p in n1_leaves:
            for n2p in n2_leaves:
                pw = self.pw_score(n1p, n2p)
                if best_pw is None or best_pw.data.numpy()[0] < pw.data.numpy()[0]:
                    best_pw = pw
                    best_pw_n1 = n1p.id
                    best_pw_n2 = n2p.id
                count += 1
        end_time = time.time()
        self.time_in_best_pw[0] += end_time - start_time
        self.time_in_best_pw[1] += end_time - start_time
        self.time_in_best_pw[2] += count
        self.time_in_best_pw[3] += count
        self.time_in_best_pw[4] += 1
        self.time_in_leaves_best_pw[0] += end_time_in_leaves - start_time_in_leaves
        self.time_in_leaves_best_pw[1] += end_time_in_leaves - start_time_in_leaves
        return best_pw,best_pw_n1,best_pw_n2

    def approx_best_pairwise_budget(self,n1, n2):
        start_time_in_leaves = time.time()
        best_pw = None
        best_pw_n1 = None
        best_pw_n2 = None
        budget = self.config.approx_best_pairwise_budget
        # Require n2_leaves > n1_leaves
        n1_leaves = n1.leaves()
        n2_leaves = n2.leaves()
        end_time_in_leaves = time.time()
        count = 0
        if len(n1_leaves) * len(n2_leaves) <= budget:
            start_time = time.time()
            for n1p in n1_leaves:
                for n2p in n2_leaves:
                    pw = self.pw_score(n1p, n2p)
                    if best_pw is None or best_pw.data.numpy()[0] < pw.data.numpy()[0]:
                        best_pw = pw
                        best_pw_n1 = n1p.id
                        best_pw_n2 = n2p.id
                    count += 1
            end_time = time.time()
        else:
            start_time = time.time()
            num_n1_leaves_minus_one = len(n1_leaves) - 1
            num_n2_leaves_minus_one = len(n2_leaves) - 1
            for i in range(budget):
                n1_l = self.random.randint(0,num_n1_leaves_minus_one)
                n2_l = self.random.randint(0,num_n2_leaves_minus_one)
                n1p = n1_leaves[n1_l]
                n2p = n2_leaves[n2_l]
                pw = self.pw_score(n1p, n2p)
                if best_pw is None or best_pw.data.numpy()[0] < pw.data.numpy()[0]:
                    best_pw = pw
                    best_pw_n1 = n1p.id
                    best_pw_n2 = n2p.id
                count += 1
            end_time = time.time()
        self.time_in_best_pw[0] += end_time - start_time
        self.time_in_best_pw[1] += end_time - start_time
        self.time_in_best_pw[2] += count
        self.time_in_best_pw[3] += count
        self.time_in_leaves_best_pw[0] += end_time_in_leaves - start_time_in_leaves
        self.time_in_leaves_best_pw[1] += end_time_in_leaves - start_time_in_leaves
        self.time_in_best_pw[4] += 1
        return best_pw,best_pw_n1,best_pw_n2

    def score_function_np(self, n1, n2):
        # For now we are checking all pairs of the pw scores.
        best_pw,best_pw_n1,best_pw_n2 = self.best_pairwise(n1,n2)
        e_score = self.model.e_score(self.hallucinate_merge(n1, n2, best_pw.data.numpy()[0])).data.numpy()[0]
        # if self.config.expit_e_score:
        #     e_score = expit(e_score)
        return e_score

    def score_np(self,n):
        return self.score_function_np(n.children[0],n.children[1])

    def find_insert(self, leaf_node, new_node):
        number_e_scores = 0
        start_time = time.time()
        pw_score = self.best_pairwise(leaf_node,new_node)[0].data.numpy()[0]
        # print('find_insert(%s,%s,%s) ' % (leaf_node.id, new_node.id, pw_score))
        curr = leaf_node
        ap = self.hallucinate_merge(curr, new_node, pw_score)
        new_score = self.model.e_score(ap).data.numpy()[0]
        number_e_scores += 1
        time_before_rotation = time.time()
        # print('\tcurr %s' % curr.id)
        # print('\tcurr.parent %s' % curr.parent.id if curr.parent else "None")
        # print('\tnew_score %s' % new_score)
        if self.perform_rotation:
            while curr.parent is not None:
                # print('\t curr.parent.my_score %s' % curr.parent.my_score)
                if new_score > curr.parent.my_score:
                    # print('\tnew_score > curr.parent.my_score, breaking')
                    break
                else:
                    curr = curr.parent
                    ap = self.hallucinate_merge(curr, new_node, pw_score)
                    new_score = self.model.e_score(ap).data.numpy()[0]
                    number_e_scores += 1
                    # print('\tcurr %s' % curr.id)
                    # print('\tcurr.parent %s' % (curr.parent.id if curr.parent else "None"))
                    # print('\tnew_score %s' % new_score)
        time_after_rotation = time.time()
        self.rotation_time[0] += time_after_rotation - start_time
        self.rotation_time[1] += time_after_rotation - start_time
        self.rotation_comps[0] += number_e_scores
        self.rotation_comps[1] += number_e_scores
        return curr, ap, new_score, time_before_rotation,time_after_rotation

    def update_with_leaf(self,curr,new_leaf,new_leaf_anc):
        node_who_is_not_anc = curr.children[0] if curr.children[0] not in new_leaf_anc else curr.children[1]
        new_pt_best_pw, _, _ = self.best_pairwise(new_leaf,node_who_is_not_anc)
        new_pt_best_pw_np = new_pt_best_pw.data.numpy()[0]
        if curr.as_ment.attributes.aproj_local['my_pw'] < new_pt_best_pw_np:
            curr.as_ment.attributes.aproj_local['my_pw'] = new_pt_best_pw_np
        e_score = self.model.e_score(self.hallucinate_merge(curr.children[0], curr.children[1],curr.as_ment.attributes.aproj_local['my_pw']))
        return e_score

    def update_for_new(self,curr,new_leaf,new_leaf_anc,for_leaf=False):
        if for_leaf:
            e_score = self.update_with_leaf(curr,new_leaf,new_leaf_anc).data.numpy()[0]
        else:
            e_score = self.score_np(curr)
        # if e_score != curr.my_score:
            # print('Updated my_score %s of curr my_score %s aproj_local[\'es\'] %s to be %s' % (curr.my_score,
            #                                                                                    curr.as_ment.attributes.aproj_local[
            #                                                                                        'es'] if 'es' in curr.as_ment.attributes.aproj_local else "None",
            #                                                                                    curr.id, e_score))
        curr.my_score = e_score
        curr.as_ment.attributes.aproj_local['es'] = e_score

    def pure_subtree_with(self,new_pt,nearest):
        for l in nearest.leaves():
            if new_pt.pts[0][1] != l.pts[0][1]:
                return False
        return True

    def insert(self, p,p_idx):
        """
        Incrementally add p to the tree.

        :param p - (MentObject,GroundTruth,Id)
        """

        p_ment = MentNode([p], aproj=p[0].attributes)
        p_ment.cluster_marker = True
        start_time = time.time()
        print('Inserting p (%s,%s,%s) into tree ' % (p_ment.id,p[1],p[2]))
        if self.root is None:
            self.root = p_ment
            self.nn_structure.insert(p_ment)
        else:
            # Find k nearest neighbors

            time_start_placement = time.time()
            if self.config.add_to_mention:
                offlimits = set([d.nsw_node for d in self.root.descendants() if d.point_counter > 1 if d.nsw_node])
            else:
                offlimits = set()

            # print('##########################################')
            # print("#### KNN SEARCH W/ New Point %s #############" % p_ment.id)

            insert_start_time = time.time()
            knn_and_score,num_searched_approx = self.nn_structure.knn_and_score_offlimits(p_ment, offlimits, k=self.nn_k,
                                                                      r=self.nsw_r)
            insert_end_time = time.time()
            self.insert_comps[0] += num_searched_approx
            self.insert_comps[1] += num_searched_approx
            self.insert_time[0] += insert_end_time - insert_start_time
            self.insert_time[1] += insert_end_time - insert_start_time
            self.num_computations += num_searched_approx

            approximate_closest_node, approx_closest_score = knn_and_score[0][1].v, knn_and_score[0][0]

            # possible_nn_with_same_class = p[1] in self.observed_classes

            # print("#KnnSearchRes\tNewMention\tapprox=%s\tapprox_score=%s" %
            #       (approximate_closest_node.id,approx_closest_score))
            #
            # print("#NumSearched\tNewMention\tapprox=%s\tnsw_edges=%s"
            #       "\ttree_nodes=%s\tscore=%s\tposs=%s"
            #       % (
            #         num_searched_approx,
            #         self.nn_structure.num_edges,p_idx * 2 - 1,
            #         approx_closest_score,possible_nn_with_same_class
            # ))
            #
            # print('##########################################')
            # print()
            # print('##########################################')
            # print("############## KNN ADD %s #############" % p_ment.id)
            #
            #
            # print('##########################################')
            # print()
            # print('##########################################')
            # print('############## Find Insert Stop ##########')

            # Find where to be added / rotate
            insert_node, new_ap, new_score, time_before_rotation,time_finish_placement = self.find_insert(approximate_closest_node,
                                                              p_ment)
            # print('Splitting Down at %s with new scores %s' % (insert_node.id, new_score))


            # print('#TimeNNFindTime\t%s\t%s' % (time_before_rotation - time_start_placement,time_before_rotation-start_time))
            # print('#TimeUntilAfterRotation\t%s\t%s' % (time_finish_placement - time_start_placement,time_finish_placement-start_time))

            time_before_insert = time.time()
            # Add yourself to the knn structures
            num_comp_insertions = self.nn_structure.insert(p_ment)
            time_after_insert = time.time()

            self.insert_comps[0] += num_comp_insertions
            self.insert_comps[1] += num_comp_insertions
            self.insert_time[0] += time_after_insert - time_before_insert
            self.insert_time[1] += time_after_insert - time_before_insert

            # print('#TimeAddPointToNSW\t%s\t%s' % (time_after_insert-time_before_insert,time_after_insert-start_time))

            # Add the point
            new_internal_node = insert_node.split_down(p_ment, new_ap, new_score)

            assert p_ment.root() == insert_node.root(), "p_ment.root() %s == insert_node.root() %s" % (
            p_ment.root(), insert_node.root())
            assert p_ment.lca(
                insert_node) == new_internal_node, "p_ment.lca(insert_node) %s == new_internal_node %s" % (
            p_ment.lca(insert_node), new_internal_node)

            # print('Created new node %s ' % new_internal_node.id)

            # Update throughout the tree.
            if new_internal_node.parent:
                new_internal_node.parent.update_aps(p[0].attributes,self.model.sub_ent_model)

            # update all the entity scores
            before_update_time = time.time()
            curr = new_internal_node
            new_leaf_anc = p_ment._ancestors()
            num_updates_here = 0
            while curr:
                self.update_for_new(curr,p_ment,new_leaf_anc,True)
                curr = curr.parent
                num_updates_here += 1
            after_update_time = time.time()
            self.insert_comps[0] += num_updates_here
            self.insert_comps[1] += num_updates_here
            self.insert_time[0] += after_update_time - before_update_time
            self.insert_time[1] += after_update_time - before_update_time


            # print('#TimeForUpdateOfNewPt\t%s\t%s' %(after_update_time-before_update_time,after_update_time-start_time))
            # print('##########################################')
            # print()
            # print('##########################################')
            # print("############## KNN ADD %s #############" % new_internal_node.id)

            # Add the newly created node to the NN structure
            time_before_insert = time.time()
            num_comp_insertions = self.nn_structure.insert(new_internal_node)
            time_after_insert = time.time()
            self.insert_comps[0] += num_comp_insertions
            self.insert_comps[1] += num_comp_insertions
            self.insert_time[0] += time_after_insert - time_before_insert
            self.insert_time[1] += time_after_insert - time_before_insert

            # print('#TimeAddInternalNodetoNSW\t%s\t%s' % (time_after_insert - time_before_insert, time_after_insert - start_time))

            # print()
            # print('##########################################')
            # print()

            self.root = self.root.root()
            time_before_graft = time.time()
            total_graft_comps = 0
            if self.perform_graft:
                graft_index = 0

                curr = new_internal_node
                while curr.parent:
                    time_before_this_graft = time.time()
                    # print()
                    # print("=============================================")
                    # print('Curr %s CurrType %s ' % (curr.id, type(curr)))
                    #
                    # print('Finding Graft for %s ' % curr.id)
                    #
                    # print('##########################################')
                    # print("#### KNN SEARCH W/ Node %s #########" % curr.id)

                    time_before_offlimits = time.time()
                    offlimits = set(
                        [x.nsw_node for x in (curr.siblings() + curr.descendants() + curr._ancestors() + [curr])])
                    time_after_offlimits = time.time()
                    # print('#TimeFindOfflimits\t%s\t%s' % (time_after_offlimits-time_before_offlimits,time_after_offlimits-start_time))

                    time_before_graft_nn_search = time.time()
                    knn_and_score,num_searched_approx = self.nn_structure.knn_and_score_mention(curr,offlimits,
                                                                            k=self.nn_k,
                                                                            r=self.nsw_r)
                    time_after_graft_nn_search = time.time()
                    # print('#TimeNNGraftSearch\t%s\t%s' %(time_after_graft_nn_search-time_before_graft_nn_search,time_after_graft_nn_search-start_time))
                    self.num_computations += num_searched_approx
                    total_graft_comps += num_searched_approx

                    # if len(knn_and_score) == 0:
                    #     print("#NumSearched\tGraft\tapprox=%s\texact=%s\tnsw_edges=%s\terror="
                    #       % (num_searched_approx,self.nn_structure.num_edges,
                    #          p_idx * 2))
                    # print('##########################################')
                    # print()

                    if len(knn_and_score) > 0:
                        approximate_closest_node, approx_closest_score = knn_and_score[0][1].v, knn_and_score[0][0]
                        # print("#NumSearched\tGraft\tapprox=%s\tnsw_edges=%s\ttree_nodes=%s\terror=%s"
                        #      % (num_searched_approx, self.nn_structure.num_edges,
                        #         p_idx * 2, np.abs(approx_closest_score)))
                        # print("#KnnSearchRes\tGraft\tapprox=%s\tapprox_score=%s" %
                        #       (approximate_closest_node.id, approx_closest_score))

                        def allowable_graft(n):
                            if n.deleted:
                                print('Deleted')
                                return False
                            if n.parent is None:
                                # print('Parent is None')
                                return False
                            if curr in n.siblings():
                                # print('curr in sibs')
                                return False
                            lca = curr.lca(n)
                            if lca != curr and lca != n:
                                # print("Found candidate - returning true")
                                return True
                            else:
                                # print('lca = curr %s lca = n %s' % (lca == curr, lca == n))
                                return False

                        # allowed = allowable_graft(best)
                        allowed = True
                        if not allowed:
                            # self.graft_recorder.records.append(GraftMetaData(self, curr, best, False,False,False))
                            pass
                        else:
                            # print(approx_closest_score)
                            # print(curr.parent.my_score)
                            # print(approximate_closest_node.parent.my_score)
                            # print('Best %s BestTypes %s ' % (approximate_closest_node.id,type(approximate_closest_node)))

                            you_like_them_better = approx_closest_score > curr.parent.my_score
                            they_like_you_better = approx_closest_score > approximate_closest_node.parent.my_score

                            approx_says_perform_graft = you_like_them_better and they_like_you_better
                            is_allowable = True
                            while you_like_them_better \
                                    and not they_like_you_better \
                                    and is_allowable \
                                    and approximate_closest_node.parent \
                                    and approximate_closest_node.parent.parent:
                                approximate_closest_node = approximate_closest_node.parent
                                is_allowable = allowable_graft(approximate_closest_node)
                                if is_allowable:
                                    best_pw,best_pw_n1,best_pw_n2 = self.best_pairwise(curr,approximate_closest_node)
                                    new_ap_graft = self.hallucinate_merge(curr, approximate_closest_node,
                                                                      best_pw.data.numpy()[0])
                                    approx_closest_score = self.model.e_score(new_ap_graft).data.numpy()[0]
                                    total_graft_comps += 1
                                    you_like_them_better = approx_closest_score > curr.parent.my_score
                                    they_like_you_better = approx_closest_score > approximate_closest_node.parent.my_score

                                    approx_says_perform_graft = you_like_them_better and they_like_you_better


                            # if you like them better than your current sibling, but they don't like you better then you
                            # want to check the parent of them.


                            # print('(Approx.) Candidate Graft: (best: %s, score: %s) to (%s,par.score %s) from (%s,par.score %s)' %
                            #       (approximate_closest_node.id,approx_closest_score,curr.id,curr.parent.my_score,approximate_closest_node.id,approximate_closest_node.parent.my_score))
                            # Perform Graft
                            # print("#GraftSuggestions\tp_idx=%s\tg_idx=%s\tapprox=%s" %
                            #       (p_idx,graft_index,approx_says_perform_graft))

                            if approx_says_perform_graft:
                                approximate_closest_node_sib = approximate_closest_node.siblings()[0]

                                # Write the tree before the graft
                                if self.config.write_every_tree:
                                    Graphviz.write_tree(os.path.join(self.config.canopy_out,
                                                                     'tree_%s_before_graft_%s.gv' % (
                                                                     p_idx, graft_index)), self.root,
                                                        [approximate_closest_node.id, curr.id],[p_ment.id])
                                # self.graft_recorder.records.append(GraftMetaData(self, best, curr, True, True, False))
                                # print("Performing graft: ")
                                best_pw,best_pw_n1,best_pw_n2 = self.best_pairwise(curr,approximate_closest_node)
                                # print('best_pw = %s %s %s' % (best_pw_n1,best_pw_n2,best_pw))
                                new_ap_graft = self.hallucinate_merge(curr,approximate_closest_node,best_pw.data.numpy()[0])
                                new_graft_internal = curr.graft_to_me(approximate_closest_node, new_aproj=new_ap_graft, new_my_score=None) # We don't want a pw guy here.

                                # print('Finished Graft')
                                # print('updating.....')
                                # Update nodes

                                # This updates the ancestors of the current node after the graft

                                before_update_time = time.time()
                                curr_update = new_graft_internal
                                while curr_update:
                                    e_score = self.score_np(curr_update)
                                    total_graft_comps += 1
                                    # if e_score != curr_update.my_score:
                                    #     print(
                                    #         'Updated my_score %s of curr my_score %s aproj_local[\'es\'] %s to be %s' % (
                                    #             curr_update.my_score,
                                    #             curr_update.as_ment.attributes.aproj_local[
                                    #             'es'] if 'es' in curr_update.as_ment.attributes.aproj_local else "None",
                                    #             curr_update.id, e_score))
                                    curr_update.my_score = e_score
                                    curr_update.as_ment.attributes.aproj_local['es'] = e_score
                                    if curr_update.parent is None:
                                        self.root = curr_update
                                    curr_update = curr_update.parent
                                after_update_time = time.time()


                                # This updates the ancestors of the node which was grafted to you:
                                sibling_of_grafted_node = approximate_closest_node_sib
                                curr_update = sibling_of_grafted_node.parent
                                while curr_update:
                                    e_score = self.score_np(curr_update)
                                    total_graft_comps += 1
                                    # if e_score != curr_update.my_score:
                                    #     print(
                                    #         '[From Graftees old sib] Updated my_score %s of curr my_score %s aproj_local[\'es\'] %s to be %s' % (
                                    #             curr_update.my_score,
                                    #             curr_update.as_ment.attributes.aproj_local[
                                    #             'es'] if 'es' in curr_update.as_ment.attributes.aproj_local else "None",
                                    #             curr_update.id, e_score))
                                    curr_update.my_score = e_score
                                    curr_update.as_ment.attributes.aproj_local['es'] = e_score
                                    curr_update = curr_update.parent

                                print('#TimeForUpdateInGraft\t%s\t%s' % (after_update_time-before_update_time,after_update_time - start_time))
                                # print('##########################################')
                                # print("############## KNN ADD %s #############" % new_graft_internal.id)
                                # print('Adding new node to NSW')
                                insert_comps = self.nn_structure.insert(new_graft_internal)
                                total_graft_comps += insert_comps
                                # print('##########################################')
                                # Write the tree after the graft
                                if self.config.write_every_tree:
                                    Graphviz.write_tree(os.path.join(self.config.canopy_out,
                                                                     'tree_%s_post_graft_%s.gv' % (p_idx, graft_index)),
                                                        self.root, [approximate_closest_node.id, curr.id],[p_ment.id])

                            # else:
                                # self.graft_recorder.records.append(GraftMetaData(self, best, curr, False, True, False))
                                # print('Chose not to graft.')

                    # else:
                        # self.graft_recorder.records.append(GraftMetaData(self, None, curr, False, False, True))
                        # print('No possible grafts for %s ' % curr.id)
                    graft_index += 1
                    curr = curr.parent
                    time_after_this_graft = time.time()
                    print("#TimeAfterThisGraftProposal\t%s\t%s" % (
                    time_after_this_graft - time_before_this_graft, time_after_this_graft - start_time))
                    # print("=============================================")
                    # print()
                    end_time = time.time()
                    if curr.parent is None:
                        self.grafting_time[0] += end_time - time_before_graft
                        self.grafting_time[1] += end_time - time_before_graft
                        self.grafting_comps[0] += total_graft_comps
                        self.grafting_comps[1] += total_graft_comps
                        print("#TimeAfterAllGrafts\t%s\t%s" % (
                            end_time - time_before_graft, end_time - start_time))
        end_time = time.time()
        print('Done Inserting p (%s,%s,%s) into tree in %s seconds  ' % (p_ment.id, p[1], p[2],end_time-start_time))
        self.observed_classes.add(p[1])
        sys.stdout.flush()
        if self.config.write_every_tree:
            if len(self.config.canopy_out) > 0:
                Graphviz.write_tree(os.path.join(self.config.canopy_out,
                                             'tree_%s.gv' % p_idx), self.root,[], [p_ment.id])
                if self.config.nn_structure == 'nsw':
                    GraphvizNSW.write_nsw(os.path.join(self.config.canopy_out, 'nsw_%s.gv' % p_idx), self.nn_structure)
        return p_ment

    def build_dendrogram(self):
        idx = 0
        tot_time = 0
        for p in self.dataset:
            print('[build dendrogram] Inserting pt number %s into tree' % idx)
            start_time = time.time()
            p_ment = self.insert(p, idx)
            idx += 1
            end_time = time.time()
            this_pt_time = end_time-start_time
            tot_time += this_pt_time
            if self.config.time_debug:
                self.e_hac_comps[0] = ((idx * (idx - 1)) / 2) * np.log2(idx)
                self.e_hac_comps[1] += ((idx * (idx - 1)) / 2) * np.log2(idx)
                p_depth = p_ment.depth()
                print('#TimePerPoint\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % (idx, this_pt_time, tot_time, p_depth,
                                                                                 self.insert_time[0],
                                                                                 self.insert_time[1],
                                                                                 self.rotation_time[0],
                                                                                 self.rotation_time[1],
                                                                                 self.grafting_time[0],
                                                                                 self.grafting_time[1]))
                print('#CompPerPoint\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % (
                idx, self.e_hac_comps[0],self.e_hac_comps[1],
                self.insert_comps[0] + self.rotation_comps[0] + self.grafting_comps[0],
                self.insert_comps[1] + self.rotation_comps[1] + self.grafting_comps[1],
                self.insert_comps[0],
                self.insert_comps[1],
                self.rotation_comps[0],
                self.rotation_comps[1],
                self.grafting_comps[0],
                self.grafting_comps[1]))

                print('#HallucinateMergeTime\t%s\t%s\t%s\t%s\t%s\t%s' % (idx, this_pt_time, tot_time, p_depth,
                                                             self.time_in_hallucinate_merge[0],
                                                             self.time_in_hallucinate_merge[1]))
                print('#BestPWTime\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % (idx, this_pt_time, tot_time, p_depth,
                                                                       self.time_in_best_pw[0],
                                                                       self.time_in_best_pw[1],
                                                                       (self.time_in_best_pw[1] / self.time_in_best_pw[4]) if self.time_in_best_pw[4] > 0 else 0,
                                                                       self.time_in_best_pw[4],
                                                                       self.time_in_leaves_best_pw[0],
                                                                       self.time_in_leaves_best_pw[1]))
                self.reset_time_stats()
        self.graft_recorder.report()
        return self.root

    def build_dendrogram_rand_order(self):
        idx = 0
        import random
        r = random.Random(self.config.random_seed)
        r.shuffle(self.dataset)
        for p in self.dataset:
            print('[build dendrogram] Inserting pt number %s into tree' % idx)
            self.insert(p,idx)
            idx += 1
        self.graft_recorder.report()
        return self.root


class Perch(Gerch):
    def __init__(self, config, dataset, model):
        super(Perch,self).__init__(config,dataset,model,perform_rotation=True,perform_graft=False)

class Greedy(Gerch):
    def __init__(self, config, dataset, model):
        super(Greedy,self).__init__(config,dataset,model,perform_rotation=False,perform_graft=False)
