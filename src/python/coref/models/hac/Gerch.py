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

from scipy.special import expit

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
        self.mention_nn_structure = new_nn_structure(self.config.mention_nn_structure,self.config,
                                             self.score_function_np)
        self.nn_structure = new_nn_structure(self.config.nn_structure,self.config,
                                             self.score_function_np)
        self.graft_recorder = GraftMetaDataRecorder()
        self.pair_to_pw = {}
        self.observed_classes =set()
        self.nn_k = self.config.nn_k if not self.config.exact_nn else np.inf
        self.nsw_r = self.config.nsw_r
        self.num_computations = 0

    def hallucinate_merge(self,n1, n2, pw_score,debug_pw_score=None):
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

        if 'es' in n1.as_ment.attributes.aproj_local:
            left_child_entity_score = n1.as_ment.attributes.aproj_local['es']
        if 'es' in n2.as_ment.attributes.aproj_local:
            right_child_entity_score = n2.as_ment.attributes.aproj_local['es']

        if left_child_entity_score >= right_child_entity_score:
            ap.aproj_local['child_e_max'] = left_child_entity_score
            ap.aproj_local['child_e_min'] = right_child_entity_score
        else:
            ap.aproj_local['child_e_max'] = right_child_entity_score
            ap.aproj_local['child_e_min'] = left_child_entity_score

        # print('score_fn')
        # print('ap.aproj_local')
        # print(ap.aproj_local)
        # print('n1.as_ment.attributes[\'es\']')
        # print(n1.as_ment.attributes['es'])
        # print('n2.as_ment.attributes[\'es\']')
        # print(n2.as_ment.attributes['es'])
        # print('n1.as_ment.attributes.aproj_local')
        # print(n1.as_ment.attributes.aproj_local)
        # print('n1.as_ment.attributes.aproj_local')
        # print(n1.as_ment.attributes.aproj_local)
        # print()
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

    def best_pairwise(self,n1, n2):
        best_pw = None
        best_pw_n1 = None
        best_pw_n2 = None
        n1_leaves = n1.leaves()
        n2_leaves = n2.leaves()
        for n1p in n1_leaves:
            for n2p in n2_leaves:
                pw = self.pw_score(n1p, n2p)
                if best_pw is None or best_pw.data.numpy()[0] < pw.data.numpy()[0]:
                    best_pw = pw
                    best_pw_n1 = n1p.id
                    best_pw_n2 = n2p.id
        return best_pw,best_pw_n1,best_pw_n2

    def score_function_np(self, n1, n2):
        # For now we are checking all pairs of the pw scores.
        best_pw,best_pw_n1,best_pw_n2 = self.best_pairwise(n1,n2)
        return self.model.e_score(self.hallucinate_merge(n1, n2, best_pw.data.numpy()[0])).data.numpy()[0]

    def score_function(self, n1, n2):
        # For now we are checking all pairs of the pw scores.
        best_pw,best_pw_n1,best_pw_n2 = self.best_pairwise(n1,n2)
        return self.model.e_score(self.hallucinate_merge(n1, n2, best_pw.data.numpy()[0]))

    def score(self,n):
        return self.score_function(n.children[0],n.children[1])

    def find_best_across(self,n):
        # best_pw,best_pw_n1,best_pw_n2 = self.best_pairwise(n.children[0],n.children[1])
        # n.best_across_debug =  "%s %s %s" % (best_pw.data.numpy()[0],best_pw_n1,best_pw_n2)
        n.best_across_debug = "no debug"

    def find_insert(self, leaf_node, new_node, pw_score):
        print('find_insert(%s,%s,%s) ' % (leaf_node.id, new_node.id, pw_score))
        curr = leaf_node
        ap = self.hallucinate_merge(curr, new_node, pw_score)
        new_score = self.model.e_score(ap).data.numpy()[0]
        print('\tcurr %s' % curr.id)
        print('\tcurr.parent %s' % curr.parent.id if curr.parent else "None")
        print('\tnew_score %s' % new_score)
        if self.perform_rotation:
            while curr.parent is not None:
                print('\t curr.parent.my_score %s' % curr.parent.my_score)
                if new_score > curr.parent.my_score:
                    print('\tnew_score > curr.parent.my_score, breaking')
                    break
                else:
                    curr = curr.parent
                    ap = self.hallucinate_merge(curr, new_node, pw_score)
                    new_score = self.model.e_score(ap).data.numpy()[0]
                    print('\tcurr %s' % curr.id)
                    print('\tcurr.parent %s' % (curr.parent.id if curr.parent else "None"))
                    print('\tnew_score %s' % new_score)
        return curr, ap, new_score

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
            e_score = self.score(curr).data.numpy()[0]
        if e_score != curr.my_score:
            print('Updated my_score %s of curr my_score %s aproj_local[\'es\'] %s to be %s' % (curr.my_score,
                                                                                               curr.as_ment.attributes.aproj_local[
                                                                                                   'es'] if 'es' in curr.as_ment.attributes.aproj_local else "None",
                                                                                               curr.id, e_score))
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

        print()
        print()
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('Inserting p (%s,%s,%s) into tree ' % (p_ment.id,p[1],p[2]))
        if self.root is None:
            self.root = p_ment
            # self.mention_nn_structure.insert(p_ment)
            self.nn_structure.insert(p_ment)
        else:
            # Find k nearest neighbors

            if self.config.add_to_mention:
                offlimits = set([d.nsw_node for d in self.root.descendants() if d.point_counter > 1 if d.nsw_node])
            else:
                offlimits = set()

            print('##########################################')
            print("#### KNN SEARCH W/ New Point %s #############" % p_ment.id)

            knn_and_score,num_searched_approx = self.nn_structure.knn_and_score_offlimits(p_ment, offlimits, k=self.nn_k,
                                                                      r=self.nsw_r)
            self.num_computations += num_searched_approx

            approximate_closest_node, approx_closest_score = knn_and_score[0][1].v, knn_and_score[0][0]

            possible_nn_with_same_class = p[1] in self.observed_classes

            print("#KnnSearchRes\tNewMention\tapprox=%s\tapprox_score=%s" %
                  (approximate_closest_node.id,approx_closest_score))

            print("#NumSearched\tNewMention\tapprox=%s\tnsw_edges=%s"
                  "\ttree_nodes=%s\tscore=%s\tposs=%s"
                  % (
                    num_searched_approx,
                    self.nn_structure.num_edges,p_idx * 2 - 1,
                    approx_closest_score,possible_nn_with_same_class
            ))

            print('##########################################')
            print()
            print('##########################################')
            print("############## KNN ADD %s #############" % p_ment.id)

            # Add yourself to the knn structures
            self.nn_structure.insert(p_ment)
            print('##########################################')
            print()
            print('##########################################')
            print('############## Find Insert Stop ##########')

            # Find where to be added / rotate
            insert_node, new_ap, new_score = self.find_insert(knn_and_score[0][1].v,
                                                              p_ment,
                                                              knn_and_score[0][0])
            print('Splitting Down at %s with new scores %s' % (insert_node.id, new_score))

            # Add the point
            new_internal_node = insert_node.split_down(p_ment, new_ap, new_score)
            assert p_ment.root() == insert_node.root(), "p_ment.root() %s == insert_node.root() %s" % (
            p_ment.root(), insert_node.root())
            assert p_ment.lca(
                insert_node) == new_internal_node, "p_ment.lca(insert_node) %s == new_internal_node %s" % (
            p_ment.lca(insert_node), new_internal_node)

            print('Created new node %s ' % new_internal_node.id)

            # Update throughout the tree.
            if new_internal_node.parent:
                new_internal_node.parent.update_aps(p[0].attributes,self.model.sub_ent_model)

            # update all the entity scores
            curr = new_internal_node
            new_leaf_anc = p_ment._ancestors()
            while curr:
                self.update_for_new(curr,p_ment,new_leaf_anc,True)
                curr = curr.parent
            print('##########################################')
            print()
            print('##########################################')
            print("############## KNN ADD %s #############" % new_internal_node.id)

            # Add the newly created node to the NN structure
            self.nn_structure.insert(new_internal_node)
            print()
            print('##########################################')
            print()

            self.root = self.root.root()

            if self.perform_graft:
                graft_index = 0

                curr = new_internal_node
                while curr.parent:
                    print()
                    print("=============================================")
                    print('Curr %s CurrType %s ' % (curr.id, type(curr)))

                    # find nearest neighbor in entity space (not one of your descendants)
                    def filter_condition(n):
                        if n.deleted or n == curr:
                            return False
                        else:
                            return True

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

                    print('Finding Graft for %s ' % curr.id)

                    print('##########################################')
                    print("#### KNN SEARCH W/ Node %s #########" % curr.id)

                    offlimits = set([x.nsw_node for x in (curr.siblings() + curr.descendants() + curr._ancestors() + [curr])])

                    knn_and_score,num_searched_approx = self.nn_structure.knn_and_score_offlimits(curr, offlimits, k=self.nn_k,
                                                                              r=self.nsw_r)

                    self.num_computations += num_searched_approx

                    if len(knn_and_score) == 0:
                        print("#NumSearched\tGraft\tapprox=%s\texact=%s\tnsw_edges=%s\terror="
                          % (num_searched_approx,self.nn_structure.num_edges,
                             p_idx * 2))
                    print('##########################################')
                    print()

                    if len(knn_and_score) > 0:
                        approximate_closest_node, approx_closest_score = knn_and_score[0][1].v, knn_and_score[0][0]
                        approximate_closest_node_sib = approximate_closest_node.siblings()[0]
                        print("#NumSearched\tGraft\tapprox=%s\tnsw_edges=%s\ttree_nodes=%s\terror=%s"
                             % (num_searched_approx, self.nn_structure.num_edges,
                                p_idx * 2, np.abs(approx_closest_score)))
                        print("#KnnSearchRes\tGraft\tapprox=%s\tapprox_score=%s" %
                              (approximate_closest_node.id, approx_closest_score))

                        # allowed = allowable_graft(best)
                        allowed = True
                        if not allowed:
                            # self.graft_recorder.records.append(GraftMetaData(self, curr, best, False,False,False))
                            pass
                        else:
                            print(approx_closest_score)
                            print(curr.parent.my_score)
                            print(approximate_closest_node.parent.my_score)
                            print('Best %s BestTypes %s ' % (approximate_closest_node.id,type(approximate_closest_node)))
                            approx_says_perform_graft = approx_closest_score > curr.parent.my_score and approx_closest_score > approximate_closest_node.parent.my_score

                            print('(Approx.) Candidate Graft: (best: %s, score: %s) to (%s,par.score %s) from (%s,par.score %s)' %
                                  (approximate_closest_node.id,approx_closest_score,curr.id,curr.parent.my_score,approximate_closest_node.id,approximate_closest_node.parent.my_score))
                            # Perform Graft
                            print("#GraftSuggestions\tp_idx=%s\tg_idx=%s\tapprox=%s" %
                                  (p_idx,graft_index,approx_says_perform_graft))

                            if approx_says_perform_graft:

                                # Write the tree before the graft
                                if self.config.write_every_tree:
                                    Graphviz.write_tree(os.path.join(self.config.canopy_out,
                                                                     'tree_%s_before_graft_%s.gv' % (
                                                                     p_idx, graft_index)), self.root,
                                                        [approximate_closest_node.id, curr.id],[p_ment.id])
                                # self.graft_recorder.records.append(GraftMetaData(self, best, curr, True, True, False))
                                print("Performing graft: ")
                                best_pw,best_pw_n1,best_pw_n2 = self.best_pairwise(curr,approximate_closest_node)
                                print('best_pw = %s %s %s' % (best_pw_n1,best_pw_n2,best_pw))
                                new_ap_graft = self.hallucinate_merge(curr,approximate_closest_node,best_pw.data.numpy()[0])
                                new_graft_internal = curr.graft_to_me(approximate_closest_node, new_aproj=new_ap_graft, new_my_score=None) # We don't want a pw guy here.

                                print('Finished Graft')
                                print('updating.....')
                                # Update nodes
                                curr_update = new_graft_internal
                                while curr_update:
                                    e_score = self.score(curr_update).data.numpy()[0]
                                    if e_score != curr_update.my_score:
                                        print(
                                            'Updated my_score %s of curr my_score %s aproj_local[\'es\'] %s to be %s' % (
                                            curr.my_score,
                                            curr.as_ment.attributes.aproj_local[
                                                'es'] if 'es' in curr.as_ment.attributes.aproj_local else "None",
                                            curr.id, e_score))
                                        curr_update.my_score = e_score
                                    curr.as_ment.attributes.aproj_local['es'] = e_score
                                    curr_update = curr_update.parent
                                # Set the root of the tree after the graft: TODO: This could be sped up by being in the update loop
                                self.root = new_graft_internal.root()
                                print('##########################################')
                                print("############## KNN ADD %s #############" % new_graft_internal.id)
                                print('Adding new node to NSW')
                                self.nn_structure.insert(new_graft_internal)
                                print('Updating NSW for sibling of node that was grafted')
                                self.nn_structure.add_tree_edges(approximate_closest_node_sib)
                                print('##########################################')
                                # Write the tree after the graft
                                if self.config.write_every_tree:
                                    Graphviz.write_tree(os.path.join(self.config.canopy_out,
                                                                     'tree_%s_post_graft_%s.gv' % (p_idx, graft_index)),
                                                        self.root, [approximate_closest_node.id, curr.id],[p_ment.id])

                            else:
                                # self.graft_recorder.records.append(GraftMetaData(self, best, curr, False, True, False))
                                print('Chose not to graft.')

                    else:
                        # self.graft_recorder.records.append(GraftMetaData(self, None, curr, False, False, True))
                        print('No possible grafts for %s ' % curr.id)
                    graft_index += 1
                    curr = curr.parent
                    print("=============================================")
                    print()
        print()
        print('Done inserting p (%s,%s,%s) into tree ' % (p_ment.id, p[1], p[2]))
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print()
        self.observed_classes.add(p[1])
        if self.config.write_every_tree:
            if len(self.config.canopy_out) > 0:
                Graphviz.write_tree(os.path.join(self.config.canopy_out,
                                             'tree_%s.gv' % p_idx), self.root,[], [p_ment.id])
                if self.config.nn_structure == 'nsw':
                    GraphvizNSW.write_nsw(os.path.join(self.config.canopy_out, 'nsw_%s.gv' % p_idx), self.nn_structure)


    def insert_instrumented(self, p,p_idx):
        """
        Incrementally add p to the tree.

        :param p - (MentObject,GroundTruth,Id)
        """

        p_ment = MentNode([p], aproj=p[0].attributes)
        p_ment.cluster_marker = True

        print()
        print()
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('Inserting p (%s,%s,%s) into tree ' % (p_ment.id,p[1],p[2]))
        if self.root is None:
            self.root = p_ment
            # self.mention_nn_structure.insert(p_ment)
            self.nn_structure.insert(p_ment)
        else:
            # Find k nearest neighbors

            # offlimits = set([d.nsw_node for d in self.root.descendants() if d.point_counter > 1 if d.nsw_node])
            offlimits = set() #[d.nsw_node for d in self.root.descendants() if d.point_counter > 1 if d.nsw_node])

            print('##########################################')
            print("#### KNN SEARCH W/ New Point %s #############" % p_ment.id)

            knn_and_score,num_searched_approx = self.nn_structure.knn_and_score_offlimits(p_ment, offlimits, k=self.nn_k,
                                                                      r=self.nsw_r)
            exact_knn_and_score,num_searched_exact = self.nn_structure.exact_knn(p_ment, offlimits, k=self.nn_k,
                                                                      r=self.nsw_r)
            self.num_computations += num_searched_approx

            approximate_closest_node, approx_closest_score = knn_and_score[0][1].v, knn_and_score[0][0]
            exact_closest_node, exact_closest_score = exact_knn_and_score[0][1].v, exact_knn_and_score[0][0]

            possible_nn_with_same_class = p[1] in self.observed_classes
            approx_pure_subtree = self.pure_subtree_with(p_ment, approximate_closest_node)
            exact_pure_subtree = self.pure_subtree_with(p_ment, exact_closest_node)



            approx_eq_exact = approximate_closest_node == exact_closest_node
            print("#KnnSearchRes\tNewMention\t%s\tapprox=%s\tapprox_score=%s\texact=%s\texact_score=%s" %
                  (approx_eq_exact,approximate_closest_node.id,approx_closest_score,
                   exact_closest_node.id,exact_closest_score))

            print("#NumSearched\tNewMention\tapprox=%s\texact=%s\tnsw_edges=%s"
                  "\ttree_nodes=%s\tspeedup=%s\terror=%s\tposs=%s\tapprox_same_class=%s\t"
                  "exact_same_class=%s" % (
            num_searched_approx, num_searched_exact,
            self.nn_structure.num_edges,p_idx * 2 - 1,
            num_searched_approx-num_searched_exact,
            np.abs(approx_closest_score-exact_closest_score),possible_nn_with_same_class,
            approx_pure_subtree,exact_pure_subtree))

            print('##########################################')
            print()
            print('##########################################')
            print("############## KNN ADD %s #############" % p_ment.id)

            # Add yourself to the knn structures
            self.nn_structure.insert(p_ment)
            print('##########################################')
            print()
            print('##########################################')
            print('############## Find Insert Stop ##########')

            # Find where to be added / rotate
            insert_node, new_ap, new_score = self.find_insert(knn_and_score[0][1].v,
                                                              p_ment,
                                                              knn_and_score[0][0])
            print('Splitting Down at %s with new scores %s' % (insert_node.id, new_score))

            # Add the point
            new_internal_node = insert_node.split_down(p_ment, new_ap, new_score)
            assert p_ment.root() == insert_node.root(), "p_ment.root() %s == insert_node.root() %s" % (
            p_ment.root(), insert_node.root())
            assert p_ment.lca(
                insert_node) == new_internal_node, "p_ment.lca(insert_node) %s == new_internal_node %s" % (
            p_ment.lca(insert_node), new_internal_node)

            print('Created new node %s ' % new_internal_node.id)

            # Update throughout the tree.
            if new_internal_node.parent:
                new_internal_node.parent.update_aps(p[0].attributes,self.model.sub_ent_model)

            # update all the entity scores
            curr = new_internal_node
            new_leaf_anc = p_ment._ancestors()
            while curr:
                self.update_for_new(curr,p_ment,new_leaf_anc,True)
                curr = curr.parent
            print('##########################################')
            print()
            print('##########################################')
            print("############## KNN ADD %s #############" % new_internal_node.id)

            # Add the newly created node to the NN structure
            self.nn_structure.insert(new_internal_node)
            print()
            print('##########################################')
            print()

            self.root = self.root.root()

            if self.perform_graft:
                graft_index = 0

                curr = new_internal_node
                while curr.parent:
                    print()
                    print("=============================================")
                    print('Curr %s CurrType %s ' % (curr.id, type(curr)))

                    # find nearest neighbor in entity space (not one of your descendants)
                    def filter_condition(n):
                        if n.deleted or n == curr:
                            return False
                        else:
                            return True

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

                    print('Finding Graft for %s ' % curr.id)

                    print('##########################################')
                    print("#### KNN SEARCH W/ Node %s #########" % curr.id)

                    offlimits = set([x.nsw_node for x in (curr.siblings() + curr.descendants() + curr._ancestors() + [curr])])

                    knn_and_score,num_searched_approx = self.nn_structure.knn_and_score_offlimits(curr, offlimits, k=self.nn_k,
                                                                              r=self.nsw_r)
                    exact_knn_and_score,num_searched_exact = self.nn_structure.exact_knn(curr, offlimits, k=self.nn_k,
                                                                              r=self.nsw_r)
                    self.num_computations += num_searched_approx

                    if len(knn_and_score) == 0:
                        print("#NumSearched\tGraft\tapprox=%s\texact=%s\tnsw_edges=%s\ttree_nodes=%s\tspeedup=%s\terror="
                          % (num_searched_approx, num_searched_exact,self.nn_structure.num_edges,
                             p_idx * 2,num_searched_exact-num_searched_approx))
                    print('##########################################')
                    print()

                    if len(knn_and_score) > 0:

                        approximate_closest_node, approx_closest_score = knn_and_score[0][1].v, knn_and_score[0][0]
                        exact_closest_node, exact_closest_score = exact_knn_and_score[0][1].v, exact_knn_and_score[0][0]
                        print("#NumSearched\tGraft\tapprox=%s\texact=%s\tnsw_edges=%s\ttree_nodes=%s\tspeedup=%s\terror=%s"
                             % (num_searched_approx, num_searched_exact, self.nn_structure.num_edges,
                                p_idx * 2, num_searched_exact - num_searched_approx, np.abs(approx_closest_score-exact_closest_score)))
                        approx_eq_exact = approximate_closest_node == exact_closest_node
                        print("#KnnSearchRes\tGraft\t%s\tapprox=%s\tapprox_score=%s\texact=%s\texact_score=%s" %
                              (approx_eq_exact, approximate_closest_node.id, approx_closest_score,
                               exact_closest_node.id, exact_closest_score))

                        # allowed = allowable_graft(best)
                        allowed = True
                        if not allowed:
                            # self.graft_recorder.records.append(GraftMetaData(self, curr, best, False,False,False))
                            pass
                        else:
                            print(approx_closest_score)
                            print(curr.parent.my_score)
                            print(approximate_closest_node.parent.my_score)
                            print('Best %s BestTypes %s ' % (approximate_closest_node.id,type(approximate_closest_node)))
                            approx_says_perform_graft = approx_closest_score > curr.parent.my_score and approx_closest_score > approximate_closest_node.parent.my_score
                            exact_says_perform_graft = exact_closest_score > curr.parent.my_score and exact_closest_score > exact_closest_node.parent.my_score

                            print('(Approx.) Candidate Graft: (best: %s, score: %s) to (%s,par.score %s) from (%s,par.score %s)' %
                                  (approximate_closest_node.id,approx_closest_score,curr.id,curr.parent.my_score,approximate_closest_node.id,approximate_closest_node.parent.my_score))
                            print(
                                '(Exact.) Candidate Graft: (best: %s, score: %s) to (%s,par.score %s) from (%s,par.score %s)' %
                                (exact_closest_node.id, exact_closest_score, curr.id, curr.parent.my_score,
                                 exact_closest_node.id, exact_closest_node.parent.my_score))
                            # Perform Graft
                            print("#GraftSuggestions\tp_idx=%s\tg_idx=%s\tsame=%s\tapprox=%s\texact=%s" %
                                  (p_idx,graft_index,approx_says_perform_graft==exact_says_perform_graft,
                                   approx_says_perform_graft,exact_says_perform_graft))

                            if approx_says_perform_graft:

                                # Write the tree before the graft
                                if self.config.write_every_tree:
                                    Graphviz.write_tree(os.path.join(self.config.canopy_out,
                                                                     'tree_%s_before_graft_%s.gv' % (
                                                                     p_idx, graft_index)), self.root,
                                                        [approximate_closest_node.id, curr.id],[p_ment.id])
                                # self.graft_recorder.records.append(GraftMetaData(self, best, curr, True, True, False))
                                print("Performing graft: ")
                                best_pw,best_pw_n1,best_pw_n2 = self.best_pairwise(curr,approximate_closest_node)
                                print('best_pw = %s %s %s' % (best_pw_n1,best_pw_n2,best_pw))
                                new_ap_graft = self.hallucinate_merge(curr,approximate_closest_node,best_pw.data.numpy()[0])
                                new_graft_internal = curr.graft_to_me(approximate_closest_node, new_aproj=new_ap_graft, new_my_score=None) # We don't want a pw guy here.

                                print('Finished Graft')
                                print('updating.....')
                                # Update nodes
                                curr_update = new_graft_internal
                                while curr_update:
                                    e_score = self.score(curr_update).data.numpy()[0]
                                    if e_score != curr_update.my_score:
                                        print(
                                            'Updated my_score %s of curr my_score %s aproj_local[\'es\'] %s to be %s' % (
                                            curr.my_score,
                                            curr.as_ment.attributes.aproj_local[
                                                'es'] if 'es' in curr.as_ment.attributes.aproj_local else "None",
                                            curr.id, e_score))
                                        curr_update.my_score = e_score
                                    curr.as_ment.attributes.aproj_local['es'] = e_score
                                    curr_update = curr_update.parent
                                # Set the root of the tree after the graft: TODO: This could be sped up by being in the update loop
                                self.root = new_graft_internal.root()
                                print('##########################################')
                                print("############## KNN ADD %s #############" % new_graft_internal.id)
                                print('Adding new node to NSW')
                                self.nn_structure.insert(new_graft_internal)
                                print('##########################################')
                                # Write the tree after the graft
                                if self.config.write_every_tree:
                                    Graphviz.write_tree(os.path.join(self.config.canopy_out,
                                                                     'tree_%s_post_graft_%s.gv' % (p_idx, graft_index)),
                                                        self.root, [approximate_closest_node.id, curr.id],[p_ment.id])

                            else:
                                # self.graft_recorder.records.append(GraftMetaData(self, best, curr, False, True, False))
                                print('Chose not to graft.')

                    else:
                        # self.graft_recorder.records.append(GraftMetaData(self, None, curr, False, False, True))
                        print('No possible grafts for %s ' % curr.id)
                    graft_index += 1
                    curr = curr.parent
                    print("=============================================")
                    print()
        print()
        print('Done inserting p (%s,%s,%s) into tree ' % (p_ment.id, p[1], p[2]))
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print()
        self.observed_classes.add(p[1])
        if self.config.write_every_tree:
            if len(self.config.canopy_out) > 0:
                Graphviz.write_tree(os.path.join(self.config.canopy_out,
                                             'tree_%s.gv' % p_idx), self.root,[], [p_ment.id])
                if self.config.nn_structure == 'nsw':
                    GraphvizNSW.write_nsw(os.path.join(self.config.canopy_out, 'nsw_%s.gv' % p_idx), self.nn_structure)


    def build_dendrogram(self):
        idx = 0
        for p in self.dataset:
            print('[build dendrogram] Inserting pt number %s into tree' % idx)
            self.insert(p, idx)
            idx += 1
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


class BGerch(Gerch):
    def __init__(self, config, dataset, model):
        super(BGerch,self).__init__(config,dataset,model,perform_rotation=True,perform_graft=True)


    def insert(self, p,p_idx):
        """
        Incrementally add p to the tree.

        :param p - (MentObject,GroundTruth,Id)
        """

        p_ment = MentNode([p], aproj=p[0].attributes)
        p_ment.cluster_marker = True

        print()
        print()
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('Inserting p (%s,%s,%s) into tree ' % (p_ment.id,p[1],p[2]))
        if self.root is None:
            self.root = p_ment
            # self.mention_nn_structure.insert(p_ment)
            self.nn_structure.insert(p_ment)
        else:
            # Find k nearest neighbors

            if self.config.add_to_mention:
                offlimits = set([d.nsw_node for d in self.root.descendants() if d.point_counter > 1 if d.nsw_node])
            else:
                offlimits = set()

            print('##########################################')
            print("#### KNN SEARCH W/ New Point %s #############" % p_ment.id)

            knn_and_score,num_searched_approx = self.nn_structure.knn_and_score_offlimits(p_ment, offlimits, k=self.nn_k,
                                                                      r=self.nsw_r)
            self.num_computations += num_searched_approx

            approximate_closest_node, approx_closest_score = knn_and_score[0][1].v, knn_and_score[0][0]

            possible_nn_with_same_class = p[1] in self.observed_classes

            print("#KnnSearchRes\tNewMention\tapprox=%s\tapprox_score=%s" %
                  (approximate_closest_node.id,approx_closest_score))

            print("#NumSearched\tNewMention\tapprox=%s\tnsw_edges=%s"
                  "\ttree_nodes=%s\tscore=%s\tposs=%s"
                  % (
                    num_searched_approx,
                    self.nn_structure.num_edges,p_idx * 2 - 1,
                    approx_closest_score,possible_nn_with_same_class
            ))

            print('##########################################')
            print()
            print('##########################################')
            print("############## KNN ADD %s #############" % p_ment.id)

            # Add yourself to the knn structures
            self.nn_structure.insert(p_ment)
            print('##########################################')
            print()
            print('##########################################')
            print('############## Find Insert Stop ##########')

            # Find where to be added / rotate
            insert_node, new_ap, new_score = self.find_insert(knn_and_score[0][1].v,
                                                              p_ment,
                                                              knn_and_score[0][0])
            print('Splitting Down at %s with new scores %s' % (insert_node.id, new_score))

            # Add the point
            new_internal_node = insert_node.split_down(p_ment, new_ap, new_score)
            assert p_ment.root() == insert_node.root(), "p_ment.root() %s == insert_node.root() %s" % (
            p_ment.root(), insert_node.root())
            assert p_ment.lca(
                insert_node) == new_internal_node, "p_ment.lca(insert_node) %s == new_internal_node %s" % (
            p_ment.lca(insert_node), new_internal_node)

            print('Created new node %s ' % new_internal_node.id)

            # Update throughout the tree.
            if new_internal_node.parent:
                new_internal_node.parent.update_aps(p[0].attributes,self.model.sub_ent_model)

            # update all the entity scores
            curr = new_internal_node
            new_leaf_anc = p_ment._ancestors()
            while curr:
                self.update_for_new(curr,p_ment,new_leaf_anc,True)
                curr = curr.parent
            print('##########################################')
            print()
            print('##########################################')
            print("############## KNN ADD %s #############" % new_internal_node.id)

            # Add the newly created node to the NN structure
            self.nn_structure.insert(new_internal_node)
            print()
            print('##########################################')
            print()

            self.root = self.root.root()

            if self.perform_graft:
                graft_index = 0
                banish_index = 0

                curr = new_internal_node
                while curr.parent:
                    grafted = False
                    print()
                    print("=============================================")
                    print('Curr %s CurrType %s ' % (curr.id, type(curr)))

                    print('Finding Graft for %s ' % curr.id)

                    print('##########################################')
                    print("#### KNN SEARCH W/ Node %s #########" % curr.id)

                    offlimits = set([x.nsw_node for x in (curr.siblings() + curr.descendants() + curr._ancestors() + [curr])])

                    knn_and_score,num_searched_approx = self.nn_structure.knn_and_score_offlimits(curr, offlimits, k=self.nn_k,
                                                                              r=self.nsw_r)

                    self.num_computations += num_searched_approx

                    if len(knn_and_score) == 0:
                        print("#NumSearched\tGraft\tapprox=%s\texact=%s\tnsw_edges=%s\terror="
                          % (num_searched_approx,self.nn_structure.num_edges,
                             p_idx * 2))
                    print('##########################################')
                    print()

                    if len(knn_and_score) > 0:
                        approximate_closest_node, approx_closest_score = knn_and_score[0][1].v, knn_and_score[0][0]
                        approximate_closest_node_sib = approximate_closest_node.siblings()[0]
                        print("#NumSearched\tGraft\tapprox=%s\tnsw_edges=%s\ttree_nodes=%s\terror=%s"
                             % (num_searched_approx, self.nn_structure.num_edges,
                                p_idx * 2, np.abs(approx_closest_score)))
                        print("#KnnSearchRes\tGraft\tapprox=%s\tapprox_score=%s" %
                              (approximate_closest_node.id, approx_closest_score))

                        # allowed = allowable_graft(best)
                        allowed = True
                        if not allowed:
                            # self.graft_recorder.records.append(GraftMetaData(self, curr, best, False,False,False))
                            pass
                        else:
                            print(approx_closest_score)
                            print(curr.parent.my_score)
                            print(approximate_closest_node.parent.my_score)
                            print('Best %s BestTypes %s ' % (approximate_closest_node.id,type(approximate_closest_node)))
                            approx_says_perform_graft = approx_closest_score > curr.parent.my_score and approx_closest_score > approximate_closest_node.parent.my_score

                            print('(Approx.) Candidate Graft: (best: %s, score: %s) to (%s,par.score %s) from (%s,par.score %s)' %
                                  (approximate_closest_node.id,approx_closest_score,curr.id,curr.parent.my_score,approximate_closest_node.id,approximate_closest_node.parent.my_score))
                            # Perform Graft
                            print("#GraftSuggestions\tp_idx=%s\tg_idx=%s\tapprox=%s" %
                                  (p_idx,graft_index,approx_says_perform_graft))

                            if approx_says_perform_graft:
                                grafted = True
                                # Write the tree before the graft
                                if self.config.write_every_tree:
                                    Graphviz.write_tree(os.path.join(self.config.canopy_out,
                                                                     'tree_%s_before_graft_%s.gv' % (
                                                                     p_idx, graft_index)), self.root,
                                                        [approximate_closest_node.id, curr.id],[p_ment.id])
                                # self.graft_recorder.records.append(GraftMetaData(self, best, curr, True, True, False))
                                print("Performing graft: ")
                                best_pw,best_pw_n1,best_pw_n2 = self.best_pairwise(curr,approximate_closest_node)
                                print('best_pw = %s %s %s' % (best_pw_n1,best_pw_n2,best_pw))
                                new_ap_graft = self.hallucinate_merge(curr,approximate_closest_node,best_pw.data.numpy()[0])
                                new_graft_internal = curr.graft_to_me(approximate_closest_node, new_aproj=new_ap_graft, new_my_score=None) # We don't want a pw guy here.

                                print('Finished Graft')
                                print('updating.....')
                                # Update nodes
                                curr_update = new_graft_internal
                                while curr_update:
                                    e_score = self.score(curr_update).data.numpy()[0]
                                    if e_score != curr_update.my_score:
                                        print(
                                            'Updated my_score %s of curr my_score %s aproj_local[\'es\'] %s to be %s' % (
                                            curr.my_score,
                                            curr.as_ment.attributes.aproj_local[
                                                'es'] if 'es' in curr.as_ment.attributes.aproj_local else "None",
                                            curr.id, e_score))
                                        curr_update.my_score = e_score
                                    curr.as_ment.attributes.aproj_local['es'] = e_score
                                    curr_update = curr_update.parent
                                # Set the root of the tree after the graft: TODO: This could be sped up by being in the update loop
                                self.root = new_graft_internal.root()
                                print('##########################################')
                                print("############## KNN ADD %s #############" % new_graft_internal.id)
                                print('Adding new node to NSW')
                                self.nn_structure.insert(new_graft_internal)
                                print('Updating NSW for sibling of node that was grafted')
                                self.nn_structure.add_tree_edges(approximate_closest_node_sib)
                                print('##########################################')
                                # Write the tree after the graft
                                if self.config.write_every_tree:
                                    Graphviz.write_tree(os.path.join(self.config.canopy_out,
                                                                     'tree_%s_post_graft_%s.gv' % (p_idx, graft_index)),
                                                        self.root, [approximate_closest_node.id, curr.id],[p_ment.id])

                            else:
                                # self.graft_recorder.records.append(GraftMetaData(self, best, curr, False, True, False))
                                print('Chose not to graft.')

                    else:
                        # self.graft_recorder.records.append(GraftMetaData(self, None, curr, False, False, True))
                        print('No possible grafts for %s ' % curr.id)
                    graft_index += 1

                    if not grafted:
                        curr = curr.siblings()[0]

                        print()
                        print("=============================================")
                        print('Curr %s CurrType %s ' % (curr.id, type(curr)))

                        print('Finding Graft for %s ' % curr.id)

                        print('##########################################')
                        print("#### KNN SEARCH W/ Node %s #########" % curr.id)

                        offlimits = set(
                            [x.nsw_node for x in (curr.siblings() + curr.descendants() + curr._ancestors() + [curr])])

                        knn_and_score, num_searched_approx = self.nn_structure.knn_and_score_offlimits(curr, offlimits,
                                                                                                       k=self.nn_k,
                                                                                                       r=self.nsw_r)

                        self.num_computations += num_searched_approx

                        if len(knn_and_score) == 0:
                            print("#NumSearched\tGraft\tapprox=%s\texact=%s\tnsw_edges=%s\terror="
                                  % (num_searched_approx, self.nn_structure.num_edges,
                                     p_idx * 2))
                        print('##########################################')
                        print()

                        if len(knn_and_score) > 0:
                            approximate_closest_node, approx_closest_score = knn_and_score[0][1].v, knn_and_score[0][0]
                            approximate_closest_node_sib = approximate_closest_node.siblings()[0]
                            print("#NumSearched\tGraft\tapprox=%s\tnsw_edges=%s\ttree_nodes=%s\terror=%s"
                                  % (num_searched_approx, self.nn_structure.num_edges,
                                     p_idx * 2, np.abs(approx_closest_score)))
                            print("#KnnSearchRes\tGraft\tapprox=%s\tapprox_score=%s" %
                                  (approximate_closest_node.id, approx_closest_score))

                            # allowed = allowable_graft(best)
                            allowed = True
                            if not allowed:
                                # self.graft_recorder.records.append(GraftMetaData(self, curr, best, False,False,False))
                                pass
                            else:
                                print(approx_closest_score)
                                print(curr.parent.my_score)
                                print(approximate_closest_node.parent.my_score)
                                print(
                                    'Best %s BestTypes %s ' % (approximate_closest_node.id, type(approximate_closest_node)))
                                approx_says_perform_graft = approx_closest_score > curr.parent.my_score and approx_closest_score > approximate_closest_node.parent.my_score

                                print(
                                    '(Approx.) Candidate Graft: (best: %s, score: %s) to (%s,par.score %s) from (%s,par.score %s)' %
                                    (approximate_closest_node.id, approx_closest_score, curr.id, curr.parent.my_score,
                                     approximate_closest_node.id, approximate_closest_node.parent.my_score))
                                # Perform Graft
                                print("#GraftSuggestions\tp_idx=%s\tg_idx=%s\tapprox=%s" %
                                      (p_idx, graft_index, approx_says_perform_graft))

                                if approx_says_perform_graft:

                                    # Write the tree before the graft
                                    if self.config.write_every_tree:
                                        Graphviz.write_tree(os.path.join(self.config.canopy_out,
                                                                         'tree_%s_before_banish_%s.gv' % (
                                                                             p_idx, graft_index)), self.root,
                                                            [approximate_closest_node.id, curr.id], [p_ment.id])
                                    # self.graft_recorder.records.append(GraftMetaData(self, best, curr, True, True, False))
                                    print("Performing graft: ")
                                    best_pw, best_pw_n1, best_pw_n2 = self.best_pairwise(curr, approximate_closest_node)
                                    print('best_pw = %s %s %s' % (best_pw_n1, best_pw_n2, best_pw))
                                    new_ap_graft = self.hallucinate_merge(curr, approximate_closest_node,
                                                                          best_pw.data.numpy()[0])
                                    new_graft_internal = approximate_closest_node.graft_to_me(curr, new_aproj=new_ap_graft,
                                                                          new_my_score=None)  # We don't want a pw guy here.

                                    print('Finished Graft')
                                    print('updating.....')
                                    # Update nodes
                                    curr_update = new_graft_internal
                                    while curr_update:
                                        e_score = self.score(curr_update).data.numpy()[0]
                                        if e_score != curr_update.my_score:
                                            print(
                                                'Updated my_score %s of curr my_score %s aproj_local[\'es\'] %s to be %s' % (
                                                    curr.my_score,
                                                    curr.as_ment.attributes.aproj_local[
                                                        'es'] if 'es' in curr.as_ment.attributes.aproj_local else "None",
                                                    curr.id, e_score))
                                            curr_update.my_score = e_score
                                        curr.as_ment.attributes.aproj_local['es'] = e_score
                                        curr_update = curr_update.parent
                                    # Set the root of the tree after the graft: TODO: This could be sped up by being in the update loop
                                    self.root = new_graft_internal.root()
                                    print('##########################################')
                                    print("############## KNN ADD %s #############" % new_graft_internal.id)
                                    print('Adding new node to NSW')
                                    self.nn_structure.insert(new_graft_internal)
                                    print('Updating NSW for sibling of node that was grafted')
                                    self.nn_structure.add_tree_edges(approximate_closest_node_sib)
                                    print('##########################################')
                                    # Write the tree after the graft
                                    if self.config.write_every_tree:
                                        Graphviz.write_tree(os.path.join(self.config.canopy_out,
                                                                         'tree_%s_post_banish_%s.gv' % (p_idx, graft_index)),
                                                            self.root, [approximate_closest_node.id, curr.id], [p_ment.id])

                                else:
                                    # self.graft_recorder.records.append(GraftMetaData(self, best, curr, False, True, False))
                                    print('Chose not to banish.')

                        else:
                            # self.graft_recorder.records.append(GraftMetaData(self, None, curr, False, False, True))
                            print('No possible banish for %s ' % curr.id)
                        banish_index += 1
                    curr = curr.parent
                    print("=============================================")
                    print()
        print()
        print('Done inserting p (%s,%s,%s) into tree ' % (p_ment.id, p[1], p[2]))
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print()
        self.observed_classes.add(p[1])
        if self.config.write_every_tree:
            if len(self.config.canopy_out) > 0:
                Graphviz.write_tree(os.path.join(self.config.canopy_out,
                                             'tree_%s.gv' % p_idx), self.root,[], [p_ment.id])
                if self.config.nn_structure == 'nsw':
                    GraphvizNSW.write_nsw(os.path.join(self.config.canopy_out, 'nsw_%s.gv' % p_idx), self.nn_structure)

