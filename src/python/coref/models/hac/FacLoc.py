
from coref.models.hac.Gerch import Gerch
from coref.models.core.MentNode import MentNode
from coref.models.core.AttributeProjection import AttributeProjection

class FacLoc(Gerch):
    """
    Store the clusters as the roots of the tree.
    Every time you see a new datapoint, add it to an existing cluster.
    if the resulting score of adding it to that cluster (from the root of the cluster)
    would be worse than before, create a new cluster for the datapoint.
    
    """


    def __init__(self, config, dataset, model):
        super(FacLoc,self).__init__(config,dataset,model,perform_rotation=False,perform_graft=False)
        self.clusters = []

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


