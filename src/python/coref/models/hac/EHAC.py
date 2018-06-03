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

"""A Wrapper for model-based hierarchical agglomerative clustering."""
import torch

import numpy as np

from scipy.special import expit

from coref.models.core.AttributeProjection import AttributeProjection
from coref.models.core.MentNode import MentNode


class EHAC(object):
    """Wrapper around Entity-level Hierarchical Agglomerative Clustering.

    In the entity-level variant, we have a model that can score a group of
    nodes rather than pairs.  The algorithm is similar to something like HAC
    with Ward's linkage function except for that this "linkage" is a learned
    model.

    One assumption in this implementation is that an a group of nodes (called
    an entity here) knows about the best edge that connects its two children.
    Because of this, there is special logic in here that looks for the best
    pair of nodes to cross a cut.  We also assume that there is a trained
    pairwise model.
    """
    def __init__(self, config, dataset, model):
        """Init.

        Args:
            config - a config object.
            dataset - a list of data items (mention, label, id).
            model - a model that has a pw_score_mat(mat) function.
        """
        self.config = config
        self.pts = dataset
        self.model = model
        self.num_computations = 0
        self.roots = []         # The current roots.
        for d in dataset:
            self.roots.append(MentNode([d], aproj=d[0].attributes))
            self.roots[-1].cluster_marker = True

        # Get all pairwise scores first.
        num_pairs = int(len(dataset) * (len(dataset) - 1) / 2.0)
        fvs = torch.zeros(num_pairs, model.pw_concat_dim)
        curr_row = 0
        for i in range(len(self.roots)):
            for j in range(i + 1, len(self.roots)):
                fv = model.pw_extract_features(
                    self.roots[i].pts[0][0].attributes,
                    self.roots[j].pts[0][0].attributes).data
                fvs[curr_row, :] = fv
                curr_row += 1


        all_scores = model.pw_score_mat(fvs)

        self.mergers = []      # All pairwise scores.
        self.sorted_mergers = []   # All merges and entity scores.
        self.pair_to_pw = {}
        curr_row = 0
        for i in range(len(self.roots)):
            for j in range(i + 1, len(self.roots)):
                n1, n2 = self.roots[i], self.roots[j]
                pw_score = all_scores[curr_row].data.cpu().numpy()[0]
                self.pair_to_pw[(n1, n2)] = pw_score
                ap = self.hallucinate_merge(n1, n2, pw_score)
                e_score = self.model.e_score(ap)
                np_e_score = e_score.data.cpu().numpy()[0]
                ap.aproj_local['es'] = np_e_score
                self.sorted_mergers.append((n1, n2, np_e_score, ap, e_score))
                curr_row += 1

        self.sorted_mergers.sort(key=lambda x: -x[2])


    def hallucinate_merge(self,n1, n2, pw_score):
        """Compute the attribute projection of mergingin n1 and n2.

        Compute the attribute projection of merging n1 and n2, assuming that
        pw_score is the best pairwise score of merging them.

        Args:
             n1 - a MentNode.
             n2 - another MentNode.
             pw_score - a double representing the best pairwise score btw n1&n2.

        Returns:
            The hallucinated attribute projection.
        """
        self.num_computations += 1
        ap = AttributeProjection()
        ap.update(n1.as_ment.attributes,self.model.sub_ent_model)
        ap.update(n2.as_ment.attributes,self.model.sub_ent_model)
        num_ms = n1.point_counter + n2.point_counter
        if 'tes' in ap.aproj_sum:
            ap.aproj_sum['tea'] = ap['tes'] / num_ms
        new_ap = AttributeProjection()
        new_ap.aproj_sum['pw'] = pw_score
        new_ap.aproj_bb['pw_bb'] = (pw_score, pw_score)
        ap.update(new_ap)
        ap.aproj_local['my_pw'] = pw_score
        ap.aproj_local['new_edges'] = n1.point_counter * n2.point_counter

        left_child_entity_score = 1.0
        right_child_entity_score = 1.0

        if 'es' in n1.as_ment.attributes.aproj_local:
            left_child_entity_score = n1.as_ment.attributes.aproj_local['es']
            if self.config.expit_e_score:
                left_child_entity_score = expit(left_child_entity_score)
        if 'es' in n2.as_ment.attributes.aproj_local:
            right_child_entity_score = n2.as_ment.attributes.aproj_local['es']
            if self.config.expit_e_score:
                right_child_entity_score = expit(right_child_entity_score)

        if left_child_entity_score >= right_child_entity_score:
            ap.aproj_local['child_e_max'] = left_child_entity_score
            ap.aproj_local['child_e_min'] = right_child_entity_score
        else:
            ap.aproj_local['child_e_max'] = right_child_entity_score
            ap.aproj_local['child_e_min'] = left_child_entity_score

        # print('hallucinate_merge(%s,%s)' %(n1.id,n2.id))
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

    def clean_mergers(self, entity):
        """Remove invalid mergers from sorted_mergers.

        Remove all mergers from sorted mergers that both members of the merge
        are part of sorted_mergers.

        Args:
            entity - a set of MentNodes.

        Return:
            Nothing -- filters self.sorted_mergers.
        """
        self.sorted_mergers = [
            (n1, n2, np_e_score, ap, e_score)
            for (n1, n2, np_e_score, ap, e_score) in self.sorted_mergers
            if n1 not in entity and n2 not in entity]

    def best_merge_btw(self, n1, n2):
        """Compute the best pairwise score between a leaf of n1 and n2.

        Args:
            n1 - the root of a tree of MentNodes.
            n2 - the root of a tree of MentNodes.

        Returns:
            The highest scoring pair where one element of the pair is from n1
            and the other is from n2. Also return the two leaves that yield
            the high score.
        """
        best = None
        for l1 in n1.leaves():
            for l2 in n2.leaves():
                if (l1, l2) in self.pair_to_pw:
                    pws = self.pair_to_pw[(l1, l2)]
                else:
                    pws = self.pair_to_pw[(l2, l1)]
                if best is None or pws > best[0]:
                    best = (pws, l1, l2)
        return best

    def add_scores_with_entity(self, new_entity):
        """Add new mergers wrt to the new entity.

        Loop through self.roots and compute the score of merging new_entity with
        each root. Add the merger to sorted mergers and then resort the list.

        Args:
            new_entity - a MentNode.

        Returns:
            Nothing - modifies the list of sorted mergers.
        """
        for r in self.roots:
            if r != new_entity:
                assert r.root() == r
                assert new_entity.root() == new_entity
                pw_score, _, _ = self.best_merge_btw(new_entity, r)
                ap = self.hallucinate_merge(new_entity, r, pw_score)
                e_score = self.model.e_score(ap)
                np_e_score = e_score.data.cpu().numpy()[0]
                ap.aproj_local['es'] = np_e_score
                self.sorted_mergers.append((new_entity, r, np_e_score, ap,
                                            e_score))
        self.sorted_mergers.sort(key=lambda x: -x[2])

    def next_agglom(self):
        """Return merger (all mergers should always be valid)."""
        if self.sorted_mergers:
            n1, n2, np_e_score, ap, e_score = self.sorted_mergers.pop(0)
            assert n1.root() != n2.root()
            assert n1 in self.roots
            assert n2 in self.roots
            assert n1.root() == n1
            assert n2.root() == n2
            return n1, n2, np_e_score, ap, e_score
        return None

    def merge(self, n1, n2, ap):
        """Merge n1 and n2.

        In merging n1 and n2, also perform the following side effects: 1) remove
        n1 and n2 from self.roots, 2) add merged to self.roots.

        Args:
            n1 - the root of a tree of MentNodes.
            n2 - the root of a tree of MentNodes.
            ap - the resultant attribute projection

        Returns:
            The result of the merge.
        """
        assert n1 in self.roots
        assert n2 in self.roots
        assert n1.root() == n1
        assert n2.root() == n2
        assert n1 != n2
        merged = MentNode(n1.pts + n2.pts, aproj=ap)
        # NOTE: 4/15/2018 - NBGM Changed entity score here to not have the sigmoid.
        predicted_score = ap.aproj_local['es']
        # assert predicted_score <= 1.0
        # assert predicted_score >= 0.0
        print('Merging %s and %s with score %s' % (n1.id,n2.id,predicted_score))
        merged.my_score = predicted_score
        self.roots.append(merged)
        merged.children.append(n1)
        merged.children.append(n2)
        n1.parent = merged
        n2.parent = merged
        self.roots.remove(n1)
        self.roots.remove(n2)
        # print('attributes of merged node: merged.attributes =  %s' % merged.as_ment.attributes)
        return merged

    def build_dendrogram(self):
        """Run exact greedy inference with the entity model.

        At each iteration, merge the two tree roots that results in the highest
        scoring merge according to the model.

        Args:
            None.

        Return:
            A pointer to the root of the tree.
        """
        while len(self.roots) > 1:
            res = self.next_agglom()
            if res:
                n1, n2, np_e_score, ap, e_score = res
                assert n1.root() == n1
                assert n2.root() == n2
                merged = self.merge(n1, n2, ap)
                merged.cluster_marker = True
                n1.cluster_marker = True
                n2.cluster_marker = True

                # clean the list of sorted mergers.
                self.clean_mergers({merged.children[0], merged.children[1]})

                # update the scores in sorted mergers and resort.
                self.add_scores_with_entity(merged)

        assert len(self.roots) == 1
        return self.roots[0]
