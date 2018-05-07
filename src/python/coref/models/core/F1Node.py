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

from collections import defaultdict
from heapq import heappush, heappop

import numpy as np

from coref.models.core.Node import Node


class F1Node(Node):
    """A node that can compute it's best F1 partition."""
    def __init__(self):
        super().__init__()
        self.local_tp = 0
        self.local_fp = 0
        self.best_tp = 0
        self.best_fp = 0
        self.best_partition = []
        # NOTE: The type of myscore MUST be a numpy float not a torch float.
        self.my_score = -np.inf
        self.prev_best_score = -np.inf
        self.pred_partition = []
        self.f1_ready = False
        self.curr_height = None

    def annotate_heights(self):
        root = self.root()
        root.curr_height = root.height()
        frontier = [root]
        while frontier:
            x = frontier.pop(0)
            assert x.curr_height is not None
            assert x.curr_height >= 0
            if x.children:
                x.children[0].curr_height = x.curr_height - 1
                x.children[1].curr_height = x.curr_height - 1
                frontier.append(x.children[0])
                frontier.append(x.children[1])

    def tp_fp(self):
        """Compute the number of tp and fp of combining my children."""
        assert(len(self.children) == 2)
        same = 0
        diff = 0
        child_0_pts = [l.pts[0] for l in self.children[0].leaves()] if not self.children[0].pts else self.children[0].pts
        child_1_pts = [l.pts[0] for l in self.children[1].leaves()] if not self.children[1].pts else self.children[1].pts
        for pt1 in child_0_pts:
            for pt2 in child_1_pts:
                if pt1[1] == pt2[1] and pt1[1] != "None" and pt2[1] != "None":
                    same += 1
                elif pt1[1] != "None" and pt2[1] != "None":
                    diff += 1
        return same, diff

    def compute_gt(self):
        """Compute the ground truth number of true positives."""
        label_to_num = defaultdict(int)
        for l in self.leaves():
            if l.pts[0][1] != "None":
                label_to_num[l.pts[0][1]] += 1
        num_gt = 0
        for v in label_to_num.values():
            if v > 1:
                num_gt += (v * (v-1) * 0.5)
        return num_gt

    def update(self, gt):
        """Update myself with tp and fp and the best partition below me."""
        tp, fp = self.tp_fp()
        new_tp = self.children[0].local_tp + self.children[1].local_tp + tp
        new_fp = self.children[0].local_fp + self.children[1].local_fp + fp
        self.local_tp = new_tp
        self.local_fp = new_fp

        prev_tp = self.children[0].best_tp + self.children[1].best_tp
        prev_fp = self.children[0].best_fp + self.children[1].best_fp
        local_f1 = F1Node.f1(self.local_tp, self.local_fp, gt)
        prev_f1 = F1Node.f1(prev_tp, prev_fp, gt)
        if local_f1 > prev_f1:
            self.best_partition = [self]
            self.best_tp = self.local_tp
            self.best_fp = self.local_fp
        else:
            self.best_partition = self.children[0].best_partition + \
                                  self.children[1].best_partition
            self.best_tp = prev_tp
            self.best_fp = prev_fp
        self.f1_ready = True

    @staticmethod
    def f1(tp, fp, gt):
        """Compute the f1 from tps, fps and gt."""
        return float(2.0 * tp / (gt + tp + fp)) if (gt + tp + fp) > 0.0 else 0.0

    def _make_f1_ready(self):
        """Compute the tp and fps for each node in the tree."""
        assert(self.root() == self)
        gt = self.compute_gt()
        leaves = [l for l in self.leaves()]
        frontier = []
        for l in leaves:
            l.best_partition = [l]
            if l.parent:
                t = (l.parent.height(), l.parent)
                if t not in frontier:
                    heappush(frontier, t)

        while frontier:
            (_, n) = heappop(frontier)
            n.update(gt)
            if n.parent:
                t = (n.parent.height(), n.parent)
                if t not in frontier:
                    heappush(frontier, t)

    def partition_best(self):
        """Return a list of nodes representing the partition with highest f1."""
        if self.f1_ready:
            return self.best_partition
        else:
            self._make_f1_ready()
            return self.best_partition

    def partition_cluster_marker(self):
        """Compute the partition according to the marked nodes in the tree."""
        if not self.f1_ready:
            self._make_f1_ready()
        pred_partition = []
        for l in self.leaves():
            in_entity = True
            prev_node = l
            for a in l._ancestors():
                if in_entity is False and a.cluster_marker:
                    raise Exception('Entity markers are screwed up.')
                elif in_entity and a.cluster_marker is False:
                    in_entity = False
                    if prev_node not in pred_partition:
                        pred_partition.append(prev_node)
                prev_node = a

            if prev_node.cluster_marker is True:
                assert prev_node == self.root()
                pred_partition.append(prev_node)
                break
        return pred_partition

    def partition_threshold(self, t):
        """Compute the partition according to a threshold."""
        if not self.f1_ready:
            self._make_f1_ready()
        pred_partition = []
        frontier = []
        non_entities = set()
        self.annotate_heights()
        for l in self.leaves():
            heappush(frontier, (l.curr_height, l))
        while frontier:
            _, l = heappop(frontier)
            if l in pred_partition:
                continue
            elif l.parent and l.parent in non_entities:
                pred_partition.append(l)
            elif l.parent and l.parent.my_score >= t:
                if (l.parent.curr_height, l.parent) not in frontier:
                    heappush(frontier, (l.parent.curr_height, l.parent))
            elif l.parent:
                pred_partition.append(l)
                for a in l._ancestors():
                    non_entities.add(a)
            else:
                pred_partition.append(l)  # l is root
        return pred_partition

    def find_partition_threshold_heuristic(self):
        """Find the best partition threshold using a heuristic.

        Visit each node in the tree and create a list of all scores
        (node.my_score). Sort the list. Then, select k evenly spaced-out scores
        from the list, where k is log_2(len(list)).  Test each score as the
        partition threshold and remember the index of the best one. Compute the
        indices of the scores immediately before and after the best score in the
        original list.  Subselect from the original list all the elements
        between these two indices (before and after the best score) and repeat.
        The computation ends when the subselected list has a single element.
        Return the best threshold found throughout this process.

        The intuition here is as follows. If we compute the F1 of various
        threshold in increasing order of threshold, we find that the F1s,
        are somewhat random locally (small perturbations) but there is neither
        a global increasing or decreasing trend.  The average in a sliding
        window seems to change smoothly but not consistently increase or
        decrease. Therefore, we probe evenly spaced out scores, find the best,
        and explore the corresponding local neighborhood better.

        Args:
            None

        Returns:
            Best threshold, best pre, rec and f1.
        """
        assert self == self.root()

        # Visit all nodes and collect scores.
        frontier = [self]
        all_scores = set()
        while frontier:
            x = frontier.pop(0)
            if x.children:
                frontier.append(x.children[0])
                frontier.append(x.children[1])
            all_scores.add(x.my_score)

        # compute F1 using min and max threshold, store the best.
        min_pre, min_rec, min_f1 = self.f1_threshold(min(all_scores))
        max_pre, max_rec, max_f1 = self.f1_threshold(max(all_scores))
        if min_f1 > max_f1:
            best_f1_and_thresh = (min(all_scores), min_pre, min_rec, min_f1)
        else:
            best_f1_and_thresh = (max(all_scores), max_pre, max_rec, max_f1)

        # search for best by subselected logN remaining elements to scan.
        sorted_scores = sorted(all_scores)
        while len(sorted_scores) > 1:
            log_num_scores = np.log2(len(sorted_scores))
            interval = int(np.floor(len(sorted_scores) /
                                    log_num_scores))
            best_i = 0
            for i in range(0, len(sorted_scores), interval):
                score = sorted_scores[i]
                pre, rec, f1 = self.f1_threshold(score)
                # DEBUG
                predicted = self.partition_threshold(score)
                assert sum([len(e.pts) for e in predicted]) == len(self.pts)
                # print(pre, rec, f1, score, len(predicted))
                if f1 >= best_f1_and_thresh[3]:
                    best_f1_and_thresh = (score, pre, rec, f1)
                    best_i = i
            minus_1 = max(1, best_i - interval)
            plus_1 = min(len(sorted_scores) - 1, best_i + interval)
            sorted_scores = sorted_scores[minus_1: plus_1]
        return best_f1_and_thresh

    def f1_best(self):
        """Return the pre, rec and f1 of the best partition."""
        if not self.f1_ready:
            self._make_f1_ready()
        gt = self.compute_gt()
        tp_ub = 0
        fp_ub = 0
        for e in self.best_partition:
            tp_ub += e.local_tp
            fp_ub += e.local_fp
        f1 = F1Node.f1(tp_ub, fp_ub, gt)
        if tp_ub + fp_ub == 0.0:
            return 0.0, tp_ub / gt if gt > 0 else 0.0, f1
        else:
            return tp_ub / (tp_ub + fp_ub) if (tp_ub + fp_ub) > 0 else 0.0, tp_ub / gt if gt > 0 else 0.0, f1

    def tp_fp_gt_best(self):
        """Return the tp, fp and gt of the best partition."""
        if not self.f1_ready:
            self._make_f1_ready()
        gt = self.compute_gt()
        tp_ub = 0
        fp_ub = 0
        for e in self.best_partition:
            tp_ub += e.local_tp
            fp_ub += e.local_fp
        return tp_ub,fp_ub,gt

    def tp_fp_gt_cluster_marker(self):
        """Return the tp, fp and get of the predicted partition"""
        if not self.f1_ready:
            self._make_f1_ready()
        tp = 0
        fp = 0
        gt = self.compute_gt()
        for e in self.partition_cluster_marker():
            tp += e.local_tp
            fp += e.local_fp
        return tp,fp,gt

    def f1_cluster_marker(self):
        """Return the pre, rec and f1 of the ."""
        if not self.f1_ready:
            self._make_f1_ready()
        tp = 0
        fp = 0
        gt = self.compute_gt()
        for e in self.partition_cluster_marker():
            tp += e.local_tp
            fp += e.local_fp
        f1 = F1Node.f1(tp, fp, gt)
        return tp / (tp + fp) if tp + fp > 0 else 0.0, tp / gt if gt > 0 else 0, f1

    def f1_threshold(self, t):
        """Return the pre, rec and f1 of the ."""
        if not self.f1_ready:
            self._make_f1_ready()
        tp = 0
        fp = 0
        gt = self.compute_gt()
        for e in self.partition_threshold(t):
            tp += e.local_tp
            fp += e.local_fp
        f1 = F1Node.f1(tp, fp, gt)
        return tp / (tp + fp) if tp + fp > 0 else 0.0, tp / gt if gt > 0 else 0, f1

    def tp_fp_gt_threshold(self, t):
        """Return the pre, rec and f1 of the ."""
        if not self.f1_ready:
            self._make_f1_ready()
        tp = 0
        fp = 0
        gt = self.compute_gt()
        for e in self.partition_threshold(t):
            tp += e.local_tp
            fp += e.local_fp
        return tp, fp, gt

    # TODO(AK): Needs refactor badly.
    def predict_coref(self, model, feat_fun):
        """Use a model to predict the best partition."""
        frontier = []
        for l in self.leaves():
            l.my_score = model.score(feat_fun(l)).cpu().data.numpy()
            l.prev_best_score = l.my_score
            l.pred_partition = [l]
            heappush(frontier, (l.parent.height(), l.parent))

        while frontier:
            (_, target) = heappop(frontier)
            target.my_score = model.score(feat_fun(target)).cpu().data.numpy()
            best_prev_score = target.children[0].prev_best_score + \
                              target.children[1].prev_best_score
            if target.my_score > best_prev_score:
                target.prev_best_score = target.my_score
                target.pred_partition = [target]
            else:
                target.prev_best_score = best_prev_score
                target.pred_partition = target.children[0].pred_partition + \
                    target.children[1].pred_partition
        return self.root().pred_partition
