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

import datetime
import os
import random
import sys

from scipy.special import expit

import torch.optim
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss,MSELoss

from coref.train.pw.TrainAuthorBatcher import TrainAuthorModelBatcher
from coref.models.core.Ment import Ment,MentLoader
from coref.models.core.F1Node import F1Node
from coref.models.hac.EHAC import EHAC
from coref.router.Utils import save_dict_to_json, sofl
from coref.util.IO import mkdir_p

from coref.models.hac import new_clustering_scheme


class MergePreTrainer(object):
    """The MergePreTrainer trains a model to agglomerate.

    The MergePreTrainer is able to train an agglomeration model. In particular,
    the model being trained can take any group of data points (i.e., mentions)
    and returns a score for how cohesive the group is. The MergePreTrainer
    trains the score function to estimate the precision of the input group of
    data points.
    """
    def __init__(self, config, vocab, model):
        self.model = model
        self.trainer = 'MergePreTrainer'
        self.config = config
        self.vocab = vocab
        self.bceloss = BCEWithLogitsLoss()
        self.buffer_size = config.buffer_size
        self.obj = 'pre'
        self.e_opt = torch.optim.Adam(self.model.sub_ent_model.parameters(),
                                      lr=config.e_lr,
                                      weight_decay=config.l2penalty)
        self.pw_opt = torch.optim.Adam(self.model.pw_model.parameters(),
                                       lr=config.pw_lr,
                                       weight_decay=config.l2penalty)

        self.one_label = Variable(torch.ones(1))
        self.zero_label = Variable(torch.zeros(1))

    # TODO(AK): This method should probably be moved to a file containing other
    # TODO(AK): pseudo-objectives.
    @staticmethod
    def _pre_of_new(n1, n2):
        """Returns the precision of the node created by merging n1 and n2.

        Args:
            n1 - a Node (see models.core)
            n2 - a Node (see models.core)

        Returns:
            The precision of Union(n1, n2) which is # True positive / (# False
            positive + # True positives).
        """
        n1_class_counts = n1.class_counts()
        n2_class_counts = n2.class_counts()
        tp, fp = 0.0, 0.0
        n1_total = sum([count for c, count in n1_class_counts.items()
                        if c != 'None'])
        n2_total = sum([count for c, count in n2_class_counts.items()
                        if c != 'None'])

        for c1, count1 in n1_class_counts.items():
            if c1 != 'None':
                tp += (count1 * (count1 - 1.0)) / 2.0
                fp += count1 * (n1_total - count1) / 2.0
        for c2, count2 in n2_class_counts.items():
            if c2 != 'None':
                tp += (count2 * (count2 - 1.0)) / 2.0
                fp += count2 * (n2_total - count2) / 2.0

        for c1, count1 in n1_class_counts.items():
            for c2, count2 in n2_class_counts.items():
                if c1 != 'None' and c1 == c2:
                    tp += count1 * count2
                elif c1 != 'None' and c2 != 'None' and c1 != c2:
                    fp += count1 * count2

        # TODO(AK): probably want more assertions.
        assert tp + fp > 0.0
        assert tp / (tp + fp) <= 1.0
        assert tp / (tp + fp) >= 0.0

        return tp / (tp + fp)

    def write_training_data(self, ms, outfile):
        """Read in a collections of mentions and create pw training data."""
        batcher = TrainAuthorModelBatcher(self.config)
        loader = MentLoader()
        for m1, m2, lbl in ms:
            m1obj = loader.load_ment(m1,self.model)[0]
            m2obj = loader.load_ment(m2,self.model)[0]
            lbl = float(lbl)
            fv = self.model.pw_extract_features(
                m1obj.attributes, m2obj.attributes).cpu().data.numpy()
            batcher.add_example(fv, lbl, (m1obj.mid, m2obj.mid))
        batcher.save(outfile)

    def train(self, batcher, outdir, dev_batcher=None):
        """Train the PW model then train the entity model to refine PW model.

        Train the PW model to minimize binary classification loss. Then train
        the entity model to score merges (see the train_e method).

        Args:
             batcher - batcher for pairwise training
             outdir - where to write the trained models, etc.

        Returns:
            None
        """
        diagnostics = {}

        train_canopies = []
        if self.config.train_files.lower() != 'none' and self.config.train_files.lower() != 'empty':
            with open(self.config.train_files, 'r') as fin:
                for line in fin:
                    train_canopies.append(line.strip())

        print('train_canopies')
        print(train_canopies)

        random.shuffle(train_canopies)
        train_data_dir = os.path.join('data', self.config.dataset_name,
                                      'eval', 'train', 'canopy')
        iter_count = 0
        iters_per_file = max(1, int(self.config.refine_itrs / len(train_canopies))) if len(train_canopies) > 0 else 0

        print("entity iter_per_file %s" % iters_per_file)

        # Turn off learning for the entity model features.
        for param in self.model.sub_ent_model.parameters():
            param.requires_grad = False

        # Train the pairwise model.
        print('[TRAINING PAIRWISE]')
        self.train_pw(batcher, outdir, None)
        sys.stdout.flush()

        # Load the best PW model.
        self.model = torch.load(self.config.best_model)

        # Turn off learning for the pairwise model
        for param in self.model.pw_model.parameters():
            param.requires_grad = False
        # Turn on learning for the entity model
        for param in self.model.sub_ent_model.parameters():
            param.requires_grad = True

        # Set the e_optimizer up (do it here because we've loaded the model).
        self.e_opt = torch.optim.Adam(self.model.sub_ent_model.parameters(),
                                      lr=self.config.e_lr,
                                      weight_decay=self.config.l2penalty)

        # Train the entity model.
        print('[TRAINING ENTITY]')
        best_f1 = 0.0

        for idx, train_canopy in enumerate(train_canopies):
            train_file = os.path.join(train_data_dir, train_canopy, 'ments.json')
            ms = []
            for m in Ment.load_ments(train_file, self.model):
                if not self.config.only_use_labeled_dev or (
                        m[1] is not None and m[1] != "None"):
                    ms.append(m)
            best_f1,iter_count = self.train_e(ms, diagnostics,
                                              iter_start_count=iter_count,
                                              num_iterations=iters_per_file,
                                              prev_best_f1=best_f1)

    def train_pw(self, batcher, outdir, dev_batcher=None):
        """Trains the pairwise component of this model.

        The PW component of the model takes to instances and predicts whether
        they are in the same class or not. It is trained to minimize the
        BCEWithLogitsLoss (logistic loss).

        Args:
            batcher - a batching object that can load up training pairs.
            outdir - where to write out the trained model.

        Returns:
              Nothing - trains the pairwise model and writes to a file.
        """
        diagnostics = {}
        best_f1 = 0.0
        output_path = os.path.join(outdir, 'trained-models')
        os.makedirs(output_path)
        count = 1
        total_loss = 0.0
        for batch in batcher.get_next_batch():
            if count % self.config.eval_every == 0:
                # Find the best partition threshold.
                # We now assume that there are multiple canopies in the dev set.
                f1_ub, f1, best_t = self.dev_eval_multiple(count, diagnostics,
                                                           output_path)
                sofl('[F1 UPPER BOUND] %s' % f1_ub)
                sofl('[F1 THRESHOLD]   %s\t%s' % (f1, best_t))
                print('[PW WEIGHTS]')
                print(self.model.pw_output_layer.weight)
                print('[PW BIAS]')
                print(self.model.pw_output_layer.bias)
                print('[E WEIGHTS]')
                print(self.model.e_output_layer.weight)
                print('[E BIAS]')
                print(self.model.e_output_layer.bias)
                if f1 > best_f1:
                    best_f1 = f1
                    new_best_model = os.path.join(output_path,
                                                  'pw_model.%d.torch' % count)
                    torch.save(self.model, new_best_model)
                    self.config.trained_model = new_best_model
                    self.config.partition_threshold = float(best_t) # convert numpy
                    self.config.best_model = new_best_model
                    self.config.save_config(self.config.experiment_out_dir)

            if count % 100 == 0:
                print('Training on batch %d' % count)
                print('PW WEIGHTS')
                print(self.model.pw_output_layer.weight)
                print('PW BIAS')
                print(self.model.pw_output_layer.bias)
                print('E WEIGHTS')
                print(self.model.e_output_layer.weight)
                print('E BIAS')
                print(self.model.e_output_layer.bias)
            self.pw_opt.zero_grad()
            l = self.compute_loss(batch)
            total_loss += l.data.numpy()[0]
            if count % 100 == 0 and count > 0:
                sofl('[PW Model] Batches %s | Examples %s | This Batch Loss %s | Average Loss %s' % (count,count*self.config.batch_size,l.data.numpy()[0],total_loss/count))
            l.backward()
            self.pw_opt.step()

            count += 1
            sys.stdout.flush()
            if count > self.config.pw_num_minibatches:
                sofl('Hit limit of number of pw minibatches %s' % count)
                break

        f1_ub, f1, best_t = self.dev_eval_multiple(count, diagnostics,
                                                   output_path)
        if f1 > best_f1:
            new_best_model = os.path.join(output_path,
                                          'pw_model.%d.torch' % count)
            torch.save(self.model, new_best_model)
            self.config.partition_threshold = float(best_t)  # convert from numpy.
            self.config.best_model = new_best_model
            self.config.save_config(self.config.experiment_out_dir)


    def train_e(self, data, diagnostics, iter_start_count=1,
                num_iterations=None, prev_best_f1=0.0):
        """Train the agglomeration function (i.e., the entity model).

        To train, sample a batch of mentions from the train set and call the
        train_pwe_episode function. Run the dev eval every K iterations
        (defined in the config). Write all of the trained models to the output
        directory and remember the best model (by writing it to the config).

        Args:
            data - the dataset of points (triples: (data, label, id)).
            diagnostics - a dictionary of stats.
            iter_start_count - the iteration to start training from.
            num_iterations - the number of iterations to train for.
            prev_best_f1 - the best f1 we've achieved previously.

        Returns:
            The best f1 on the dev set (achieved by a trained model).
        """
        print("train_e // iter_start_count %s // num_iterations %s // prev_best_f1 %s" %
              (iter_start_count,num_iterations,prev_best_f1))

        e_to_ms = {}
        labels = []
        for d in data:
            label = d[1]
            labels.append(label)
            if label in e_to_ms:
                e_to_ms[label].append(d)
            else:
                e_to_ms[label] = [d]

        # num_iterations = self.config.refine_itrs if num_iterations is None else\
        #     num_iterations
        # you have to specify the number of iterations, even if this is just passing
        # in self.config.refine_itrs, requiring this because otherwise it is too easy
        # to have confusion with multiple training canopies vs num iterations
        assert num_iterations is not None

        output_path = os.path.join(
            self.config.experiment_out_dir, 'trained-models',
            self.config.model_name, self.config.trainer_name)
        mkdir_p(output_path)
        best_f1 = prev_best_f1
        for iter in range(iter_start_count,
                          iter_start_count + num_iterations):
            print("[TRAIN ITER (ENTITY MODEL)]\t %s" % iter)
            if iter % self.config.e_eval_every == 0:
                _, f1, thresh = self.dev_eval_multiple(iter, diagnostics, output_path)
                print('[WEIGHT]')
                print('self.model.e_output_layer.weight')
                print(self.model.e_output_layer.weight)

                if f1 >= best_f1:
                    self.config.best_model = self.config.model_filename
                    self.config.partition_threshold = float(thresh)
                    self.config.save_config(self.config.experiment_out_dir)
                    best_f1 = f1
            num_to_train_w = self.config.train_tree_size
            # Randomly sample mentions.
            random.shuffle(data)
            train_set = data[:num_to_train_w]
            sofl('[DEV TREE USING] EHAC')
            self.train_pwe_episode_ehac(train_set)
            sys.stdout.flush()  # to update the output in the log file.

        last_iter_no = iter_start_count + num_iterations -1
        print("train_e last_iter_no %s" % last_iter_no)
        if last_iter_no % self.config.eval_every != 0 and last_iter_no > 0:
            print('Running dev eval one last time %s ' % last_iter_no)
            _, f1, thresh = self.dev_eval_multiple(last_iter_no, diagnostics,
                                       output_path)
            if f1 >= best_f1:
                self.config.best_model = self.config.model_filename
                self.config.partition_threshold = float(thresh)
                self.config.save_config(self.config.experiment_out_dir)
                best_f1 = f1
        print('returning best_f1 %s iter_count %s' % (best_f1,last_iter_no+1))
        return best_f1, last_iter_no + 1

    def train_pwe_episode_ehac(self, dataset):
        """Train the entity model.

        Take the input dataset and run an exact inference with the entity model.
        Specifically, compute all merge scores and pick the best according to
        the entity model. For each such merge, use the model to estimate the
        precision, compute the true precision, and get a gradient. Apply all the
        gradients once training is done. (Not in the code yet but, we also want
        to show the model a perfect trajectory so that it can learn to recognize
        global maxima.

        Args:
            dataset - dataset to run on.

        Returns:
            Nothing.
        """
        ehac = EHAC(self.config, dataset, self.model)

        # features_debug = []
        weight_accum = []
        ex_accum = []
        label_accum = []
        # Get instantaneous reward for each potential merge
        for i in range(len(ehac.sorted_mergers)):
            n1, n2, np_e_score, ap, e_score = ehac.sorted_mergers[i]
            precision = MergePreTrainer._pre_of_new(n1, n2)
            assert len(n1.root().pts) * len(n2.root().pts) == 1
            # features_debug.append(ehac.model.sub_ent_model.emb(ap))
            weight_accum.append(1.0)
            ex_accum.append(e_score)
            label_accum.append(precision)

        res = ehac.next_agglom()
        while len(ehac.sorted_mergers) > 0:
            if res:
                n1, n2, np_e_score, ap, e_score = res
                assert n1.root() == n1
                assert n2.root() == n2
                assert n1 != n2
                precision = MergePreTrainer._pre_of_new(n1, n2)

                print('pw_score, np_e_score, pre')
                pw_score = ap.aproj_local['my_pw']
                print(pw_score, np_e_score, precision)

                print('e_score %s ap.aproj_local %s' % (np_e_score,ap.aproj_local))
                merged = ehac.merge(n1, n2, ap)
                # clean the list of sorted mergers.
                ehac.clean_mergers({merged.children[0], merged.children[1]})
                # update the scores in sorted mergers and resort.
                ehac.add_scores_with_entity(merged)

                # After merging compute the scores of merging with the new root.
                # To do this, find all the active roots and for each such root,
                # find the best mention for joining that root to e. Compute the
                # precision of each of those merges and train the model to
                # estimate those precisions.
                for (n1, n2, np_e_score, ap, e_score) in ehac.sorted_mergers:
                    if n1 == merged or n2 == merged:
                        precision = MergePreTrainer._pre_of_new(n1, n2)
                        weight = 1.0
                        # features_debug.append(ehac.model.sub_ent_model.emb(ap))
                        weight_accum.append(weight)
                        ex_accum.append(e_score)
                        label_accum.append(precision)

                # TODO(AK): might consider using Pedram's suggestion of fitting
                # TODO(AK): the optimal set of clusters.

            res = ehac.next_agglom()

        if ex_accum:
            self.e_grad_at(torch.cat(ex_accum),
                           Variable(torch.FloatTensor(label_accum)),
                           weight=torch.FloatTensor(weight_accum),features_debug=[])
        print('[WEIGHT]')
        print('self.model.e_output_layer.weight')
        print(self.model.e_output_layer.weight)
        print('E BIAS')
        print(self.model.e_output_layer.bias.data)

    def compute_loss(self, batch):
        """Compute the loss on a batch of instances."""
        exs, labels = batch
        exs = Variable(torch.FloatTensor(exs))
        labels = Variable(torch.FloatTensor(labels))
        if self.config.use_cuda:
            exs = exs.cuda()
            labels = labels.cuda()
        scores = self.model.pw_score_mat(exs)
        return self.bceloss(scores.squeeze(1), labels)

    def e_grad_at(self, e_score, label, weight=torch.ones(1),features_debug = []):
        """Compute the gradient at e_score wrt labels."""
        self.e_opt.zero_grad()
        # loss_fn = BCEWithLogitsLoss(weight=weight, size_average=False)
        loss_fn = MSELoss(size_average=False)
        # print('Inputs:')
        # print('\t e_score.shape = %s' % e_score.shape)
        # print('\t label = %s' % label.shape)
        # print('\t len(weight) = %s ' % len(weight))
        # print('\t len(feature_debug) = %s' % len(features_debug))

        # for i in range(len(features_debug)):
        #     print('Example %s\t%f\t%s\t%f\t%s' % (i,label[i].data.numpy()[0],weight[i],e_score[i].data.numpy()[0],features_debug[i]))
        loss = loss_fn(e_score, label)
        # print('[GRAD]')
        # print('self.model.e_output_layer.weight.grad')
        # print(self.model.e_output_layer.weight.grad)

        print('[LOSS]')
        print(loss)
        loss.backward()

        # print('[GRAD]')
        # print('self.model.e_output_layer.weight.grad')
        # print(self.model.e_output_layer.weight.grad)
        # print('e loss')
        # print(loss.cpu().data.numpy())
        self.e_opt.step()
        # print('[WEIGHT]')
        # print('self.model.e_output_layer.weight')
        # print(self.model.e_output_layer.weight)
        # print('[PW WEIGHT]')
        # print('self.model.pw_output_layer.weight')
        # print(self.model.pw_output_layer.weight)

    def best_threshold_among_files(self, roots, thresholds):
        """Find the best threshold for a set of trees."""
        best_t = 0.0
        best_f1 = -1
        for t in thresholds:
            tp, fp, gt = 0, 0, 0
            for r in roots:
                tp_r, fp_r, gt_r = r.tp_fp_gt_threshold(t)
                tp += tp_r
                fp += fp_r
                gt += gt_r
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / gt if gt > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        return best_f1, best_t

    def dev_eval_multiple(self, iter, diagnostics, output_path):
        sofl("")
        sofl('[BEGIN DEV EVAL]')

        # assume that dev files are in data/datasetname/canopy/dev/canopy/ments.json
        dev_data_dir = os.path.join('data', self.config.dataset_name,
                                    'eval', 'dev', 'canopy')
        dev_canopies = []
        dev_start = datetime.datetime.now()

        # Get dev files.
        with open(self.config.dev_files, 'r') as fin:
            for line in fin:
                dev_canopies.append(line.strip())

        micro_TP = 0
        micro_FP = 0
        micro_GT = 0

        micro_up_TP = 0
        micro_up_FP = 0
        micro_up_GT = 0
        dev_tree_roots = []
        all_scores = set()

        # Make sure you are not overwriting, if you are there is something wrong with the iteration count
        assert not os.path.exists(os.path.join(output_path,
                                               "pw_iter_{}.torch".format(iter))), \
            "Model exists and would be overwritten %s" %os.path.join(output_path,
                                                                                     "pw_iter_{}.torch".format(iter))

        sofl('[SAVING MODEL...')
        torch.save(self.model,
                   os.path.join(output_path,
                                "pw_iter_{}.torch".format(iter)))

        for idx, dev_canopy in enumerate(dev_canopies):
            dev_file = os.path.join(dev_data_dir, dev_canopy, 'ments.json')
            ms = []
            for m in Ment.load_ments(dev_file, self.model):
                if not self.config.only_use_labeled_dev or (
                        m[1] is not None and m[1] != "None"):
                    ms.append(m)
            self.config.random.shuffle(ms)
            # Restrict
            canopy_size = min(self.config.dev_max_canopy_size,len(ms))
            sofl('Loaded canopy %s with %s mentions and restricted the size of the canopy to %s' % (dev_canopy,len(ms),canopy_size))
            ms = ms[:canopy_size]
            inf_start = datetime.datetime.now()
            sofl('[DEV TREE USING] %s' %self.config.clustering_scheme)
            dev_tree = new_clustering_scheme(self.config, ms, self.model)
            dev_tree_roots.append(dev_tree.build_dendrogram())
            inf_end = datetime.datetime.now()
            inf_time_seconds = (inf_end - inf_start).total_seconds()
            sofl('[INFERENCE IN %ss]' % inf_time_seconds)
            diagnostics['dev_hacsl_time_%s' % dev_canopy] = inf_time_seconds
            save_dict_to_json(diagnostics, os.path.join(output_path,
                                                        'diagnostics.json'))

            sofl('[SCORING TREE...]')
            score_start = datetime.datetime.now()
            pre_ub, rec_ub, f1_ub = dev_tree_roots[-1].f1_best()
            pre, rec, f1 = dev_tree_roots[-1].f1_cluster_marker()
            tp_ub, fp_ub, gt_ub = dev_tree_roots[-1].tp_fp_gt_best()
            tp, fp, gt = dev_tree_roots[-1].tp_fp_gt_cluster_marker()

            micro_TP += tp
            micro_FP += fp
            micro_GT += gt

            micro_up_TP += tp_ub
            micro_up_FP += fp_ub
            micro_up_GT += gt_ub

            sofl("")
            sofl('[(Python) UPPER BOUND P/R/F1 %s]:\t%s\t%s\t%s' % (
                dev_canopy, pre_ub, rec_ub, f1_ub))
            sofl('[PREDICTED P/R/F1 %s]\t%s\t%s\t%s' % (
                dev_canopy, pre, rec, f1))

            # Visit all nodes and collect scores.
            frontier = [dev_tree_roots[-1]]
            while frontier:
                x = frontier.pop(0)
                if x.children:
                    frontier.append(x.children[0])
                    frontier.append(x.children[1])
                all_scores.add(x.my_score)
            score_end = datetime.datetime.now()
            score_time_seconds = (score_end - score_start).total_seconds()
            sofl('[SCORING IN %ss]' % score_time_seconds)

        dev_end = datetime.datetime.now()
        inf_time_seconds = dev_end - dev_start
        sofl('[DEV TOTAL TIME IN %ss]' % inf_time_seconds)
        pre = micro_TP / (
        micro_TP + micro_FP) if micro_TP + micro_FP > 0.0 else 0.0
        rec = micro_TP / (micro_GT) if micro_GT > 0.0 else 0.0
        f1 = 2.0 * (pre * rec) / (pre + rec) if (pre + rec) > 0 else 0
        pre_ub = micro_up_TP / (micro_up_TP + micro_up_FP) if (
                                                              micro_up_TP + micro_up_FP) > 0.0 else 0.0
        rec_ub = micro_up_TP / (micro_up_GT) if micro_up_GT > 0 else 0.0
        f1_ub = 2.0 * (pre_ub * rec_ub) / (pre_ub + rec_ub) if (
                                                               pre_ub + rec_ub) > 0 else 0.0
        sofl('[(Python) UPPER BOUND P/R/F1 %s]:\t%s\t%s\t%s' % (
            'micro', pre_ub, rec_ub, f1_ub))
        sofl('[PREDICTED P/R/F1 %s]\t%s\t%s\t%s' % ('micro', pre, rec, f1))
        score_obj = {"inf_time": inf_time_seconds, "pre": pre, "rec": rec,
                     "f1": f1, "pre_ub": pre_ub, "rec_ub": rec_ub,
                     "f1_ub": f1_ub, "config": self.config.__dict__}
        save_dict_to_json(score_obj, os.path.join(
            output_path, "dev_scores_iter_{}.json".format(iter)))
        self.config.model_filename = os.path.join(
            output_path, "pw_iter_{}.torch".format(iter))

        self.config.save_config(output_path,
                                filename='pwe_iter_%d.config' % iter)

        threshold_start = datetime.datetime.now()
        print('[FIND BEST OVERALL THRESHOLD]')
        sorted_tree_scores = sorted(list(all_scores))
        num_to_try = len(sorted_tree_scores) * self.config.fraction_of_thresholds_to_try_dev
        interval = int(len(sorted_tree_scores) / num_to_try)
        best_f, best_t = None, None
        # best_partition = None
        for i in range(0,len(sorted_tree_scores),interval):
            t = sorted_tree_scores[i]
            tp = 0
            fp = 0
            total_gt = 0
            for root in dev_tree_roots:
                total_gt += root.compute_gt()
                predicted = root.partition_threshold(t)
                assert sum([e.point_counter for e in predicted]) == root.point_counter
                for e in predicted:
                    tp += e.local_tp
                    fp += e.local_fp
            pre = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec = tp / total_gt if total_gt > 0.0 else 0.0
            f1 = F1Node.f1(tp, fp, total_gt)

            sofl('t, pre, rec, f1')
            sofl('%s, %s, %s, %s' % (t, pre, rec, f1))
            if best_f is None or best_f < f1:
                best_f = f1
                best_t = t
                # best_partition = predicted

        sofl('[BEST THRESHOLD F1] %s %s' % (best_t,best_f))
        sofl('self.model.pw_output_layer.weight')
        sofl('%s' % self.model.pw_output_layer.weight)
        sofl('self.model.e_output_layer.weight')
        sofl('%s' % self.model.e_output_layer.weight)
        sofl('self.model.pw_output_layer.bias')
        sofl('%s' % self.model.pw_output_layer.bias)
        sofl('self.model.e_output_layer.bias')
        sofl('%s' % self.model.e_output_layer.bias)
        # sofl('[Best Partition Stats]')
        # for c in best_partition:
        #     print('c.as_ment.attributes.aproj_local')
        #     print(c.as_ment.attributes.aproj_local)
        sofl('[END DEV EVAL]')
        threshold_end = datetime.datetime.now()
        threshold_time_seconds = threshold_end - threshold_start
        sofl('[THRESHOLD TIME IN %ss]' % threshold_time_seconds)
        sofl("")
        return f1_ub, best_f, best_t
