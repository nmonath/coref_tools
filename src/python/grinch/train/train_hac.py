"""
Copyright (C) 2018 IBM Corporation.
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

import os, sys, random, datetime

import torch.optim
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss

from grinch.util.Misc import sofl
from grinch.util.IO import mkdir_p
from grinch.xdoccoref import new_grinch
from collections import defaultdict
from sklearn.cluster.k_means_ import KMeans

from grinch.xdoccoref.Vocab import TypedVocab
from grinch.util.Config import Config
from grinch.models.core.F1Node import F1Node
from grinch.xdoccoref import build_model

import numpy as np
import grinch

import logging
from coref.train.hac.MergePreTrainer import MergePreTrainer
from grinch.xdoccoref.Load import load_mentions_from_file
from grinch.util.IO import lines


class TrainHac(MergePreTrainer):
    def __init__(self, config, vocab, model):
        # super(TrainHac, self).__init__(config,vocab,model)
        self.model = model
        self.trainer = 'MergePreTrainer'
        self.config = config
        self.vocab = vocab
        self.bceloss = BCEWithLogitsLoss()
        self.buffer_size = config.buffer_size
        self.obj = 'pre'
        self.e_opt = torch.optim.Adam(self.model.entity_parameters(),
                                      lr=config.e_lr,
                                      weight_decay=config.l2penalty)
        self.pw_opt = torch.optim.Adam(self.model.pairwise_parameters(),
                                       lr=config.pw_lr,
                                       weight_decay=config.l2penalty)

        self.one_label = Variable(torch.ones(1))
        self.zero_label = Variable(torch.zeros(1))
        self.logger = logging.getLogger('MergePreTrainer')
        self.logger.setLevel(logging.INFO)
        self.random = random.Random(self.config.random_seed)
        self.best_f1 = 0
        self.best_thresh = None
        self.mention_to_entity = None
        self.entity_to_mention = None
        self.surfaceform_to_entity = None
        self.token_to_entity = None

    def compute_loss(self, batch):
        if self.config.produce_sample_pairs:
            return self.compute_log_loss(batch)
        else:
            return self.compute_rank_loss(batch)

    def compute_log_loss(self, batch):
        """Compute the loss on a batch of instances."""
        m1, m2, labels = batch
        labels = Variable(torch.FloatTensor(labels))
        if self.config.use_cuda:
            labels = labels.cuda()
        scores = self.model.batch_singleton_scores(m1, m2)
        losses = self.bceloss(scores, labels)
        return losses

    def compute_rank_loss(self, batch):
        """Compute the loss on a batch of instances."""
        m1, m2, m3 = batch
        pos = self.model.batch_singleton_scores(m1, m2)
        neg = self.model.batch_singleton_scores(m1, m3)  # this runs the networks twice
        if self.config.use_cuda:
            target = torch.ones(len(m1)).cuda()
        else:
            target = torch.ones(len(m1))
        losses = self.bceloss(pos - neg, target)
        return losses

    def train(self, mentions, batcher, outdir, dev_data):
        # Train Pairwise
        self.logger.info("Training Pairwise for at most %s mini batches, eval every %s" % (
        self.config.pw_num_minibatches, self.config.eval_every))
        self.train_pw(batcher, outdir, dev_data)
        self.logger.info('Train entity....')
        self.model = torch.load(self.config.best_model)
        for p in self.model.parameters():
            p.requires_grad = True
        self.e_opt = torch.optim.Adam(self.model.entity_parameters(),
                                      lr=config.e_lr,
                                      weight_decay=config.l2penalty)
        self.train_ehac(mentions, dev_data)

    def train_pw(self, batcher, outdir, dev_data):
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
            if count % self.config.eval_every == 0 or (count == 1 and self.config.eval_first):
                self.dev_eval(dev_data)
            self.pw_opt.zero_grad()
            l = self.compute_loss(batch)
            total_loss += l.data.item()  # .numpy()[0]
            if count % 1 == 0 and count > 0:
                sofl('[PW Model] Batches %s | Examples %s | This Batch Loss %s | Average Loss %s' % (
                count, count * self.config.batch_size, l.data.item(), total_loss / count))
            l.backward()
            self.pw_opt.step()
            self.model.detach_and_set_to_none(batch[0], batch[1])
            count += 1
            sys.stdout.flush()
            if count > self.config.pw_num_minibatches:
                sofl('Hit limit of number of pw minibatches %s' % count)
                break

    def train_ehac(self, dataset, dev_data):
        self.logger.info("Training Entity Model Agglomerative ...")
        self.logger.info("Train Tree Size %s" % self.config.train_tree_size)
        for i in range(self.config.e_iterations):
            self.logger.info("Starting iteration %s of %s" % (i, self.config.e_iterations))
            if self.config.random_sample_ehac:
                subset = self.random.sample(dataset, self.config.train_tree_size)
            else:
                subset = self.choose_ehac_dataset(dataset, self.config.train_tree_num_entites)
            self.train_ehac_episode(subset)
            if i % self.config.e_eval_every:
                self.logger.info('DEV EVAL // train_ehac')
                self.dev_eval(dev_data)

    def entity_based_sample(self, num_entities, dataset):
        max_per_entity = int(self.config.train_tree_size / num_entities)
        by_entity = defaultdict(list)
        for d in dataset:
            by_entity[d.gt].append(d)
        entities = list(by_entity.keys())
        entity_sample = self.random.sample(entities, num_entities)
        sample = []
        for e in entity_sample:
            sample.extend(by_entity[e][:max_per_entity])
        return sample

    def choose_ehac_dataset(self, dataset, max_num_entities):
        assert max_num_entities < self.config.train_tree_size
        if self.mention_to_entity is None:
            self.logger.info('choose_ehac_dataset - priming sampling structures')
            self.mention_to_entity = dict()
            self.entity_to_mention = defaultdict(list)
            self.surfaceform_to_entity = defaultdict(set)
            self.token_to_entity = defaultdict(set)
            self.ambiguous_tokens = []
            non_singletons = set()

            def update_for_ment(entMent):
                self.mention_to_entity[entMent] = entMent.gt
                self.entity_to_mention[entMent.gt].append(entMent)
                self.surfaceform_to_entity[entMent.name_spelling].add(entMent.gt)
                for s in entMent.name_spelling.split(" "):
                    if len(s) > 2 and s.lower() not in grinch.xdoccoref.stopwords:
                        self.token_to_entity[s].add(entMent.gt)
                        if len(self.token_to_entity[s]) > 1:
                            self.ambiguous_tokens.append(s)
                if len(self.entity_to_mention[entMent.gt]) > 1:
                    non_singletons.add(entMent.gt)

            for idx, entMent in enumerate(dataset):
                if idx % 100 == 0:
                    self.logger.info('choose_ehac_dataset - preprocessed %s mentions' % idx)
                update_for_ment(entMent)
        # pick some mentions that have token overlap
        tok = self.ambiguous_tokens[self.random.randint(0, len(self.ambiguous_tokens) - 1)]
        self.logger.info('choose_ehac_dataset - Selected token %s ' % tok)
        ents = self.token_to_entity[tok]
        if max_num_entities < len(ents):
            self.logger.info(
                'choose_ehac_dataset - max num entities %s < len(ents) %s ' % (max_num_entities, len(ents)))
            ents = self.random.sample(ents, max_num_entities)
        max_per_entity = int(self.config.train_tree_size / len(ents))
        self.logger.info(
            'max_per_entity %s max_num_entities %s len(ents) %s' % (max_per_entity, max_num_entities, len(ents)))
        sample = []
        for e in ents:
            if max_per_entity < len(self.entity_to_mention[e]):
                self.logger.info('max_per_entity %s max_num_entities %s len(self.entity_to_mention[e]) %s' % (
                    max_per_entity, max_num_entities, len(self.entity_to_mention[e])))
                sample.extend(self.random.sample(self.entity_to_mention[e], max_per_entity))
            else:
                sample.extend(self.entity_to_mention[e])
        return sample

    def train_ehac_episode(self, dataset):
        self.model = self.model.train()
        ehac = new_grinch(self.config, self.model, False)
        # Compute examples from pairs
        pairs, pair_scores = ehac.prime(dataset)
        labels = torch.zeros_like(pair_scores)
        if self.config.use_cuda:
            labels = labels.cuda()
        for i in range(len(pairs)):
            n1, n2 = pairs[i]
            precision = TrainHac._pre_of_new(n1, n2)
            labels[i] = precision

        # Compute Loss on these pairs
        losses = self.compute_loss_hac(pair_scores, labels)
        if losses is not None:
            ehac.logger.info('[train] Loss of Pairwise Batch %s' % (losses.cpu().item()))
        else:
            ehac.logger.info('[train] Loss of Pairwise Batch NONE!')

        # Build the rest of the dendrogram, getting examples from this as well.
        while len(ehac.roots) > 2:  # don't compute gradient for last step...
            new_node, pairs, pair_scores = ehac.take_one_step()
            # if a move could be made
            if new_node:
                if pair_scores is not None and pair_scores.size(0) > 0:
                    labels = torch.zeros_like(pair_scores)
                    if self.config.use_cuda:
                        labels = labels.cuda()
                    for i in range(len(pairs)):
                        n1, n2 = pairs[i]
                        precision = TrainHac._pre_of_new(n1, n2)
                        labels[i] = precision
                    pair_loss = self.compute_loss_hac(pair_scores, labels)
                    if pair_loss:
                        if losses is None:
                            losses = pair_loss
                        else:
                            losses += pair_loss

                        ehac.logger.info(
                            '[train] Loss of Batch based on new node %s = %s' % (new_node.id, pair_loss.cpu().item()))
                    else:
                        ehac.logger.info(
                            '[train] Loss of Batch based on new node %s = None' % (new_node.id))
                else:
                    ehac.logger.info(
                        '[train] New node %s had no moves to be made by canopies' % (new_node.id))
            else:
                ehac.logger.info(
                    '[train] No move could be made with %s roots left' % len(ehac.roots))
                break

        losses.backward()

        # Take a gradient step
        self.e_opt.step()

    def train_ehac_episode_rank(self, dataset):

        self.model = self.model.train()
        ehac = new_grinch(self.config, self.model, False)
        # Compute examples from pairs
        pairs, pair_scores = ehac.prime(dataset)
        labels = torch.zeros_like(pair_scores)
        if self.config.use_cuda:
            labels = labels.cuda()
        for i in range(len(pairs)):
            n1, n2 = pairs[i]
            precision = TrainHac._pre_of_new(n1, n2)
            labels[i] = precision

        # Compute Loss on these pairs
        losses = self.compute_loss_hac(pair_scores, labels)
        ehac.logger.info('[train] Loss of Pairwise Batch %s' % (losses.cpu().item()))

        # Build the rest of the dendrogram, getting examples from this as well.
        while len(ehac.roots) > 2:  # don't compute gradient for last step...
            new_node, pairs, pair_scores = ehac.take_one_step()
            # if a move could be made
            if new_node:
                if pair_scores is not None and pair_scores.size(0) > 0:
                    labels = torch.zeros_like(pair_scores)
                    if self.config.use_cuda:
                        labels = labels.cuda()
                    for i in range(len(pairs)):
                        n1, n2 = pairs[i]
                        precision = TrainHac._pre_of_new(n1, n2)
                        labels[i] = precision
                    pair_loss = self.compute_loss_hac(pair_scores, labels)
                    losses += pair_loss
                    ehac.logger.info(
                        '[train] Loss of Batch based on new node %s = %s' % (new_node.id, pair_loss.cpu().item()))
                else:
                    ehac.logger.info(
                        '[train] New node %s had no moves to be made by canopies' % (new_node.id))
            else:
                ehac.logger.info(
                    '[train] No move could be made with %s roots left' % len(ehac.roots))
                break

        losses.backward()

        # Take a gradient step
        self.e_opt.step()

    def compute_loss_hac(self, scores, labels):
        if self.config.use_rank_training_hac:
            return self.compute_loss_hac_rank(scores, labels)
        else:
            return self.compute_loss_hac_log(scores, labels)

    def compute_loss_hac_rank(self, scores, labels):
        score_diffs = (scores.unsqueeze(0) - scores.unsqueeze(1)).view(-1, 1)
        better_pairs = ((labels.unsqueeze(0) - labels.unsqueeze(1)) > 0).view(-1, 1)
        score_diffs = score_diffs[better_pairs]
        if score_diffs.size(0) > 0:
            self.e_opt.zero_grad()
            loss_fn = BCEWithLogitsLoss(size_average=True)
            loss = loss_fn(score_diffs, torch.ones_like(score_diffs))
            return loss
        else:
            return None

    def compute_loss_hac_log(self, scores, labels):
        """Compute the gradient at e_score wrt labels."""
        self.e_opt.zero_grad()
        loss_fn = BCEWithLogitsLoss(size_average=True)
        loss = loss_fn(scores, labels)
        return loss

    def find_partition_threshold(self, grinch):
        """ Pick the partition threshold on the given tree

        :param dataset:
        :return:
        """
        grinch.root._make_f1_ready()
        all_scores = []
        root = grinch.root
        frontier = [root]
        while frontier:
            next = frontier.pop(0)
            if next.children:
                all_scores.append(next.score())
                frontier.extend(next.children)

        number_to_try = len(all_scores) * self.config.fraction_of_thresholds_to_try_dev
        if number_to_try == len(all_scores):
            scores_to_try = all_scores
        else:
            scores_to_try = KMeans(number_to_try).fit(np.array([[x] for x in all_scores])).cluster_centers_
        best_f = None
        best_t = None
        scores_to_try = sorted(scores_to_try)
        for i in range(0, len(scores_to_try)):
            t = scores_to_try[i]
            tp = 0
            fp = 0
            total_gt = 0
            total_gt += root.compute_gt()
            predicted = root.partition_threshold(t)
            assert sum([e.point_counter for e in predicted]) == root.point_counter
            for e in predicted:
                tp += e.local_tp
                fp += e.local_fp
            pre = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec = tp / total_gt if total_gt > 0.0 else 0.0
            f1 = F1Node.f1(tp, fp, total_gt)

            self.logger.debug('t, pre, rec, f1')
            self.logger.debug('%s, %s, %s, %s' % (t, pre, rec, f1))
            if best_f is None or best_f < f1:
                best_f = f1
                best_t = t
                # best_partition = predicted

        self.logger.info('[BEST THRESHOLD F1] %s %s' % (best_t, best_f))
        best_t = best_t.cpu().item()
        return best_f, best_t

    def dev_eval(self, dataset):
        # detach so as not to store gradients!
        self.model.detach()
        self.model.eval()
        dev_grinch = new_grinch(self.config, self.model)
        dev_grinch.build_dendrogram(dataset)
        this_best_f, this_best_t = self.find_partition_threshold(dev_grinch)
        if self.best_f1 is None or this_best_f >= self.best_f1:
            self.best_f1 = this_best_f
            self.best_thresh = this_best_t
            self.logger.info(
                'Saving model file: %s ' % os.path.join(self.config.experiment_out_dir, 'best_model.torch'))
            if os.path.exists(os.path.join(self.config.experiment_out_dir, 'best_model.torch')):
                try:
                    from shutil import copyfile
                    copyfile(os.path.join(self.config.experiment_out_dir, 'best_model.torch'),
                             os.path.join(self.config.experiment_out_dir, 'second_best_model.torch'))
                except:
                    self.logger.warning('could not move model file %s')
                    pass
            torch.save(dev_grinch.sim_model, os.path.join(self.config.experiment_out_dir, 'best_model.torch'))
            self.config.partition_threshold = self.best_thresh
            self.config.best_model = os.path.join(self.config.experiment_out_dir, 'best_model.torch')
            self.config.save_config(self.config.experiment_out_dir)
        mkdir_p(os.path.join(self.config.experiment_out_dir,'models'))
        ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second)
        torch.save(dev_grinch.sim_model,
                   os.path.join(self.config.experiment_out_dir, 'models', 'model-%s.torch' % (ts)))
        torch.save(dev_grinch.sim_model,
                   os.path.join(self.config.experiment_out_dir, 'model-latest.torch' ))
        self.config.best_model = os.path.join(self.config.experiment_out_dir, 'model-latest.torch')
        self.config.save_config(self.config.experiment_out_dir,filename='config-latest.json')
        self.config.best_model = os.path.join(self.config.experiment_out_dir, 'best_model.torch')
        self.model.attach()
        self.model.train()


def choose_dev_dataset(dataset, max_num_entities, num_pts, samples_to_draw=4):
    if len(dataset) < num_pts:
        return dataset
    mention_to_entity = dict()
    entity_to_mention = defaultdict(list)
    surfaceform_to_entity = defaultdict(set)
    token_to_entity = defaultdict(set)
    ambiguous_tokens = []
    non_singletons = set()

    def update_for_ment(entMent):
        mention_to_entity[entMent] = entMent.gt
        entity_to_mention[entMent.gt].append(entMent)
        surfaceform_to_entity[entMent.name_spelling].add(entMent.gt)
        for s in entMent.name_spelling.split(" "):
            if len(s) > 2 and s.lower() not in grinch.xdoccoref.stopwords:
                token_to_entity[s].add(entMent.gt)
                if len(token_to_entity[s]) > 1:
                    ambiguous_tokens.append(s)
        if len(entity_to_mention[entMent.gt]) > 1:
            non_singletons.add(entMent.gt)

    size_per_sample = int(num_pts / samples_to_draw)
    sample = []
    dev_rand = random.Random(1917)
    for idx, entMent in enumerate(dataset):
        if idx % 100 == 0:
            print('choose_dev - preprocessed %s mentions' % idx)
        update_for_ment(entMent)
    toks = []
    for i in range(samples_to_draw):
        # pick some mentions that have token overlap
        tok = ambiguous_tokens[dev_rand.randint(0, len(ambiguous_tokens) - 1)]
        print('choose_dev - Selected token %s ' % tok)
        toks.append(tok)
    for tok in toks:
        print('processing tok %s' % tok)
        ents = token_to_entity[tok]
        if max_num_entities < len(ents):
            print(
                'choose_dev - max num entities %s < len(ents) %s ' % (max_num_entities, len(ents)))
            ents = dev_rand.sample(ents, max_num_entities)
        max_per_entity = int(size_per_sample / len(ents))
        print(
            'max_per_entity %s max_num_entities %s len(ents) %s' % (max_per_entity, max_num_entities, len(ents)))

        for e in ents:
            if max_per_entity < len(entity_to_mention[e]):
                print('max_per_entity %s max_num_entities %s len(entity_to_mention[e]) %s' % (
                    max_per_entity, max_num_entities, len(entity_to_mention[e])))
                sample.extend(dev_rand.sample(entity_to_mention[e], max_per_entity))
            else:
                sample.extend(entity_to_mention[e])
    dev_rand.shuffle(sample)
    return sample


if __name__ == '__main__':
    config = Config(sys.argv[1])
    ts = sys.argv[2] if len(sys.argv) > 2 else None

    now = datetime.datetime.now()
    if ts is None:
        ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second)

    # exp_out/dataset/model/cluster/time

    config.experiment_out_dir = os.path.join(
        config.experiment_out_dir, config.dataset_name,
        config.model_name, config.clustering_scheme, ts)
    mkdir_p(config.experiment_out_dir)
    print('Output Dir: %s ' % config.experiment_out_dir)

    # Load data from this file
    mention_file = config.train_files[0]
    vocab_file = config.vocab_file
    vocab = TypedVocab(vocab_file)
    # mentions = [m for m in Ment.load_ments(mention_file, json_attrs=True)]
    from grinch.xdoccoref.PretrainedModels import build_elmo, build_ft

    ft = build_ft()
    elmo = build_elmo()
    mentions = [m for m in load_mentions_from_file(mention_file, ft, elmo, use_cuda=config.use_cuda)]
    model = build_model(config, vocab)

    if config.use_cuda:
        model = model.cuda()

    trainer = TrainHac(config, vocab, model)
    if config.train_pair_file is not None and config.train_pair_file != "":
        pairs = []
        for line in lines(config.train_pair_file):
            splt = line.strip().split("\t")
            if config.produce_sample_pairs:
                pairs.append((splt[0], splt[1], int(splt[2])))
            else:
                pairs.append((splt[0], splt[1], splt[2]))
        print('Loaded %s pairs' % len(pairs))
    from grinch.xdoccoref.XDocBatcher import XDocBatcher

    batcher = XDocBatcher(config, mentions, pairs, return_one_epoch=False)

    # Load dev data
    dev_mention_file = config.dev_files[0]
    dev_data = []
    # for m in Ment.load_ments(dev_mention_file,json_attrs=True):
    #     dev_data.append(m)
    for m in load_mentions_from_file(dev_mention_file, ft, elmo, use_cuda=config.use_cuda):
        dev_data.append(m)

    # Create dev data
    dev_data = choose_dev_dataset(dev_data, 40, config.dev_max_canopy_size, config.dev_num_sf_samples)

    trainer.train(mentions, batcher, config.experiment_out_dir, dev_data)
    config.best_model = os.path.join(config.experiment_out_dir, 'best_model.torch')
    config.partition_threshold = trainer.best_thresh
    config.save_config(config.experiment_out_dir)