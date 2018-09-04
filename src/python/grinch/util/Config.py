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


import json
import random
import os
from grinch.util.Misc import filter_json
import numpy as np

class Config(object):
    def __init__(self,filename=None):
        # Settings
        self.config_name = filename
        self.dataset_name = 'dataset'

        # Optimization
        self.optimizer = "Adam"
        self.eval_every = 5000
        self.e_eval_every = 100
        self.e_lr = 0.1
        self.pw_lr = 0.1
        self.l2penalty = 0.0
        self.pw_num_minibatches = np.inf
        self.batch_size = 100
        self.fraction_of_thresholds_to_try_dev = 1
        self.iterations = 1

        # IO
        self.experiment_out_dir = 'exp_out'
        self.train_files = None
        self.dev_files = None
        self.test_files = None
        self.codec = 'UTF-8'
        self.points_file = None
        self.pair_filename = None
        self.dev_pair_filename = None
        self.batcher_filename = None
        self.dev_batcher_filename = None
        self.canopy_out = ''

        # Misc
        self.random_seed = np.random.randint(1,99999)
        self.use_cuda = False
        self.threads = 12
        self.debug = False
        self.allow_merge_steal = False
        self.beam = 0

        # Training Procedure
        self.train_tree_size = 20
        self.trainer_name = "MergePreTrainer"
        self.only_use_labeled_train = False
        self.only_use_labeled_dev = False
        self.refine_itrs = 10
        self.dev_max_canopy_size = 100
        self.time_debug = True

        # Model
        self.activation = 'relu'
        self.e_arch = "lin"
        self.pw_arch = "lin"
        self.model_name = "RouterModel"
        self.pairwise_model_name = "BasePairwiseModel"
        self.entity_model_name = "BaseSubEntModel"
        self.model_filename = None
        self.fasttext = None
        self.best_model = None
        self.out_by_canopy = None
        self.partition_threshold = None
        self.boundary_tree_k = 5
        self.boundary_tree_t = 2
        self.exact_knn_num = 10
        self.apply_two_name_penalty = True
        self.pass_on_incompatible_names = False
        self.expit_e_score = False
        self.use_new_ft = True

        self.clustering_scheme = 'ehac'
        self.exact_best_pairwise = True
        self.approx_best_pairwise_budget = 100

        self.fast_graft = False
        self.fast_grafts_only = False
        self.graft_beam = 5

        self.ordering = None

        # PERCH HAC Settings
        self.mention_nn_structure = 'nsw'
        self.nn_structure = 'nsw'
        self.nn_k = 10
        self.nsw_r = 5
        self.nsw_l = 10
        self.add_to_mention = True
        self.exact_nn = False
        self.max_num_samples_avl = 25
        self.max_node_rotate_size = None
        self.max_node_graft_size = None
        self.aggressive_rejection_stop = False


        # Barely still used
        self.inference_method = 'exact'
        self.buffer_size = 500

        self.mbsize = 1000

        # Deprecated?
        self.k = None
        self.update_f = None
        self.plot_stats = None
        self.mtree = None
        self.write_every_tree = True
        self.mhac_max_size = 1000
        self.pwe_super_entities = "Rand"  # Rand for Random mergers, HAC-SL fpr Single linkage mergers
        self.pwe_merge_condition = 'improves_f1'
        self.warm_start = False # whether or not to load a warm model
        self.pwe_train = ""

        # Deprecated -l2s
        self.max_ltr_exs = None
        self.num_deviations = 10000
        self.l2s_train_agglom_every = None
        self.l2s_global = False
        self.l2s_reference_policy = "pure_agglom"
        self.l2s_beta = 0.25
        self.l2s_deviate_prob = 1
        self.remove_singletons = False
        self.l2s_rollout_len = None
        self.l2s_randomize = False
        self.l2s_freeze_pw = True

        self.train_pair_file = None
        self.num_pos_samples_per_pt = 30
        self.num_rand_neg_samples_per_pt = 0
        self.num_max_num_positives_per_cluster = 100
        self.num_hard_neg_samples_per_pt = 2
        self.produce_sample_pairs = True
        self.dev_num_sf_samples = 4
        self.vocab_file = None

        self.use_mlp_scoring_layer = False
        self.use_pairwise = {'name': False, 'context': False, 'name_ft': False}
        self.typed_dims = {'name': 100, 'context': 100, 'name_ft': 100}
        self.cnn_dims = {'name': 100, 'context': 100, 'name-ft': 100}
        self.cnn_pos_dims = {'name':10,'context':10}
        self.eval_first = False
        self.single_elimination = False
        self.use_single_search = True
        self.use_canopies = True
        self.max_num_leaves = 10000000
        self.typed_k_rep = {'name': 5, 'context': 5, 'name-ft': 5}
        self.max_degree = 200
        self.dropout = 0.3
        self.update_name = True
        self.update_context = True
        self.warm_start_context = True
        self.warm_start_name = False
        self.warm_start_context_glove = False
        self.use_cosine_sim = True
        self.e_iterations = 30

        if filename:
            self.__dict__.update(json.load(open(filename)))
        self.random = random.Random(self.random_seed)

    def to_json(self):
        return json.dumps(filter_json(self.__dict__),indent=4,sort_keys=True)

    def save_config(self, exp_dir, filename='config.json'):
        with open(os.path.join(exp_dir, filename), 'w') as fout:
            fout.write(self.to_json())
            fout.write('\n')


DefaultConfig = Config()
