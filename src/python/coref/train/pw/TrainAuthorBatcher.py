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

import torch.nn as nn
import numpy as np
import torch


class TrainAuthorModelBatcher(object):
    def __init__(self, config, shuffle=True, return_one_epoch=False):
        self.config = config
        self.batch_size = self.config.batch_size
        self.shuffle = shuffle
        self.return_one_epoch = return_one_epoch
        self.pair_ids = []
        self.dataset = []
        self.labels = []
        self.frozen = False
        self.num_examples = None
        self.start_index = 0
        self.curr_epoch = 0

    def add_example(self, example, label, pair_ids):
        self.dataset.append(example)
        self.labels.append(label)
        self.pair_ids.append(pair_ids)

    def freeze(self):
        self.frozen = True
        self.num_examples = len(self.dataset)
        self.dataset = np.asarray(self.dataset)
        self.labels = np.asarray(self.labels)
        self.pair_ids = np.asarray(self.pair_ids)

    def shuffle_data(self):
        """
        Shuffles maintaining the same order.
        """
        perm = np.random.permutation(self.num_examples).astype(np.int)  # perm of index in range(0, num_questions)
        assert len(perm) == self.num_examples
        self.dataset = self.dataset[perm]
        self.labels = self.labels[perm]
        self.pair_ids = self.pair_ids[perm]

    def save(self, filename):
        self.freeze()
        if self.shuffle:
            self.shuffle_data()
        torch.save(self, filename)

    def get_next_batch(self, dev=False):
        while self.curr_epoch < self.config.iterations:
            if self.start_index > self.num_examples - self.batch_size:
                if dev:    # in case we only want to go through the exs once.
                    return
                self.curr_epoch += 1
                self.reset()

                if self.shuffle:
                    self.shuffle_data()
            else:
                num_data_returned = min(self.batch_size, self.num_examples - self.start_index)
                assert num_data_returned > 0
                end_index = self.start_index + num_data_returned
                yield self.dataset[self.start_index:end_index], \
                      self.labels[self.start_index:end_index]
                self.start_index = end_index

    def reset(self):
        self.start_index = 0
