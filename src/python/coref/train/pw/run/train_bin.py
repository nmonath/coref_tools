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

import argparse
import datetime
import errno
import os
import sys
from shutil import copytree

import torch

from coref.models import new_model
from coref.train import new_trainer
from coref.util.Config import Config
from coref.util.IO import copy_source_to_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PWE HAC on dataset')
    parser.add_argument('config', type=str, help='the config file')
    parser.add_argument('--outbase', type=str,
                        help='prefix of out dir within experiment_out_dir')
    parser.add_argument('--dataname', type=str, help='Name of dataset.')
    args = parser.parse_args()

    config = Config(args.config)
    if args.outbase:
        ts = args.outbase
        dataname = args.dataname
        ts = os.path.join(dataname, ts)
    else:
        now = datetime.datetime.now()
        ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second)

    debug = config.debug

    diagnostics = {}

    # Set up output dir
    config.experiment_out_dir = os.path.join(
        config.experiment_out_dir, ts)
    output_dir = config.experiment_out_dir

    copy_source_to_dir(output_dir,config)

    if config.batcher_filename != 'None':
        batcher = torch.load(config.batcher_filename)
    else:
        batcher = None

    model = new_model(config)
    config.save_config(config.experiment_out_dir)

    trainer = new_trainer(config, model)
    trainer.train(batcher, config.experiment_out_dir, None)
