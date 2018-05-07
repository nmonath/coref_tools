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
import os
from shutil import copytree

from coref.util.Config import Config

if __name__ == "__main__":
    """Pass through to have the same format for baseline algs"""
    parser = argparse.ArgumentParser(description='Pass through')
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
    os.makedirs(output_dir)

    # save the vocab to out dir.
    # copyfile(config.vocab_file, os.path.join(output_dir, 'vocab.tsv'))
    # save the source code.
    copytree(os.path.join(os.environ['COREF_ROOT'], 'src'),
             os.path.join(output_dir, 'src'))
    copytree(os.path.join(os.environ['COREF_ROOT'], 'bin'),
             os.path.join(output_dir, 'bin'))
    copytree(os.path.join(os.environ['COREF_ROOT'], 'config'),
             os.path.join(output_dir, 'config'))

    # save the config to outdir.
    config.save_config(output_dir)