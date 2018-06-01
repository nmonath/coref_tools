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
import numpy as np
import os
import time
import random
import logging

from grinch.util.Config import Config
from grinch.eval.EvalDendrogramPurity import eval_dendrogram_purity

from grinch.util.Graphviz import Graphviz
from grinch.util.IO import copy_source_to_dir

from grinch.models.CntrdGrinch import CntrdGrinch

def load_data(filename):
    """Load geometric data."""
    with open(filename, 'r') as f:
        for line in f:
            splits = line.strip().split('\t')
            pid, l, vec = splits[0], splits[1], np.array([float(x)
                                                          for x in splits[2:]])
            yield ((vec, l, pid))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Approximate inference PWEHAC')
    parser.add_argument('config', type=str, help='the config file')
    parser.add_argument('--outbase', type=str,
                        help='prefix of out dir within experiment_out_dir')
    parser.add_argument('--points_file', type=str,
                        help='path to the points file to evaluate with')
    parser.add_argument('--random_seed', type=str, default='config',
                        help='The random seed to use or ')
    args = parser.parse_args()

    config = Config(args.config)
    if args.outbase:
        ts = args.outbase
    else:
        now = datetime.datetime.now()
        ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second)

    if args.random_seed != 'config':
        config.random_seed = int(args.random_seed)

    print('Using random seed %s' % config.random_seed)
    debug = config.debug
    if debug:
        logging.Logger.setLevel(logging.DEBUG)
    rand = random.Random(config.random_seed)
    random.seed(config.random_seed)
    config.random = rand

    # Set up output dir
    config.experiment_out_dir = os.path.join(
        config.experiment_out_dir, 'results', config.dataset_name,
        config.model_name, config.clustering_scheme, ts)
    output_dir = config.experiment_out_dir

    config.canopy_out = os.path.join(output_dir, 'trees')

    copy_source_to_dir(output_dir, config)
    os.makedirs(config.canopy_out)

    g = Graphviz()

    print('[BEGIN TEST] %s\n' % config.dataset_name)
    data_file = os.path.join('data', 'geo', config.dataset_name)
    pts = []
    counter = 0

    data_generator = load_data(data_file)

    grinch = CntrdGrinch(config)

    print('[CLUSTERING...]')
    clustering_time_start = time.time()

    print('[BEGIN POSTPROCESSING...]')

    grinch.build_dendrogram(data_generator)

    end_time = time.time()
    total_time = end_time - clustering_time_start
    print('[TIME] %ss' % total_time)

    with open('%s/time.tsv' % output_dir, 'w') as f1f:
        f1f.write('%s\t%s\t%s\n' % (
            config.clustering_scheme, 'NO-CANOPY', total_time))

    # with open('%s/num_computations.tsv' % output_dir, 'w') as f1f:
    #     f1f.write('%s\t%s\t%s\n' % (
    #         config.clustering_scheme, 'NO-CANOPY',
    #         hac_structure.num_computations if hasattr(
    #             hac_structure, 'num_computations') else 0))

    tree_file = '%s/tree.tsv' % output_dir
    grinch.root.serialize(tree_file)

    if args.points_file:
        config.points_file = args.points_file

    print('NOT CALLING BEST PARTITION OR F1')
    # best_partition = root.partition_best()
    # pre_ub, rec_ub, f1_ub = root.f1_best()

    # print()
    # print('[(Python) UPPER BOUND P/R/F1]:\t%s\t%s\t%s' % (
    #     pre_ub, rec_ub, f1_ub))
    # print()

    if args.outbase:
        print("NOT EVALING DENDROGRAM PURITY")
    else:
        eval_dendrogram_purity(config, tree_file)

    print()
    if len(pts) < 500:
        print('[WRITING GRAPHVIZ TREE]\t%s/tree.gv' % output_dir)
        with open('%s/tree.gv' % output_dir, 'w') as treef:
            treef.write(g.graphviz_tree(grinch.root))

    print('[FINISHED POST PROCESSING]\t%s' % output_dir)
    print('[DONE.]')
