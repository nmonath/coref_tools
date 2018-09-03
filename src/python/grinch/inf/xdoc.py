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
    rand = random.Random(config.random_seed)
    random.seed(config.random_seed)
    config.random = rand

    # Set up output dir
    config.experiment_out_dir = os.path.join(
        config.experiment_out_dir, 'results', config.dataset_name,
        config.model_name, config.clustering_scheme, ts)
    output_dir = config.experiment_out_dir

    copy_source_to_dir(output_dir, config)

    g = Graphviz()

    print('[BEGIN TEST] %s\n' % config.dataset_name)
    data_file = os.path.join('data', 'geo', config.dataset_name)
    pts = []
    counter = 0

    from grinch.xdoccoref.Load import load_mentions_from_file

    data = [x for x in load_mentions_from_file(config.test_files[0])][:500]
    random.shuffle(data)

    from grinch.xdoccoref import new_grinch

    grinch = new_grinch(config)

    print('[CLUSTERING...]')
    clustering_time_start = time.time()

    print('[BEGIN POSTPROCESSING...]')

    grinch.build_dendrogram(data)

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

    thresh_partition = grinch.root.partition_threshold(
        config.partition_threshold)

    # TODO: Move this somewhere else.
    for e in thresh_partition:
        e.is_cluster_root = True

    with open('%s/gold.tsv' % output_dir, 'w') as predf:
        for e in thresh_partition:
            for l in e.leaves():
                assert len(l.pts) == 1
                m = l.pts[0]
                predf.write('%s\t%s\n' % (m.mid, m.gt))

    print('[WRITING THRESHOLD PARTITION]\t%s/thresholded.tsv' %
          output_dir)
    with open('%s/thresholded.tsv' % output_dir, 'w') as predf:
        for e in thresh_partition:
            for l in e.leaves():
                assert len(l.pts) == 1
                m =  l.pts[0]
                predf.write('%s\t%s\n' % (m.mid, e.id))

    from coref.util.EvalF1 import eval_f1
    from coref.util.EvalB3CubedF1 import eval_bcubed

    pre_pred_scala, rec_pred_scala, f1_pred_scala = eval_f1(config, '%s/thresholded.tsv' % output_dir,
                                                            '%s/gold.tsv' % output_dir, restrict_to_gold=True)

    pre_pred_scala_b3, rec_pred_scala_b3, f1_pred_scala_b3 = eval_bcubed(config,
                                                                         '%s/thresholded.tsv' % output_dir,
                                                                         '%s/gold.tsv' % output_dir,
                                                                         restrict_to_gold=True)

    with open('%s/thresholded.pw.f1.scala.tsv' % output_dir, 'w') as f1f:
        f1f.write('%s\t%s\t%s\n' % (pre_pred_scala, rec_pred_scala, f1_pred_scala))

    with open('%s/thresholded.b3.f1.scala.tsv' % output_dir, 'w') as f1f:
        f1f.write('%s\t%s\t%s\n' % (pre_pred_scala_b3, rec_pred_scala_b3, f1_pred_scala_b3))

    print()
    if len(pts) < 500:
        print('[WRITING GRAPHVIZ TREE]\t%s/tree.gv' % output_dir)
        with open('%s/tree.gv' % output_dir, 'w') as treef:
            treef.write(g.graphviz_tree(grinch.root))

    print('[FINISHED POST PROCESSING]\t%s' % output_dir)
    print('[DONE.]')
