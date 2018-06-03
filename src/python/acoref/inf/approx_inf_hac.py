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
import time
import random
from shutil import copytree

import coref
from coref.models.core.Ment import Ment
from coref.models.hac.EHAC import EHAC
from coref.util.Config import Config
from coref.util.EvalF1 import eval_f1
from coref.util.EvalB3CubedF1 import eval_bcubed
from coref.util.EvalBLANC import eval_blanc
from coref.util.Graphviz import Graphviz
from coref.util.IO import copy_source_to_dir

from coref.models.hac import new_clustering_scheme
from acoref.models.ACorefModel import ACorefModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Approximate inference PWEHAC')
    parser.add_argument('config', type=str, help='the config file')
    parser.add_argument('--test_file', type=str, help='the dataset to run on')
    parser.add_argument('--outbase', type=str,
                        help='prefix of out dir within experiment_out_dir')
    parser.add_argument('--canopy_name', type=str,
                        help='name of output canopy dir (only used with '
                             'test_file)')
    parser.add_argument('--points_file', type=str,
                        help='path to the points file to evaluate with')
    parser.add_argument('--random_seed', type=str,default='config',
                        help='The random seed to use or ')
    args = parser.parse_args()

    config = Config(args.config)
    if args.test_file:
        config.test_files = [args.test_file]
        config.out_by_canopy = [args.canopy_name]
    if args.outbase:
        ts = args.outbase
    else:
        now = datetime.datetime.now()
        ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second)

    if args.random_seed != 'config':
        config.random_seed = int(args.random_seed)

    rand = random.Random(config.random_seed)
    config.random = rand

    print('Using random seed %s' %config.random_seed)
    debug = config.debug

    # Keeping track of the dataset name
    config.dataset_name = "-".join(args.test_file.split("data/")[1].split("/")[0:3]) #todo: os agnostic

    # Set up output dir
    config.experiment_out_dir = os.path.join(
        config.experiment_out_dir, 'results', config.clustering_scheme, ts)
    output_dir = config.experiment_out_dir

    copy_source_to_dir(output_dir,config)

    # assert config.best_model is not None
    # assert config.partition_threshold is not None

    torch_model = coref.models.load_model(config)

    # Node Model is what is stored at each node in the tree
    # and what is used to compute the scores

    node_model = ACorefModel(config,torch_model,None,None,num_pts=0)

    g = Graphviz()

    for i, f in enumerate(config.test_files):

        print('[BEGIN TEST] %s\n' % f)
        pts = []
        counter = 0
        for pt in Ment.load_ments(f, model=torch_model):
            pts.append(pt)

        # Shuffle the data points
        rand.shuffle(pts)

        print('[CLUSTERING...]')
        clustering_time_start = time.time()

        print('[BEGIN POSTPROCESSING...]')
        if config.out_by_canopy:
            canopy_out = os.path.join(output_dir, config.out_by_canopy[i])
        else:
            canopy_out = os.path.join(output_dir, 'canopy_%s' % str(i))
        os.makedirs(canopy_out)
        config.canopy_out = canopy_out
        config.save_config(config.canopy_out)

        # NOTICE THE USE OF node_model here! New for the Grinch
        # the node model knows about the torch model
        hac_structure = new_clustering_scheme(config, pts, model=node_model)
        root = hac_structure.build_dendrogram()
        end_time = time.time()
        total_time = end_time - clustering_time_start
        print('[TIME] %ss' % total_time)

        with open('%s/time.tsv' % canopy_out, 'w') as f1f:
            f1f.write('%s\t%s\t%s\n' % (config.clustering_scheme, config.out_by_canopy[i], total_time))

        with open('%s/num_computations.tsv' % canopy_out, 'w') as f1f:
            f1f.write('%s\t%s\t%s\n' % (config.clustering_scheme, config.out_by_canopy[i], hac_structure.num_computations if hasattr(hac_structure,'num_computations') else 0))

        tree_file = '%s/tree.tsv' % canopy_out
        root.serialize(tree_file)

        if args.points_file:
            config.points_file = args.points_file

        # best_partition = root.partition_best()
        pre_ub, rec_ub, f1_ub = root.f1_best()

        print()
        print('[(Python) UPPER BOUND P/R/F1]:\t%s\t%s\t%s' % (
            pre_ub, rec_ub, f1_ub))
        thresh_partition = root.partition_threshold(
            config.partition_threshold)

        # TODO: Move this somewhere else.
        for e in thresh_partition:
            e.is_cluster_root = True

        pre_t, rec_t, f1_t = root.f1_threshold(config.partition_threshold)
        print('[THRESHOLD P/R/F1]\t%s\t%s\t%s' % (pre_t, rec_t, f1_t))
        print()
        with open('%s/gold.tsv' % canopy_out, 'w') as predf:
            for e in thresh_partition:
                for l in e.leaves():
                    assert len(l.pts) == 1
                    (m, l, id) = l.pts[0]
                    predf.write('%s\t%s\n' % (m.mid, m.gt))

        # print('[WRITING BEST TREE CONSISTENT PARTITION]\t%s/best.tsv' % canopy_out)
        # with open('%s/best.tsv' % canopy_out, 'w') as predf:
        #     for e in best_partition:
        #         for l in e.leaves():
        #             assert len(l.pts) == 1
        #             (m, l, id) = l.pts[0]
        #             predf.write('%s\t%s\n' % (m.mid, e.id))
        # with open('%s/best.f1.tsv' % canopy_out, 'w') as f1f:
        #     f1f.write('%s\t%s\t%s\n' % (pre_ub, rec_ub, f1_ub))

        print('[WRITING THRESHOLD PARTITION]\t%s/thresholded.tsv' %
              canopy_out)
        with open('%s/thresholded.tsv' % canopy_out, 'w') as predf:
            for e in thresh_partition:
                for l in e.leaves():
                    assert len(l.pts) == 1
                    (m, l, id) = l.pts[0]
                    predf.write('%s\t%s\n' % (m.mid, e.id))

        with open('%s/thresholded.f1.tsv' % canopy_out, 'w') as f1f:
            f1f.write('%s\t%s\t%s\n' % (pre_t, rec_t, f1_t))

        pre_pred_scala, rec_pred_scala, f1_pred_scala = eval_f1(config, '%s/thresholded.tsv' % canopy_out,
                                                                '%s/gold.tsv' % canopy_out, restrict_to_gold=True)

        pre_pred_scala_b3, rec_pred_scala_b3, f1_pred_scala_b3 = eval_bcubed(config,
                                                                             '%s/thresholded.tsv' % canopy_out,
                                                                             '%s/gold.tsv' % canopy_out,
                                                                             restrict_to_gold=True)

        with open('%s/thresholded.pw.f1.scala.tsv' % canopy_out, 'w') as f1f:
            f1f.write('%s\t%s\t%s\n' % (pre_pred_scala, rec_pred_scala, f1_pred_scala))

        with open('%s/thresholded.b3.f1.scala.tsv' % canopy_out, 'w') as f1f:
            f1f.write('%s\t%s\t%s\n' % (pre_pred_scala_b3, rec_pred_scala_b3, f1_pred_scala_b3))

        if len(pts) < 500:
            print('[WRITING GRAPHVIZ TREE]\t%s/tree.gv' % canopy_out)
            with open('%s/tree.gv' % canopy_out, 'w') as treef:
                treef.write(g.graphviz_tree(root))

        print('[FINISHED POST PROCESSING]\t%s' % canopy_out)
    print('[DONE.]')
