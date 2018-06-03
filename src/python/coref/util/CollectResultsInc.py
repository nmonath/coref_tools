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
import errno
import json
import numpy as np
import os

from collections import defaultdict

from coref.util.Config import Config


def read_f1(f):
    """Given a micro_f1 file, read precision, recall and F1"""
    with open(f, 'r') as fin:
        s = fin.readline()
        splits = s.split('\t')
        return float(splits[2]), float(splits[3]), float(splits[4])

def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result

def short_name_from_config(config):
    if config.clustering_scheme == 'mbehac':
        return "%s_mbsize_%s" % (config.clustering_scheme,config.mbsize)
    else:
        return "%s_exact_%s_atm_%s_nnk_%s_nswr_%s_ebp_%s" % (config.clustering_scheme, config.exact_nn, config.add_to_mention,
                                                           config.nn_k,config.nsw_r,config.exact_best_pairwise)

if __name__ == "__main__":
    """
    Takes in a single run's experiment out directory, for instance:
    exp_out/rexa/2018-05-05-05-34-21/run_9/results/greedy/2018-05-05-11-00-06/run_9
    
    $ ls exp_out/rexa/2018-05-05-05-34-21/run_9/results/greedy/2018-05-05-11-00-06/run_9
    shuffling_1  shuffling_10  shuffling_2  shuffling_3  shuffling_4  
    shuffling_5  shuffling_6  shuffling_7  shuffling_8  shuffling_9

    Creates a file:
        exp_out/rexa/2018-05-05-05-34-21/run_9/results/greedy/2018-05-05-11-00-06/run_9/results.json
        
        {
          "short_name": "greedy_exact_True_atm_True_nnk_5_nswr_3",
          "dataset_name":
          "mean_prec":
          "mean_rec":
          "mean_f1": 
          "std_prec":
          "std_rec":
          "std_f1":
          "min_prec":
          "min_rec":
          "min_f1":
          "max_prec":
          "max_rec":
          "max_f1":
        }
    
    And a file: exp_out/rexa/2018-05-05-05-34-21/run_9/results/greedy/2018-05-05-11-00-06/run_9/results.tsv
    which has the same order of fields as above but is tab separated
    
    
    """
    parser = argparse.ArgumentParser(description='Evalute a set of runs.')
    parser.add_argument('indir', type=str, help='the root of the runs')
    args = parser.parse_args()
    indir = args.indir

    configs = find_all('config.json',indir)
    config = Config(configs[0])
    short_name = short_name_from_config(config)
    dataset_name = config.dataset_name
    
    micro_f1_files = find_all('micro_f1_thresholded.tsv',indir)
    
    precs,recs,f1s = [],[],[]
    for f in micro_f1_files:
        p,r,f1 = read_f1(f)
        precs.append(p)
        recs.append(r)
        f1s.append(f1)
    precs = np.array(precs)
    recs = np.array(recs)
    f1s = np.array(f1s)
    results = {
              'short_name': short_name,
              'dataset_name': dataset_name,
              'mean_f1': np.mean(f1s),
              'mean_prec': np.mean(precs),
              'mean_rec': np.mean(recs),
              'std_f1': np.std(f1s),
              'std_prec': np.std(precs),
              'std_rec': np.std(recs),
              'min_f1': np.min(f1s),
              'min_prec': np.min(precs),
              'min_rec': np.min(recs),
              'max_f1': np.max(f1s),
              'max_prec': np.max(precs),
              'max_rec': np.max(recs)
               }
    outf = os.path.join(indir, 'results.json')
    with open(outf, 'w') as fout:
        fout.write('%s\n' % json.dumps(results,sort_keys=True))
    outf = os.path.join(indir, 'results.tsv')
    with open(outf, 'w') as fout:
        fout.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (
            results['short_name'],
            results['dataset_name'],
            results['mean_prec'],
            results['mean_rec'],
            results['mean_f1'],
            results['std_prec'],
            results['std_rec'],
            results['std_f1'],
            results['min_prec'],
            results['min_rec'],
            results['min_f1'],
            results['max_prec'],
            results['max_rec'],
            results['max_f1']
        ))