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

import os


def eval_bcubed(config, predicted_cluster_assignment_file,gold_cluster_assignment_file,restrict_to_gold=False):
    """ Evaluate the b cubed p/r/f1

    Writes the score to a file and returns that value.
    You must have XCLUSTER_ROOT set.

    :param config:
    :return:
    """

    outf = os.path.join(os.path.dirname(predicted_cluster_assignment_file), 'tmp-bcubed-f1.tsv')
    os.system('set -exu; sh $XCLUSTER_ROOT/bin/util/score_bcubed.sh %s %s %s %s %s %s'
              '> %s' % (predicted_cluster_assignment_file, gold_cluster_assignment_file, config.model_name, config.dataset_name,
                        config.points_file,restrict_to_gold, outf))

    prec, rec, f1 = None, None, None
    with open(outf, 'r') as fin:
        for line in fin:
            splt = line.strip().split('\t')
            prec = float(splt[-3])
            rec = float(splt[-2])
            f1 = float(splt[-1])
    return prec,rec,f1
