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


def eval_dendrogram_purity(config, filename, pf=None):
    """ Evaluate the dendrogram

    Writes the score to a file and returns that value.
    You must have XCLUSTER_ROOT set.

    :param config: 
    :return: 
    """

    outf = os.path.join(os.path.dirname(filename), 'dp.tsv')
    if pf is not None:
        points_file = pf
    else:
        points_file = config.points_file
    os.system("sh $XCLUSTER_ROOT/bin/util/score_tree.sh {} {} {} {} {} > {}"
              .format(filename, config.model_name, config.dataset_name,
                    config.threads, points_file, outf))
    cost = None
    # with open(os.path.join(config.experiment_out_dir, 'dp.tsv'), 'r') as fin:
    with open(outf, 'r') as fin:
        for line in fin:
            splt = line.strip().split("\t")
            cost = float(splt[-1])
    return cost