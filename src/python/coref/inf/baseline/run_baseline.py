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
from shutil import copytree

from coref.models.core.Ment import Ment
from coref.util.Config import Config
from coref.util.EvalF1 import eval_f1
from coref.util.EvalB3CubedF1 import eval_bcubed
from coref.util.EvalBLANC import eval_blanc
from coref.util.Graphviz import Graphviz


class ByCanopyBaseline(object):

    def __init__(self):
        pass

    def mention_to_cluster(self,ment):
        c = next(iter(ment.attributes['canopy']))
        return c.lower()


class ByNameBaseline(object):

    def __init__(self):
        pass

    def mention_to_cluster(self, ment):
        if ment.attributes['ln'] and ment.attributes['fn']:
            c = "ln_" + next(iter(ment.attributes['ln'])) + "_fn_" + next(iter(ment.attributes['fn']))
        elif ment.attributes['ln']:
            c = "ln_" + next(iter(ment.attributes['ln'])) + "_fn_"
        else:
            c = ment.mid
        return c.lower()

class ByNameBaselineStrict(object):

    def __init__(self):
        pass

    def mention_to_cluster(self, ment):
        if ment.attributes['ln'] and ment.attributes['fn'] and len(next(iter(ment.attributes['fn']))) > 2:
            c = "ln_" + next(iter(ment.attributes['ln'])) + "_fn_" + next(iter(ment.attributes['fn']))
        else:
            c = ment.mid
        return c.lower()

class ByFirstNameBaseline(object):

    def __init__(self):
        pass

    def mention_to_cluster(self, ment):
        c = next(iter(ment.attributes['canopy']))
        if ment.attributes['ln'] and ment.attributes['fn']:
            c = c + "_fn_" + next(iter(ment.attributes['fn']))
        elif ment.attributes['ln']:
            c = c + "_fn_"
        else:
            c = ment.mid
        return c.lower()

class ByFirstNameBaselineStrict(object):

    def __init__(self):
        pass

    def mention_to_cluster(self, ment):
        c = next(iter(ment.attributes['canopy']))
        if ment.attributes['ln'] and ment.attributes['fn'] and len(next(iter(ment.attributes['fn']))) > 2:
            c = c + "_fn_" + next(iter(ment.attributes['fn']))
        else:
            c = ment.mid
        return c.lower()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EHAC on dataset')
    parser.add_argument('config', type=str, help='the config file')
    parser.add_argument('--test_file', type=str, help='the dataset to run on')
    parser.add_argument('--outbase', type=str,
                        help='prefix of out dir within experiment_out_dir')
    parser.add_argument('--canopy_name', type=str,
                        help='name of output canopy dir (only used with '
                             'test_file)')
    parser.add_argument('--points_file', type=str,
                        help='path to the points file to evaluate with')
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

    debug = config.debug

    diagnostics = {}

    # Set up output dir
    config.experiment_out_dir = os.path.join(
        config.experiment_out_dir, 'results', 'pwehac', ts)
    output_dir = config.experiment_out_dir

    try:
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

    except OSError as e:
        if e.errno != errno.EEXIST:
            print('%s already exists' % output_dir)

    if config.model_name == 'ByCanopyBaseline':
        model = ByCanopyBaseline()
    elif config.model_name == 'ByNameBaseline':
        model = ByNameBaseline()
    elif config.model_name == 'ByNameBaselineStrict':
        model = ByNameBaselineStrict()
    elif config.model_name == 'ByFirstNameBaseline':
        model = ByFirstNameBaseline()
    elif config.model_name == 'ByFirstNameBaselineStrict':
        model = ByFirstNameBaselineStrict()
    else:
        raise Exception("Unknown Model: {}".format(config.model_name))

    g = Graphviz()

    for i, f in enumerate(config.test_files):
        print('[BEGIN TEST] %s\n' % f)
        pts = []
        counter = 0
        if config.out_by_canopy:
            canopy_out = os.path.join(output_dir, config.out_by_canopy[i])
        else:
            canopy_out = os.path.join(output_dir, 'canopy_%s' % str(i))
        os.makedirs(canopy_out)
        with open('%s/predicted.tsv' % canopy_out, 'w') as predf:
            with open('%s/gold.tsv' % canopy_out, 'w') as goldf:
                print('[CLUSTERING...]')
                clustering_time_start = time.time()
                for m, m.gt, m.id in Ment.load_ments(f):
                    if not m.attributes['canopy']:
                        m.attributes.aproj['canopy'] = config.out_by_canopy[i] if config.out_by_canopy else 'canopy_%s' % str(i)
                    predf.write('%s\t%s\n' % (m.mid, model.mention_to_cluster(m)))
                    if m.gt != "None":
                        goldf.write('%s\t%s\n' % (m.mid, m.gt))

                end_time = time.time()
                print('[TIME] %ss' % (end_time - clustering_time_start))

        print('[BEGIN POSTPROCESSING...]')
        pre, rec, f1 = eval_f1(config,'%s/predicted.tsv' % canopy_out,'%s/gold.tsv' % canopy_out,restrict_to_gold=True)
        print('[PREDICTED PW P/R/F1]\t%s\t%s\t%s' % (pre, rec, f1))
        print()
        with open('%s/predicted.f1.tsv' % canopy_out, 'w') as f1f:
            f1f.write('%s\t%s\t%s\n' % (pre, rec, f1))

        b3_pre, b3_rec, b3_f1 = eval_bcubed(config,'%s/predicted.tsv' % canopy_out,'%s/gold.tsv' % canopy_out,restrict_to_gold=True)
        print('[PREDICTED B3 P/R/F1]\t%s\t%s\t%s' % (b3_pre, b3_rec, b3_f1))
        print()
        with open('%s/predicted.b3f1.tsv' % canopy_out, 'w') as f1f:
            f1f.write('%s\t%s\t%s\n' % (b3_pre, b3_rec, b3_f1))

        blanc_pre, blanc_rec, blanc_f1 = eval_blanc(config, '%s/predicted.tsv' % canopy_out, '%s/gold.tsv' % canopy_out,restrict_to_gold=True)
        print('[PREDICTED BLANC P/R/F1]\t%s\t%s\t%s' % (blanc_pre, blanc_rec, blanc_f1))
        print()
        with open('%s/predicted.blancf1.tsv' % canopy_out, 'w') as f1f:
            f1f.write('%s\t%s\t%s\n' % (blanc_pre, blanc_rec, blanc_f1))

        print('[FINISHED POST PROCESSING]\t%s' % canopy_out)
    print('[DONE.]')



