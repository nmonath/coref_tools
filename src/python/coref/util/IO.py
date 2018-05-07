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
import errno
from shutil import copytree

def mkdir_p(filepath):
    try:
        os.makedirs(filepath)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def copy_source_to_dir(output_dir,config):
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