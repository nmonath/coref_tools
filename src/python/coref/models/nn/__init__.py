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

from coref.models.nn.NSW import NSW

def new_nn_structure(name,config,score_fn, dataset=[]):
    if name == 'nsw':
        return NSW(dataset,score_fn,config.nn_k,config.nsw_r,config.random_seed)
    else:
        raise Exception('Unknown nn structure %s' % config.nn_structure)