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

from coref.models.hac.EHAC import EHAC
from coref.models.hac.Gerch import Gerch,Perch,Greedy,BGerch
from coref.models.hac.MBEHAC import MBEHAC

def new_clustering_scheme(config, dataset, model):
    if config.clustering_scheme.lower() == 'ehac':
        return EHAC(config, dataset, model)
    elif config.clustering_scheme.lower() == 'gerch':
        return Gerch(config, dataset, model)
    elif config.clustering_scheme.lower() == 'perch':
        return Perch(config, dataset, model)
    elif config.clustering_scheme.lower() == 'greedy':
        return Greedy(config, dataset, model)
    elif config.clustering_scheme.lower() == 'mbehac':
        return MBEHAC(config, dataset, model)
    elif config.clustering_scheme.lower() == 'bgerch':
        return BGerch(config, dataset, model)
    else:
        raise Exception("Unknown Clustering Scheme %s" % config.clustering_scheme)
