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
from coref.models.hac.Gerch import Gerch, Perch, Greedy
from coref.models.hac.MBEHAC import MBEHAC

from geo.models.Grinch import Greedy as GeoGreedy
from geo.models.Grinch import Perch as GeoPerch
from geo.models.Grinch import Grinch as GeoGrinch

from acoref.models.Grinch import ACorefGreedy,ACorefPerch,ACorefGrinch

def new_clustering_scheme(config, dataset, model):
    if config.clustering_scheme.lower() == 'ehac':
        return EHAC(config, dataset, model)
    elif config.clustering_scheme.lower() == 'eperch':
        return EPerch(config, dataset, model)
    elif config.clustering_scheme.lower() == 'eperchgreedy':
        return EPerchGreedy(config, dataset, model)
    elif config.clustering_scheme.lower() == 'gerch' or config.clustering_scheme.lower() == 'grinch':
        return Gerch(config, dataset, model)
    elif config.clustering_scheme.lower() == 'perch':
        return Perch(config, dataset, model)
    elif config.clustering_scheme.lower() == 'greedy':
        return Greedy(config, dataset, model)
    elif config.clustering_scheme.lower() == 'mbehac':
        return MBEHAC(config, dataset, model)
    elif config.clustering_scheme.lower() == 'geo-greedy':
        return GeoGreedy(config, dataset, model)
    elif config.clustering_scheme.lower() == 'geo-perch':
        return GeoPerch(config, dataset, model)
    elif config.clustering_scheme.lower() == 'geo-grinch':
        return GeoGrinch(config, dataset, model)
    elif config.clustering_scheme.lower() == 'acoref-greedy':
        return ACorefGreedy(config, dataset, model)
    elif config.clustering_scheme.lower() == 'acoref-perch':
        return ACorefPerch(config, dataset, model)
    elif config.clustering_scheme.lower() == 'acoref-grinch':
        return ACorefGrinch(config, dataset, model)
    else:
        raise Exception("Unknown Clustering Scheme %s" % config.clustering_scheme)
