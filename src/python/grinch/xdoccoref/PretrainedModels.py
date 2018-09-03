"""
Copyright (C) 2018 IBM Corporation.
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

from sklearn.feature_extraction.hashing import _hashing
import fastText

def build_elmo():
    pass

def build_ft():
    ft = fastText.load_model('exp_out/wiki-links/2018-09-01-11-14-42/embeddings.bin')
    # ft = fastText.load_model('exp_out/wiki-links/2018-09-01-10-49-52/embeddings.bin')
    return ft

def build_canopy_hasher():
    from sklearn.feature_extraction.hashing import FeatureHasher
    fvh = FeatureHasher()
    return fvh

CanopyHasher = build_canopy_hasher()

def hash_canopies(canopy_list):
    indices, _,_ = _hashing.transform([[(c,1)] for c in canopy_list], CanopyHasher.n_features, CanopyHasher.dtype,CanopyHasher.alternate_sign)
    return indices