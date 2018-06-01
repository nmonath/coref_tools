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

from grinch.models.GNode import GNode
from grinch.models.nn.NSWNode import NSWNode

class GNSWNode(GNode,NSWNode):

    def __init__(self, ent, e_score_fn,max_degree=None):
        GNode.__init__(self,ent,e_score_fn)
        NSWNode.__init__(self,e_score_fn,max_degree)
