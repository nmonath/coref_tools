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

from collections import defaultdict
import numpy as np

class AttributeProjection(object):
    def __init__(self, values=None, cont_sum=None, bb=None):
        self.aproj = defaultdict(set)
        if values:
            self.aproj.update(values)

        self.aproj_sum = {}  # dict mapping keys to vectors. Updates will sum.
        if cont_sum:
            self.aproj_sum.update(cont_sum)

        self.aproj_bb = {}       # dict mapping keys to vector bounding box.
        if bb:
            self.aproj_bb.update(bb)

        self.aproj_local = {}  # a dict that keeps local features that do not
                               # get propagated in an update.
        self.aproj_local_debug = {}  # a dict that keeps local features that do not
        # get propagated in an update.

    def update_slow(self,other):
        for k in other.aproj.keys():
            self.aproj[k].update(other.aproj[k])

        # print('attr_proj_update')
        # print(self.aproj_sum)
        # print(other.aproj_sum)

        for k in other.aproj_sum.keys():
            if k in self.aproj_sum:
                self.aproj_sum[k] += other[k]
            else:
                self.aproj_sum[k] = np.copy(other[k])

        for k in other.aproj_bb.keys():
            if k in self.aproj_bb:
                mins = np.min(np.array([self.aproj_bb[k][0],
                                        other.aproj_bb[k][0]]), axis=0)
                maxs = np.max(np.array([self.aproj_bb[k][1],
                                        other.aproj_bb[k][1]]), axis=0)
                self.aproj_bb[k] = (mins, maxs)
            else:
                self.aproj_bb[k] = (np.copy(other[k][0]), np.copy(other[k][1]))

    def update(self, other,model=None):
        """ Add the input aproj into this projection
        
        :param other: 
        :return: 
        """
        # Per entity model defined update method
        # mutates self with other's attrproj
        if model:
            model.update(self,other)
        else:
            self.update_slow(other)

    def __getitem__(self, item):
        if item in self.aproj_sum:
            return self.aproj_sum[item]
        elif item in self.aproj_bb:
            return self.aproj_bb[item]
        elif item in self.aproj_local:
            return self.aproj_local[item]
        else:
            return self.aproj[item]

    def union(self, other):
        new_ap = AttributeProjection()
        new_ap.update(self)
        new_ap.update(other)
        return new_ap

    def to_fmt_list(self):
        attrs = []
        for k in self.aproj.keys():
            for v in self.aproj[k]:
                attrs.append('%s:%s' % (k, v))
        for k in self.aproj_sum.keys():
            if type(self.aproj_sum[k]) == float:
                attrs.append('%s:%s' % (k, self.aproj_sum[k]))
            else:
                for v in self.aproj_sum[k]:
                    attrs.append('%s:%s' % (k, v))
        for k in self.aproj_bb.keys():
            for v in self.aproj_bb[k]:
                attrs.append('%s:%s' % (k, v))
        return attrs

    def __str__(self):
        s = "{ "
        # for k in self.aproj.keys():
        #     s += k + ": [{}] ".format(", ".join(self.aproj[k]))
        for k in self.aproj_local.keys():
            s += "%s: %s " % (k,self.aproj_local[k])
        # for k in self.aproj_sum.keys():
        #     s += k + ": [{}] ".format(", ".join(str(self.aproj_sum[k])))
        s += " }"
        return s


    def __del__(self):
        self.aproj = None
        self.aproj_local = None
        self.aproj_bb = None
        self.aproj_sum = None
