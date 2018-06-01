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

from grinch.models.Ent import Ent
from copy import copy

class Centroid(Ent):
    def __init__(self, sum_vec, num_vec):
        """
        
        :param sum_vec: The sum of vectors in this cluster (could be a single data point)
        :param num_vec: The number of points in this cluster
        :return: 
        """
        super(Centroid,self).__init__()
        self.sum_vec = sum_vec
        self.num_vec = num_vec
        self.centroid = sum_vec / num_vec

    def merged_rep(self, other):
        self.sum_vec += other.sum_vec
        self.num_vec += other.num_vec
        self.centroid = self.sum_vec / self.num_vec

    def copy(self):
        return Centroid(self.sum_vec.copy(),copy(self.num_vec))