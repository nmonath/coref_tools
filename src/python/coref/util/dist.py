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

import math
from numba import jit
import numpy as np


@jit(nopython=True)
def _fast_norm(x):
    """Compute the number of x using numba.
    Args:
    x - a numpy vector (or list).
    Returns:
    The 2-norm of x.
    """
    s = 0.0
    for i in range(len(x)):
        s += x[i] ** 2
    return math.sqrt(s)


@jit(nopython=True)
def _fast_norm_diff(x, y):
    """Compute the norm of x - y using numba.
    Args:
    x - a numpy vector (or list).
    y - a numpy vector (or list).
    Returns:
    The 2-norm of x - y.
    """
    return _fast_norm(x - y)


@jit(nopython=True)
def _fast_min_to_box(mns, mxs, x):
    """Compute the minimum distance of x to a bounding box.
    Take a point x and a bounding box defined by two vectors of the min and max
    coordinate values in each dimension.  Compute the minimum distance of x to
    the box by computing the minimum distance between x and min or max in each
    dimension.  If, for dimension i,
    self.mins[i] <= x[i] <= self.maxes[i],
    then the distance between x and the box in that dimension is 0.
    Args:
    mns - a numpy array of floats representing the minimum coordinate value
        in each dimension of the bounding box.
    mxs - a numpy array of floats representing the maximum coordinate value
        in each dimension of the bounding box.
    x - a numpy array representing the point.
    Returns:
    A float representing the minimum distance betwen x and the box.
    """
    return _fast_norm(np.maximum(np.maximum(x - mxs, mns - x), 0))


@jit(nopython=True)
def _fast_max_to_box(mns, mxs, x):
    """Compute the maximum distance of x to a bounding box.
    Take a point x and a bounding box defined by two vectors of the min and max
    coordinate values in each dimension.  Compute the maximum distance of x to
    the box by computing the maximum distance between x and min or max in each
    dimension.
    Args:
    mns - a numpy array of floats representing the minimum coordinate value
        in each dimension of the bounding box.
    mxs - a numpy array of floats representing the maximum coordinate value
        in each dimension of the bounding box.
    x - a numpy array representing the point.
    Returns:
    A float representing the minimum distance betwen x and the box.
    """
    return _fast_norm(np.maximum(mxs - x, x - mns))

