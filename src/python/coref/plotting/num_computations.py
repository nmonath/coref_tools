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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys

from coref.plotting.utils import hide_top_right

if __name__ == "__main__":
    input_files = sys.argv[1:]
    labels = sys.argv[-5]
    out_file = sys.argv[-4]
    x_label = sys.argv[-3]
    y_label = sys.argv[-2]
    title = sys.argv[-1]
    labels = labels.split(" ")

    fig = plt.figure(1)
    ax = plt.subplot(111)

    colors = [
                "#ca0020",
                "#33a02c",
                "#f4a582",
                "#92c5de",
                "#0571b0"
             ]

    for input_file,label,color in zip(input_files,labels,colors):
        X = []
        Y = []
        with open(input_file) as fin:
            for idx,line in enumerate(fin):
                splt = line.split("\t")
                if len(splt) == 2:
                    X.append(float(splt[0]))
                    Y.append(float(splt[1]))
                else:
                    X.append(float(idx))
                    Y.append(float(splt[0]))
        plt.plot(X,Y,label=label,c=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(frameon=False,loc='upper left')
    hide_top_right(ax)
    plt.savefig(out_file)

