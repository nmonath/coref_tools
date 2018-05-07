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

import torch
import codecs
import subprocess
import json
import sys

import torch.nn as nn

def sofl(line):
    "Print the line to Standard Out and FLush"
    sys.stdout.write(line)
    sys.stdout.write("\n")
    sys.stdout.flush()

def cosine_similarity(set_one,set_two):
    if len(set_one) == 0 or len(set_two) == 0:
        return 0
    else:
        return len(set_one.intersection(set_two)) / (len(set_one) * len(set_two))


def activation_from_str(str_name):
    if str_name == 'relu':
        return nn.ReLU()
    elif str_name == 'sigmoid':
        return nn.Sigmoid()
    elif str_name == 'tanh':
        return nn.Tanh()


def file_lines(filename,codec):
    f = codecs.open(filename,'r',codec)
    for line in f:
        yield line.decode(codec)

    f.close()


def row_wise_dot(tensor1, tensor2):
    #print(type(tensor1), type(tensor2))
    return torch.sum(tensor1 * tensor2, dim=1,keepdim=True)


def wc_minus_l(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


def filter_json(the_dict):
    # print("filter_json")
    # print(the_dict)
    res = {}
    for k in the_dict.keys():
        # print("k : {} \t {} \t {}".format(k,the_dict[k],type(the_dict[k])))
        if type(the_dict[k]) is str or \
                        type(the_dict[k]) is float or \
                        type(the_dict[k]) is int or \
                        type(the_dict[k]) is list or \
                        type(the_dict[k]) is bool or \
                        the_dict[k] is None:
            res[k] = the_dict[k]
        elif type(the_dict[k]) is dict:
            res[k] = filter_json(the_dict[k])
    # print("res: {} ".format(res))
    return res


def save_dict_to_json(the_dict,the_file):
    with open(the_file, 'w') as fout:
        fout.write(json.dumps(filter_json(the_dict)))
        fout.write("\n")