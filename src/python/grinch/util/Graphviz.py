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


import numpy as np


class Graphviz(object):
    def __init__(self):
        self.internal_color = "lavenderblush4"
        self.colors = [
            "aquamarine", "bisque", "blue", "blueviolet", "brown", "cadetblue",
            "chartreuse", "coral", "cornflowerblue", "crimson", "darkgoldenrod",
            "darkgreen", "darkkhaki", "darkmagenta", "darkorange", "darkred",
            "darksalmon", "darkseagreen", "darkslateblue", "darkslategrey",
            "darkviolet", "deepskyblue", "dodgerblue", "firebrick",
            "forestgreen", "gainsboro", "ghostwhite", "gold", "goldenrod",
            "gray", "grey", "green", "greenyellow", "honeydew", "hotpink",
            "indianred", "indigo", "ivory", "khaki", "lavender",
            "lavenderblush", "lawngreen", "lemonchiffon", "lightblue",
            "lightcoral", "lightcyan", "lightgoldenrodyellow", "lightgray",
            "lightgreen", "lightgrey", "lightpink", "lightsalmon",
            "lightseagreen", "lightskyblue", "lightslategray", "lightslategrey",
            "lightsteelblue", "lightyellow", "limegreen", "linen", "magenta",
            "maroon", "mediumaquamarine", "mediumblue", "mediumorchid",
            "mediumpurple", "mediumseagreen", "mediumslateblue",
            "mediumturquoise", "midnightblue", "mintcream", "mistyrose",
            "moccasin", "navajowhite", "navy", "oldlace", "olive", "olivedrab",
            "orange", "orangered", "orchid", "palegoldenrod", "palegreen",
            "paleturquoise", "palevioletred", "papayawhip", "peachpuff", "peru",
            "pink", "powderblue", "purple", "red", "rosybrown", "royalblue",
            "saddlebrown", "salmon", "sandybrown", "seagreen", "seashell",
            "sienna", "silver", "skyblue", "slateblue", "slategray",
            "slategrey", "snow", "springgreen", "steelblue", "tan", "teal",
            "thistle", "tomato", "violet", "wheat", "burlywood", "chocolate"]
        self.color_map = {}
        self.color_counter = 0
        self.emphasis_1_nodes = []
        self.emphasis_1_shape = 'trapezium'
        self.emphasis_2_shape = 'invhouse'

    def format_id(self, ID):
        if not ID.startswith("id"):
            return ("id%s" % ID).replace('-', '')
        else:
            return ("%s" % ID).replace('-', '')

    def clean_label(self, s):
        return s.replace("[/:.]", "_")

    def get_node_label(self, node):
        lbl = []
        lbl.append(self.format_id(node.id))
        lbl.append('<BR/>')
        lbl.append('num pts: %d' % len(node.leaves()))
        lbl.append('<BR/>')
        try:
            lbl.append('purity: %f' % node.purity())
        except Exception:
            pass
        # TODO(AK) missing some information here like classes in node
        try:
            lbl.append('<BR/>')
            lbl.append('across: %s' % node.best_across_debug)
        except Exception:
            pass
        return ''.join(lbl)

    def get_color(self, lbl):
        if lbl in self.color_map:
            return self.color_map[lbl]
        else:
            self.color_map[lbl] = self.colors[self.color_counter]
            self.color_counter = (self.color_counter + 1) % len(self.colors)
            return self.color_map[lbl]

    def format_graphiz_node(self, node):
        s = []
        color = self.internal_color
        try:
            if node.purity() == 1.0:
                if hasattr(node,'pts') and len(node.pts) > 0:
                    if hasattr(node.pts[0][0], 'gt'):
                        color = self.get_color(node.pts[0][0].gt)
                    else:
                        color = self.get_color(node.pts[0][1])
                elif hasattr(node,'ms'):
                    color = self.get_color(node.ms[0].gt)
        except Exception:
            pass
        shape = 'egg' #if node.my_score != -np.inf and node.my_score < node.config.partition_threshold else 'polygon'
        if node.is_cluster_root:
            shape = 'polygon'
        if node.id in self.emphasis_1_nodes:
            shape = self.emphasis_1_shape
        if node.id in self.emphasis_2_nodes:
            shape = self.emphasis_2_shape
        if node.parent is None:
            if hasattr(node, 'my_score'):
                my_score = node.my_score
            else:
                my_score = 0.0
            s.append(
                '\n%s[shape=%s;style=filled;color=%s;label=<%s<BR/>%s<BR/>%s<BR/>>]'
                % (self.format_id(node.id), shape, color,
                   self.get_node_label(node), color,my_score))
            s.append(
                '\nROOTNODE[shape=star;style=filled;color=gold;label=<ROOT>]')
            s.append('\nROOTNODE->%s' % self.format_id(node.id))
        else:
            leaf_m = ""
            if hasattr(node,'pts') and len(node.pts) > 0:
                if hasattr(node.pts[0][0], 'mid'):
                    leaf_m = '%s|%s' % (node.pts[0][0].mid, node.pts[0][0].gt) \
                        if node.is_leaf() else ''
                else:
                    leaf_m = '%s|%s' % (node.pts[0][2], node.pts[0][1]) \
                        if node.is_leaf() else ''
            elif hasattr(node,'ms'):
                leaf_m = '%s|%s' % (node.ms[0].mid, node.ms[0].gt) \
                    if node.is_leaf() else ''
            if hasattr(node, 'my_score'):
                my_score = node.my_score
            else:
                my_score = 0.0
            s.append(
                '\n%s[shape=%s;style=filled;color=%s;label=<%s<BR/>%s<BR/>%s<BR/>%s<BR/>>]'
                % (self.format_id(node.id), shape, color,
                   self.get_node_label(node), color, leaf_m,node.my_score))
            s.append('\n%s->%s' % (self.format_id(node.parent.id),
                                   self.format_id(node.id)))
        return ''.join(s)

    def graphviz_tree(self, root, emphasis_1_nodes=None,emphasis_2_nodes=None):
        self.emphasis_1_nodes = emphasis_1_nodes if emphasis_1_nodes else []
        self.emphasis_2_nodes = emphasis_2_nodes if emphasis_2_nodes else []
        s = []
        s.append('digraph TreeStructure {\n')
        s.append(self.format_graphiz_node(root))
        for d in root.descendants():
            s.append(self.format_graphiz_node(d))
        s.append('\n}')
        return ''.join(s)

    @staticmethod
    def write_tree(filename, root, emphasis_1_nodes=None, emphasis_2_nodes=None):
        gv = Graphviz()
        tree = gv.graphviz_tree(root, emphasis_1_nodes,emphasis_2_nodes)
        with open(filename,'w') as fout:
            fout.write(tree)