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

class GraphvizNSW(object):
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
        lbl.append(self.format_id(node.v.id))
        lbl.append('<BR/>')
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
            if node.v.purity() == 1.0:
                if node.v.pts:
                    color = self.get_color(node.v.pts[0][0].gt)
        except Exception:
            pass
        shape = 'egg' #if node.my_score != -np.inf and node.my_score < node.config.partition_threshold else 'polygon'
        if node.is_cluster_root:
            shape = 'polygon'
        if node.id in self.emphasis_1_nodes:
            shape = self.emphasis_1_shape
        if node.id in self.emphasis_2_nodes:
            shape = self.emphasis_2_shape
        leaf_m = '%s|%s' % (node.v.pts[0][0].mid, node.v.pts[0][0].gt) \
            if node.v.is_leaf() and node.v.pts else ''
        s.append(
            '\n%s[shape=%s;style=filled;color=%s;label=<%s<BR/>%s<BR/>%s<BR/>%s<BR/>>]'
            % (self.format_id(node.id), shape, color,
               self.get_node_label(node), color, leaf_m,node.v.my_score))
        for n in node.neighbors:
            # Use lexicographic ordering of node ids to prevent duplicate edges
            if node.id < n.id:
                s.append('\n%s--%s' % (self.format_id(node.id),
                                       self.format_id(n.id)))
        return ''.join(s)

    def graphviz_nsw(self, nsw, emphasis_1_nodes=None,emphasis_2_nodes=None):
        self.emphasis_1_nodes = emphasis_1_nodes if emphasis_1_nodes else []
        self.emphasis_2_nodes = emphasis_2_nodes if emphasis_2_nodes else []
        s = []
        s.append('graph NSW {\n')
        for n in nsw.nodes:
            s.append(self.format_graphiz_node(n))
        s.append('\n}')
        return ''.join(s)

    @staticmethod
    def write_nsw(filename, root, emphasis_1_nodes=None, emphasis_2_nodes=None):
        gv = GraphvizNSW()
        tree = gv.graphviz_nsw(root, emphasis_1_nodes,emphasis_2_nodes)
        with open(filename,'w') as fout:
            fout.write(tree)