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

from coref.models.core.F1Node import F1Node
from coref.models.core.Ment import Ment


class MentNode(F1Node):
    """A node that houses a set of mentions (and attribute projections)."""
    def __init__(self, pts, aproj=None, point_counter=None, mid=None):
        """Init.

        Args:
            pts - a list of data items (mention, label, id).
            aproj - an attribute projection.
            point_counter -
        """
        super().__init__()
        self.pts = pts
        self.as_ment = Ment(aproj, aproj, mid=mid)
        if point_counter is not None:
            self.point_counter = point_counter
        else:
            self.point_counter = len(self.pts)
        self.nsw_node = None

    def debug_string(self):
        if 'my_pw' in self.as_ment.attributes.aproj_local_debug:
            return self.as_ment.attributes.aproj_local_debug['my_pw']
        else:
            return "no debug"

    def subtree_memory_usage(self):
        return 0
        # bytes = 0
        # c = self
        # q = [c]
        # while q:
        #     c = q.pop()
        #     if c.as_ment.attributes is not None:
        #         bytes += c.as_ment.attributes.memory_usage()
        #     q.extend(c.children)
        # return bytes

    def split_down(self, other,new_aproj=None,new_my_score=None):
        new_internal = MentNode(self.pts + other.pts,aproj=new_aproj,
                                point_counter=self.point_counter + other.point_counter)

        new_internal.parent = self.parent
        if new_internal.parent:
            new_internal.parent.children.remove(self)
            new_internal.parent.children.append(new_internal)
        new_internal.children.append(self)
        new_internal.children.append(other)
        self.parent = new_internal
        other.parent = new_internal
        if new_my_score is not None:
            new_internal.my_score = new_my_score
        return new_internal

    def update_aps(self,new_aproj,model=None):
        # Entity scores are no longer updated in this method
        # updated_nodes = []
        # updated_aps = []
        curr = self
        while curr:
            # Update the point counter
            curr.point_counter += 1
            # Update the attribute projections
            curr.as_ment.attributes.update(new_aproj,model=model)
            # if model:
            #     updated_nodes.append(curr)
            #     updated_aps.append(curr.as_ment.attributes)
            curr = curr.parent
        # if model:
        #     assert False, "This should not be used."
        #     fvs = torch.zeros(len(updated_aps),model.e_concat_dim)
        #     for i in range(len(updated_aps)):
        #         fvs[i,:] = model.e_extract_features(updated_aps[i]).data
        #
        #     e_scores = model.e_score_mat(fvs)
        #     for i in range(len(updated_aps)):
        #         updated_nodes[i].my_score = e_scores[i].data.numpy()[0]

    def graft_to_me(self, other,new_aproj=None,new_my_score=None):
        """ Graft other to me under new parent.

        0. Other must have parent. LCA(self,other) != self or other.
        1. Detach other from other's parent
        2. Connect other as a child of its gradmother. (Replace other's parent with other's sibling.)
        3. Then do me.split_down(other)
        :param other: 
        :return: 
        """

        assert other.parent
        lca = self.lca(other)
        assert lca != self and lca != other
        assert self not in other.siblings()
        # Checked by LCA
        # assert self.parent

        other_parent = other.parent
        other_gp = other.parent.parent
        other_sib = other.siblings()[0]

        other_sib.parent = other_gp
        if other_gp:
            other_gp.children.remove(other_parent)
            other_gp.children.append(other_sib)

        # If we want to keep around deleted nodes, we would want to keep this edge:
        other_parent.parent = None
        other_parent.children = []
        other_parent.deleted = True
        other_parent.nsw_node.delete()
        other_parent.pts = None

        print('self %s %s ' % (self, self.id))
        print('self.parent %s %s ' % (self.parent, self.parent.id if self.parent else "None"))
        print('self.children %s %s ' % (self.children, [x.id for x in self.children]))
        if self.parent:
            print('self.parent.children %s %s ' % (self.parent.children, [x.id for x in self.parent.children]))

        print('other %s %s ' % (other, other.id))
        print('other.parent %s %s ' % (other.parent, other.parent.id if other.parent else "None"))
        print('other.children %s %s ' % (other.children, [x.id for x in other.children]))
        if other.parent:
            print('other.parent.children %s %s ' % (other.parent.children, [x.id for x in other.parent.children]))

        return self.split_down(other,new_aproj=new_aproj,new_my_score=new_my_score)