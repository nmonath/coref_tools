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

from coref.models.core.F1Node import F1Node


class GNode(F1Node):
    """A node that houses a set of mentions (and attribute projections)."""
    def __init__(self, pts, e_model, point_counter=None, mid=None,
                 my_score_f=None,grinch=None):
        """Init.

        Args:
            pts - a list of data items (np.array, label, id).
            aproj - an attribute projection.
            point_counter - number of points under this node
        """
        super().__init__()
        self.pts = pts
        self.e_model = e_model
        if point_counter is not None:
            self.point_counter = point_counter
        else:
            self.point_counter = len(self.pts)
        self.nsw_node = None
        self.my_score_f = my_score_f
        self.e_model.gnode = self
        self.grinch = grinch

    def lazy_my_score(self):
        if self.my_score is None:
            self.my_score = self.my_score_f(self)
            self.grinch.num_computations += 1
        return self.my_score

    def debug_string(self):
        raise Exception("No debug method implemented for GNode.")

    def split_down(self, other, e_model, new_my_score=None):
        """Make other a sibling of self under a new parent.

        Create a new internal node that is the parent of self and other.

        Args:
            other - a GNode
            e_model - an entity model
            new_my_score - the score of the entity model (optional)

        Returns:
            The new internal node (parent of self).
        """
        # new_internal = GNode(self.pts + other.pts, e_model,
        #                      point_counter=self.point_counter + other.point_counter)
        new_internal = GNode([], e_model,
                             point_counter=self.point_counter + other.point_counter,
                             my_score_f=self.my_score_f,grinch=self.grinch)

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

    def update_aps(self, new_aproj, model=None):
        """Update self with new_aproj.

        I store an entity representation. new_aproj is a different entity
        representation that I should update myself with. Update recursively up
        the tree.

        Args:
            new_aproj - an entity model.

        Returns:
            Nothing.
        """
        # Entity scores are no longer updated in this method
        # updated_nodes = []
        # updated_aps = []
        curr = self
        while curr:
            # Update the point counter
            curr.update_from_children()
            # curr.e_model.update(new_aproj)
            # curr.my_score = curr.e_model.my_e_score()
            curr = curr.parent

    def update_from_children(self):
        """Update my model from my children.

        Args:
            None.

        Returns:
            Nothing.
        """
        if self.children:
            c1, c2 = self.children[0], self.children[1]
            self.point_counter = c1.point_counter + c2.point_counter
            self.e_model = c1.e_model.hallucinate_merge(c2.e_model)
            self.e_model.gnode = self
            # self.my_score = c1.e_model.e_score(c2.e_model)
            self.my_score = None

    def graft_to_me(self, other, new_aproj=None, new_my_score=None):
        """Graft other to me under new parent.

        0. Other must have parent. LCA(self,other) != self or other.
        1. Detach other from other's parent
        2. Connect other as a child of its gradmother. (Replace other's parent with other's sibling.)
        3. Then do me.split_down(other)
        :param other:
        :return:
        """

        assert other.parent
        # lca = self.lca(other)
        # assert lca != self
        # assert lca != other
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
        # other_parent.nsw_node.delete()
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

        return self.split_down(other,
                               self.e_model.hallucinate_merge(other.e_model),
                               None)