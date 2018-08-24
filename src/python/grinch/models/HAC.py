"""
Copyright (C) 2018 IBM Corporation.
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
from grinch.models.Grinch import Grinch

from heapq import heappush,heappop
import time

class EHAC(Grinch):
    def __init__(self,config,sim_model):
        super(EHAC,self).__init__()
        self.config = config
        self.sim_model = sim_model
        self.num_computations = 0
        self.pts = []
        self.roots = []         # The current roots.
        self.pair_to_pw = dict()
        self.sorted_mergers = []
        self.time_so_far = 0

    def prime(self,dataset):
        self.pts = dataset
        for d in dataset:
            self.roots.append(self.node_from_pt(d))
            self.roots[-1].cluster_marker = True
        all_pairs = []
        all_scores = []
        # Get all pairwise scores first.
        num_completed = 0
        num_pairs = int(len(dataset) * (len(dataset) - 1) / 2.0)
        for i in range(len(self.roots)-1):
            scores = self.sim_model.one_against_many_score(self.roots[i].ent, [x.ent for x in self.roots[i + 1:]])
            all_scores.append(scores)
            for j in range(i + 1, len(self.roots)):
                torch_score = scores[j - (i+1)]
                np_score = torch_score.cpu().item()
                n1,n2 = self.roots[i],self.roots[j]
                all_pairs.append((n1,n2))
                self.pair_to_pw[(n1, n2)] = np_score
                heappush(self.sorted_mergers, (-np_score,n1, n2, torch_score))
                num_completed += 1
                if num_completed % 100 == 0:
                    self.logger.info('[prime] Processed %s of %s distances' % (num_completed,num_pairs))
        return all_pairs,torch.cat(all_scores)

    def valid_agglom(self,n1,n2):
        return n1.parent is None and n2.parent is None

    def next_valid_agglom(self):
        res = None
        while self.sorted_mergers and (res is None or not self.valid_agglom(res[1],res[2])):
            res = heappop(self.sorted_mergers)
        return res

    def take_one_step(self):
        next_step = self.next_valid_agglom()
        if next_step:
            np_score, n1, n2, torch_score = next_step
            assert n1.root() != n2.root()
            assert n1 in self.roots
            assert n2 in self.roots
            assert n1.root() == n1
            assert n2.root() == n2
            new_node = self.node_from_nodes(n1,n2)
            n1.parent = new_node
            n2.parent = new_node
            new_node.children = [n1,n2]
            new_node.update_from_children()
            self.roots.remove(n1)
            self.roots.remove(n2)
            if len(self.roots) > 0:
                added_pairs, added_scores = self.add_scores_with_entity(new_node)
                self.roots.append(new_node)
                return new_node, added_pairs, added_scores
            else:
                self.roots.append(new_node)
                return new_node, None, None
        return None,None,None

    def add_scores_with_entity(self, new_entity):
        all_pairs = []
        num_completed = 0
        self.logger.info('[add_scores_with_entity] Comparing %s to the %s-element frontier' % (new_entity.id, len(self.roots)))
        all_scores = self.sim_model.one_against_many_score(new_entity.ent, [x.ent for x in self.roots])
        for j in range(len(self.roots)):
            torch_score = all_scores[j]
            np_score = torch_score.cpu().item()
            n1, n2 = new_entity, self.roots[j]
            all_pairs.append((n1, n2))
            self.pair_to_pw[(n1, n2)] = np_score
            heappush(self.sorted_mergers, (-np_score, n1, n2, torch_score))
            num_completed += 1
            if num_completed % 10 == 0:
                self.logger.info('[add_scores_with_entity] Processed %s of %s distances' % (num_completed, len(self.roots)))
        return all_pairs,all_scores

    def build_dendrogram(self,dataset):
        self.logger.info('[build_dendrogram] EHAC on %s element dataset' % len(dataset))
        self.prime(dataset)
        start_time = time.time()
        go_on = True
        per_move_times = []
        while go_on:
            pt_start = time.time()
            nn,np,sc = self.take_one_step()
            pt_ent = time.time()
            this_pt_time = pt_ent - pt_start
            per_move_times.append(pt_ent-pt_start)
            if len(self.roots) % 100 == 0:
                since_prime = time.time() - start_time
                self.logger.info('[NSWEHAC] Num in Forest %s || So Far %s || Since Prime %s || Last %s || Avg %s || Max %s'
                                 % (len(self.roots), since_prime + self.time_so_far,since_prime,this_pt_time,
                                    sum(per_move_times)/len(per_move_times),max(per_move_times)))
            go_on = np is not None
        assert len(self.roots) == 1
        self.root = self.roots[0]
        end_time = time.time()
        self.time_so_far += end_time-start_time
        return self.roots[0]

class NSWEHAC(EHAC):
    def __init__(self,config,sim_model):
        super(NSWEHAC,self).__init__(config,sim_model)
        self.already_agglommed = set()
        self.time_so_far = 0

    def prime(self,dataset):
        self.logger.info('[NSWEHAC] prime priority queue')
        # Steps:
        # 1. Build NSW
        # 2. Add all edges from NSW & K-nn edges.
        prime_start = time.time()
        time_per_point = []
        self.pts = dataset
        for i,d in enumerate(dataset):
            self.roots.append(self.node_from_pt(d))
            self.roots[-1].cluster_marker = True
            pt_st = time.time()
            _ = self.knn_and_insert(self.roots[-1], self.k())
            pt_end = time.time()
            this_pt_time = pt_end-pt_st
            time_per_point.append(this_pt_time)
            if i % 100 == 0:
                curr_time = time.time()
                self.logger.info('[NSWEHAC] Added %s of %s to NSW || Time so far: %s || this pt time %s || Avg Time %s'
                                 % (i,len(dataset),curr_time-prime_start,this_pt_time,
                                    sum(time_per_point)/len(time_per_point)))
        self.logger.info('[NSWEHAC] Added all to NSW || Time so far: %s || Avg Time %s'
                         % (curr_time - prime_start,
                            sum(time_per_point) / len(time_per_point)))
        edges = dict()
        edge_start = time.time()
        time_per_point = []
        for i,r in enumerate(self.roots):
            e_start = time.time()
            # returns <Torch Num, Node>
            knn = self.cknn(r,self.k(),[r])
            for torch_score,nn in knn:
                if (r,nn) not in edges and (nn,r) not in edges:
                    edges[(r,nn)] = torch_score
                    np_score = torch_score.item()
                    heappush(self.sorted_mergers, (-np_score, r, nn, torch_score))
            e_end = time.time()
            this_pt_time = e_end - e_start
            time_per_point.append(this_pt_time)
            if i % 100 == 0:
                curr_time = time.time()
                self.logger.info('[NSWEHAC] added edges for %s of %s || Time so far: %s || this pt time %s || Avg time %s' % (i,len(self.roots),
                                 curr_time-prime_start,this_pt_time,
                                    sum(time_per_point)/len(time_per_point)))
        all_scores = torch.zeros(len(edges))
        if self.config.use_cuda:
            all_scores = all_scores.cuda()
        edge_list = []
        for i,pair in enumerate(edges):
            all_scores[i] = edges[pair]
            edge_list.append(pair)
        self.time_so_far += time.time() - prime_start
        self.logger.info('[NSWEHAC] Done priming, time so far: %s' % self.time_so_far)
        return edge_list,all_scores

    def add_scores_with_entity(self, new_entity):
        all_pairs = []
        num_completed = 0
        if len(self.roots) % 100 == 0:
            self.logger.info('[add_scores_with_entity] Comparing %s to the %s-element frontier' % (new_entity.id, len(self.roots)))
        cknn = self.cknn_and_insert(new_entity,self.k(),self.already_agglommed)
        scores = torch.zeros(len(cknn))
        if self.config.use_cuda:
            scores = scores.cuda()
        for i,torch_score_nn in enumerate(cknn):
            torch_score = torch_score_nn[0]
            nn = torch_score_nn[1]
            assert nn != new_entity
            assert nn.parent is None
            np_score = torch_score.item()
            heappush(self.sorted_mergers, (-np_score, new_entity, nn, torch_score))
            all_pairs.append((new_entity,nn))
            scores[i] = torch_score
        return all_pairs,scores

    def take_one_step(self):
        next_step = self.next_valid_agglom()
        if next_step:
            np_score, n1, n2, torch_score = next_step
            assert n1.root() != n2.root()
            assert n1 in self.roots
            assert n2 in self.roots
            assert n1.root() == n1
            assert n2.root() == n2
            new_node = self.node_from_nodes(n1,n2)
            n1.parent = new_node
            n2.parent = new_node
            new_node.children = [n1,n2]
            new_node.update_from_children()
            self.roots.remove(n1)
            self.roots.remove(n2)
            self.already_agglommed.add(n1)
            self.already_agglommed.add(n2)
            if len(self.roots) > 0:
                added_pairs, added_scores = self.add_scores_with_entity(new_node)
                self.roots.append(new_node)
                return new_node, added_pairs, added_scores
            else:
                self.roots.append(new_node)
                return new_node, None, None
        return None,None,None

    def randomly_connect_the_rest(self):
        while len(self.roots) != 1:
            n1 = self.roots.pop(0)
            n2 = self.roots.pop(0)
            new_node = self.node_from_nodes(n1, n2)
            n1.parent = new_node
            n2.parent = new_node
            new_node.children = [n1, n2]
            self.roots.append(new_node)

    def next_valid_agglom(self):
        res = None
        while self.sorted_mergers and (res is None or not self.valid_agglom(res[1],res[2])):
            res = heappop(self.sorted_mergers)
        # it could be the case that the last thing popped of sorted_mergers was not a valid move.
        if len(self.sorted_mergers) == 0 and res is not None and not self.valid_agglom(res[1],res[2]):
            res = None
        return res

    def build_dendrogram(self,dataset):
        self.logger.info('[build_dendrogram] EHAC on %s element dataset' % len(dataset))
        self.prime(dataset)
        start_time = time.time()
        go_on = True
        per_move_times = []
        while go_on:
            pt_start = time.time()
            nn,np,sc = self.take_one_step()
            pt_ent = time.time()
            this_pt_time = pt_ent - pt_start
            per_move_times.append(pt_ent-pt_start)
            if len(self.roots) % 100 == 0:
                since_prime = time.time() - start_time
                self.logger.info('[NSWEHAC] Num in Forest %s || So Far %s || Since Prime %s || Last %s || Avg %s || Max %s'
                                 % (len(self.roots), since_prime + self.time_so_far,since_prime,this_pt_time,
                                    sum(per_move_times)/len(per_move_times),max(per_move_times)))
            go_on = np is not None
        self.randomly_connect_the_rest()
        assert len(self.roots) == 1
        self.root = self.roots[0]
        end_time = time.time()
        self.time_so_far += end_time-start_time
        return self.roots[0]