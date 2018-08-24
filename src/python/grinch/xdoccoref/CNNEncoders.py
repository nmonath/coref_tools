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
import torch.nn as nn
import numpy as np

from grinch.xdoccoref.MentEncoder import MentEncoder

class CNNEncoder(MentEncoder):

    def __init__(self, config, vocab, type_key, token_dim, pos_dim,output_dim=1):
        super(CNNEncoder,self).__init__(config,vocab,type_key,use_pairwise=config.use_pairwise[type_key],output_dim=output_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.token_emb = nn.Embedding(vocab.size + 1, token_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(vocab.max_len * 2 + 5, pos_dim, padding_idx=0)
        self.pos2idx = dict()
        self.pos_mention = 1
        pos_counter = 2
        for i in range(vocab.max_len+1):
            self.pos2idx[i] = pos_counter
            pos_counter += 1
        for i in range(vocab.max_len+1):
            self.pos2idx[-i] = pos_counter
            pos_counter += 1
        self.conv1d = nn.Conv1d(token_dim, token_dim, 3, stride=1, padding=1)
        self.combine_word_pos = nn.Linear(token_dim + pos_dim, token_dim)
        self.nonlin = nn.Tanh()

    def encode_word_and_pos(self, x, x_pos, x_mask):
        embs = self.token_emb(x)
        embs_pos = self.pos_emb(x_pos)
        concat = torch.cat([embs,embs_pos],dim=2)
        projd = self.combine_word_pos(concat).transpose(2,1)
        convd = self.conv1d(projd)
        masked = convd * x_mask.unsqueeze(1)
        nonlind = self.nonlin(masked)
        max_poold = torch.nn.functional.avg_pool1d(nonlind,nonlind.size()[2]).squeeze(2)
        if self.training:
            max_poold = self.dropout(max_poold)
        return max_poold

    def batch_to_ints(self,batch):
        pass

    def relative_to_ment_batch_to_ints(self, batch):
        length = min(self.vocab.max_len, max([len(self.get_ids(ment)) for ment in batch]))
        ids_per_ment_padded = np.array([self.get_ids(ment)[:length] + [0 for _ in range(length-len(self.get_ids(ment)))] for ment in batch],dtype=np.long)
        position = np.zeros_like(ids_per_ment_padded)
        mask = (ids_per_ment_padded > 0).astype(np.float32)
        for i,ment in enumerate(batch):
            m_start = min(batch[i].sentence_token_offsets[0], length)
            m_end = min(batch[i].sentence_token_offsets[-1], length)
            for j in range(length):
                if j < m_start:
                    position[i, j] = self.pos2idx[j - m_start]
                elif m_start <= j <= m_end:
                    position[i, j] = self.pos_mention
                else:
                    position[i, j] = self.pos2idx[j - m_end]

        if self.config.use_cuda:
            ids_per_ment_padded = torch.cuda.LongTensor(ids_per_ment_padded)
            position = torch.cuda.LongTensor(position)
            mask = torch.cuda.FloatTensor(mask)
        else:
            ids_per_ment_padded = torch.LongTensor(ids_per_ment_padded)
            position = torch.LongTensor(position)
            mask = torch.FloatTensor(mask)
        return ids_per_ment_padded, position, length, mask

    def relative_to_start_batch_to_ints(self, batch):
        length = min(self.vocab.max_len, max([len(self.get_ids(ment)) for ment in batch]))
        ids_per_ment_padded = np.array([self.get_ids(ment)[:length] + [0 for _ in range(length-len(self.get_ids(ment)))] for ment in batch],dtype=np.long)
        position = np.zeros_like(ids_per_ment_padded)
        mask = (ids_per_ment_padded > 0).astype(np.float32)
        for i, ment in enumerate(batch):
            for j in range(length):
                position[i, j] = self.pos2idx[j]

        if self.config.use_cuda:
            ids_per_ment_padded = torch.cuda.LongTensor(ids_per_ment_padded)
            position = torch.cuda.LongTensor(position)
            mask = torch.cuda.FloatTensor(mask)
        else:
            ids_per_ment_padded = torch.LongTensor(ids_per_ment_padded)
            position = torch.LongTensor(position)
            mask = torch.FloatTensor(mask)
        return ids_per_ment_padded, position, length, mask

    def feat_ments(self, batch):
        ids, positions, max_len, mask = self.batch_to_ints(batch)
        # [0] is for the tuple. embs is B by L by D
        embs = self.encode_word_and_pos(ids, positions, mask)
        if self.config.use_cosine_sim:
            mean_norm = torch.norm(embs, dim=1).unsqueeze(1)
            normed = torch.div(embs, mean_norm)
            return normed
        else:
            return embs

    def get_ids(self,entMent):
        pass

class NameCNNEncoder(CNNEncoder):
    def __init__(self, config, vocab):
        super(NameCNNEncoder,self).__init__(config,vocab,'name',config.cnn_dims['name'],config.cnn_pos_dims['name'])
        if config.warm_start_name:
            from grinch.xdoccoref.PretrainedModels import build_ft
            ft = build_ft()
            self.warm_start_ft(ft)

    def batch_to_ints(self, batch):
        return self.relative_to_start_batch_to_ints(batch)

    def get_ids(self,entMent):
        return entMent.name_character_n_grams_ids

    def warm_start_ft(self,ft):
        for subw in self.vocab.w2id.keys():
            subw_id = ft.get_subword_id(subw)
            emb = ft.get_input_vector(subw_id)
            if self.vocab.w2id[subw] >= 4: # 4 is start of ids
                self.token_emb.weight.data[self.vocab.w2id[subw]] = torch.from_numpy(emb.astype(np.float32))
        self.token_emb.requires_grad = self.config.update_name

class ContextCNNEncoder(CNNEncoder):
    def __init__(self, config, vocab):
        super(ContextCNNEncoder,self).__init__(config,vocab,'context',config.cnn_dims['context'],config.cnn_pos_dims['context'])
        if config.warm_start_context:
            from grinch.xdoccoref.PretrainedModels import build_ft
            ft = build_ft()
            self.warm_start_ft(ft)
        if self.config.warm_start_context_glove:
            self.warm_start_glove(self.config.warm_start_context_glove)

    def warm_start_ft(self, ft):
        for w in self.vocab.w2id.keys():
            emb = ft.get_word_vector(w)
            if self.vocab.w2id[w] >= 4:
                self.token_emb.weight.data[self.vocab.w2id[w]] = torch.from_numpy(emb.astype(np.float32))
        self.token_emb.requires_grad = self.config.update_context

    def warm_start_glove(self,glove_file):
        with open(glove_file, 'r') as fin:
            for line in fin:
                splt = line.split(" ")
                w = splt[0]
                v = np.array([float(x) for x in splt[1:]])
                if w in self.vocab.w2id and self.vocab.w2id[w] >= 4:
                    print('Warm Start Glove %s' % w)
                    self.token_emb.weight.data[self.vocab.w2id[w]] = torch.from_numpy(
                        v.astype(np.float32))
        self.token_emb.requires_grad = self.config.update_context

    def batch_to_ints(self, batch):
        return self.relative_to_ment_batch_to_ints(batch)

    def get_ids(self, entMent):
        return entMent.context_ids