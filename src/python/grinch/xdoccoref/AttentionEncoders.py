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
from grinch.xdoccoref.MentEncoder import MentEncoder

class AttnEncoder(MentEncoder):

    def __init__(self, base_encoder:MentEncoder):
        super(AttnEncoder,self).__init__(base_encoder.config,base_encoder.vocab,
                                         base_encoder.type_key,use_pairwise=base_encoder.use_pairwise,output_dim=1)
        self.base_encoder = base_encoder
        self.embedding_dim = self.config.typed_dims[self.type_key]
        self.attender = torch.nn.Sequential(torch.nn.Linear(self.embedding_dim ,self.embedding_dim), torch.nn.Tanh(), torch.nn.Linear(self.embedding_dim ,1), torch.nn.Sigmoid())

    def feat_ments(self, batch):
        return self.init_ent_fv(batch)[0]

    def init_ent_fv(self,ment):
        base_emb = self.base_encoder.feat_ments([ment])
        attn_weights = self.attender(base_emb)
        output = attn_weights * base_emb
        return output.squeeze(),attn_weights.squeeze()

    def init_ent(self,ent,ment):
        fv,attn_weights = self.init_ent_fv(ment)
        ent.typed_sums[self.type_key] += fv
        ent.typed_centroids[self.type_key] += fv
        ent.typed_counts[self.type_key] += 1 #attn_weights
        if self.use_pairwise:
            ent.typed_mat[self.type_key][0, :] = fv
            ent.typed_mat_assign[self.type_key][0] = 1
        else:
            ent.typed_mat[self.type_key] = None
            ent.typed_mat_assign[self.type_key] = None
        ent.mat = [ment.ment().attributes]

