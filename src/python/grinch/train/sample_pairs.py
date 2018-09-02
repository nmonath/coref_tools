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

import datetime,os,random,sys,gzip,grinch

from grinch.util.IO import mkdir_p
from collections import defaultdict

from grinch.util.Config import Config

if __name__ == "__main__":

    def sample(mentions,num_pos,num_neg,random, max_num_pos_per_cluster,num_hard_negative,output_pairs=True):
        mention_to_entity = dict()
        entity_to_mention = defaultdict(list)
        cluster_order = []
        surfaceform_to_entity = defaultdict(set)
        token_to_entity = defaultdict(set)
        non_singletons = set()
        def update_for_ment(entMent):
            mention_to_entity[entMent] = entMent.gt
            entity_to_mention[entMent.gt].append(entMent)
            surfaceform_to_entity[entMent.name_spelling].add(entMent.gt)
            for s in entMent.name_spelling.split(" "):
                if len(s) > 2 and s.lower() not in grinch.xdoccoref.stopwords:
                    token_to_entity[s].add(entMent.gt)
            if len(entity_to_mention[entMent.gt]) > 1:
                non_singletons.add(entMent.gt)

        def random_int(upper,notthis):
            s = random.randint(0,upper-1)
            if s == notthis:
                s += 1
                s %= upper
            return s

        def sample_tok_negative(entMent,num):
            entities = set([x for t in entMent.name_spelling.split(" ") if t in token_to_entity for x in token_to_entity[t]])
            this_entity = entMent.gt
            other_entities = list(entities.difference({this_entity}))
            if other_entities:
                for i in range(num):
                    e = random.sample(other_entities,1)[0]
                    if len(entity_to_mention[e]) >= 1:
                        m = random.sample(entity_to_mention[e],1)[0]
                        yield m

        def sample_hard_negative(entMent,num):
            entities = surfaceform_to_entity[entMent.name_spelling]
            this_entity = entMent.gt
            other_entities = list(entities.difference({this_entity}))
            if other_entities:
                for i in range(num):
                    e = random.sample(other_entities,1)[0]
                    if len(entity_to_mention[e]) >= 1:
                        m = random.sample(entity_to_mention[e],1)[0]
                        yield m

        for idx, entMent in enumerate(mentions):
            if idx % 100 == 0:
                sys.stdout.write('\rprocessed %s mentions' % idx)
            update_for_ment(entMent)
            cluster_order = list(entity_to_mention.keys())
        for cidx,entity in enumerate(cluster_order):
            if len(entity_to_mention[entity]) > 0:
                if cidx % 100 == 0 :
                    sys.stdout.write('\rprocessed %s entities' % cidx)
                c = entity_to_mention[entity]
                c = [entMent  for entMent in c if len(surfaceform_to_entity[entMent.name_spelling]) > 1 or any(len(token_to_entity[x]) > 1 for x in entMent.name_spelling.split(" ") if x in token_to_entity)]
                if len(c) > 0:
                    source_pts = random.sample(c, min(max_num_pos_per_cluster, len(c)))
                    if len(source_pts) < max_num_pos_per_cluster:
                        c = entity_to_mention[entity]
                        source_pts.extend(random.sample(c, min(max_num_pos_per_cluster, len(c) - len(source_pts))))
                else:
                    c = entity_to_mention[entity]
                    source_pts = random.sample(c, min(max_num_pos_per_cluster, len(c)))

                for midx, m in enumerate(source_pts):
                    for i_pos in range(min(num_pos, len(source_pts))):
                        rand_pos = c[random_int(len(source_pts), midx)]
                        if output_pairs:
                            yield  ((m[2], rand_pos[2], 1, m.name_spelling,m.gt, rand_pos.name_spelling,rand_pos.gt,))
                        for i_neg in range(min(min(num_pos, len(source_pts)), num_neg)):
                            rand_neg_c = entity_to_mention[cluster_order[random_int(len(cluster_order), cidx)]]
                            rand_neg_m = rand_neg_c[random_int(len(rand_neg_c), -1)]
                            assert m[1] != rand_neg_m[1]
                            if output_pairs:
                                yield ((m[2], rand_neg_m[2], 0, m.name_spelling,m.gt,rand_neg_m.name_spelling,rand_neg_m.gt))
                            else:
                                yield ((m[2], rand_pos[2], rand_neg_m[2], m.name_spelling, m.gt, rand_pos.name_spelling,
                                        rand_pos.gt, rand_neg_m.name_spelling, rand_neg_m.gt))

                        for hard_neg in sample_hard_negative(m,num_hard_negative):
                            if output_pairs:
                                yield ((m[2], hard_neg[2], 0, m.name_spelling,m.gt,hard_neg.name_spelling,hard_neg.gt))
                            else:
                                yield ((m[2], rand_pos[2], hard_neg[2], m.name_spelling, m.gt, rand_pos.name_spelling,
                                        rand_pos.gt, hard_neg.name_spelling, hard_neg.gt))
                        for hard_neg in sample_tok_negative(m, num_hard_negative):
                            if output_pairs:
                                yield ((m[2], hard_neg[2], 0, m.name_spelling,m.gt,hard_neg.name_spelling,hard_neg.gt))
                            else:
                                yield ((m[2], rand_pos[2], hard_neg[2], m.name_spelling, m.gt, rand_pos.name_spelling,
                                        rand_pos.gt, hard_neg.name_spelling, hard_neg.gt))

    # load the mentions
    config = Config(sys.argv[1])
    ts = sys.argv[2] if len(sys.argv) > 2 else None

    now = datetime.datetime.now()
    if ts is None:
        ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second)

    config.experiment_out_dir = os.path.join(
        config.experiment_out_dir, config.dataset_name,
        config.model_name, config.clustering_scheme, ts)
    mkdir_p( config.experiment_out_dir )
    print('Output Dir: %s ' % config.experiment_out_dir)

    # Load data from this file
    mention_file = config.train_files[0]
    from grinch.xdoccoref.Load import load_mentions_from_file
    from grinch.xdoccoref.PretrainedModels import build_elmo,build_ft
    ft = build_ft()
    elmo = build_elmo()
    mentions = [m for m in load_mentions_from_file(mention_file,ft,elmo,use_cuda=config.use_cuda)]
    with gzip.open(os.path.join(config.experiment_out_dir, "pairs.tsv.anno.gz"), 'wt') as fanno:
        with gzip.open(os.path.join(config.experiment_out_dir, "pairs.tsv.gz"), 'wt') as fout:
            for idx,p in enumerate(sample(mentions,config.num_pos_samples_per_pt,
                                          config.num_rand_neg_samples_per_pt,
                                          random.Random(config.random_seed),
                                          config.num_max_num_positives_per_cluster,config.num_hard_neg_samples_per_pt, config.produce_sample_pairs)):
                fout.write('%s\t%s\t%s\n' % p[0:3])
                fanno.write("%s\n" % "\t".join([str(x) for x in p]))
                if idx % 100 == 0:
                    sys.stdout.write('\rWrote %s pairs' % idx)
                    fout.flush()
