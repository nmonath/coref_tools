
import sys
import gzip

from grinch.xdoccoref.Vocab import TypedVocab
from grinch.xdoccoref.Load import load_json_mentions
from grinch.xdoccoref.PretrainedModels import build_ft
from allennlp.modules.elmo import Elmo, batch_to_ids

import numpy as np

weight_file = "resources/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
options_file = "resources/elmo_2x1024_128_2048cnn_1xhighway_options.json"
elmo = Elmo(options_file, weight_file, 2, dropout=0)


# embeddings['elmo_representations'] is length two list of tensors.
# Each element contains one layer of ELMo representations with shape
# (2, 3, 1024).
#   2    - the batch size
#   3    - the sequence length of the batch
#   1024 - the length of each ELMo vector


def get_elmo_embeddings(entMents):
    sentences = [entMent.sentence_tokens for entMent in entMents]
    character_ids = batch_to_ids(sentences)
    embeddings = elmo(character_ids)
    representations = embeddings['elmo_representations'][0].data.numpy()
    for idx,entMent in enumerate(entMents):
        span_rep = representations[idx,entMent.sentence_token_offsets[0]:entMent.sentence_token_offsets[-1]+1,:].mean(axis=0)
        yield entMent.mid,"\t".join([str(x) for x in span_rep])


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with gzip.open(output_file,'wt') as fout:
        group = []
        for entMent in load_json_mentions(input_file):
            group.append(entMent)
            if len(group) == 100:
                print('processing group')
                for mid,vec in get_elmo_embeddings(group):
                    fout.write("%s\t%s\n" % (mid,vec))
                    fout.flush()
                group = []
        if group:
            print('processing group')
            for mid, vec in get_elmo_embeddings(group):
                fout.write("%s\t%s\n" % (mid, vec))
                fout.flush()