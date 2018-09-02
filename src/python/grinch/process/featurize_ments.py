
import sys
import gzip

from grinch.xdoccoref.Vocab import TypedVocab
from grinch.xdoccoref.Load import load_json_mentions
from grinch.xdoccoref.PretrainedModels import build_ft

def process_one(entMent,typedVocab,ft):
    entMent.name_character_n_grams = ft.get_subwords(entMent.name_spelling)[0]
    return entMent


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    vocab_file = sys.argv[3]

    tv = TypedVocab(vocab_file)

    ft = build_ft()

    with gzip.open(output_file,'wt') as fout:
        for entMent in load_json_mentions(input_file):
            entMent = process_one(entMent,tv,ft)
            fout.write("%s\n" % entMent.to_json())