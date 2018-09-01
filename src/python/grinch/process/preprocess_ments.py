
import sys
import gzip

from grinch.xdoccoref.Vocab import TypedVocab
from grinch.xdoccoref.Load import load_json_mentions

def process_one(entMent,typedVocab):
    entMent.name_character_n_grams_ids = [typedVocab['name'][w] for w in entMent.name_spelling]
    entMent.context_ids = [typedVocab['context'][w] for w in entMent.context_string.split(" ")]
    return entMent


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    vocab_file = sys.argv[3]

    tv = TypedVocab(vocab_file)

    with gzip.open(output_file,'wt') as fout:
        for entMent in load_json_mentions(input_file):
            entMent = process_one(entMent,tv)
            fout.write("%s" % entMent.to_json())