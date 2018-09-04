
import sys
import gzip

from grinch.xdoccoref.Vocab import TypedVocab
from grinch.xdoccoref.Load import load_json_mentions
import spacy
from spacy.tokens import Doc
from spacy.lang.en import English

nlp = spacy.load('en_core_web_sm')

tokenizer = English().Defaults.create_tokenizer(nlp)

def process_one(entMent):
    return tokenize(entMent,tokenizer)

# def process_one(entMent,typedVocab):
#     entMent.name_character_n_grams_ids = [typedVocab['name'][w] for w in entMent.name_spelling]
#     entMent.context_ids = [typedVocab['context'][w] for w in entMent.context_string.split(" ")]
#     return entMent

def tokenize(entMent,tokenizer):
    doc = tokenizer(entMent.context_string)
    start_offset = entMent.sentence_char_offset
    end_offset = entMent.sentence_char_offset + entMent.sentence_char_len
    sentence_offsets = [t.i for t in doc]
    tokens = [t.text for t in doc]
    entMent.sentence_tokens = tokens
    entMent.sentence_token_offsets = sentence_offsets

    return entMent




if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with gzip.open(output_file,'wt') as fout:
        for entMent in load_json_mentions(input_file):
            if len(entMent.context_string.strip()) > 0:
                entMentP = process_one(entMent)
                if entMentP:
                    fout.write("%s\n" % entMentP.to_json())
                else:
                    print("ERROR processing %s %s %s" % (entMent.mid,entMent.name_spelling,entMent.context_string))
                    # trying to add one.
                    entMent.sentence_char_len += 1
                    entMentP = process_one(entMent)
                    if entMentP:
                        fout.write("%s\n" % entMentP.to_json())
                    else:
                        print(
                            "ERROR processing %s %s %s" % (entMent.mid, entMent.name_spelling, entMent.context_string))
                        # trying to add one.
                        entMent.sentence_char_len -= 2
                        entMentP = process_one(entMent)
                        if entMentP:
                            fout.write("%s\n" % entMentP.to_json())
                        else:
                            print("FATAL ERROR processing %s %s %s" % (
                            entMent.mid, entMent.name_spelling, entMent.context_string))
                            # trying to add one.
