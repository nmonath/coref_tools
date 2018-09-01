
from collections import defaultdict
import sys
from grinch.xdoccoref.Load import load_json_mentions
from grinch.xdoccoref.Vocab import TypedVocab,Vocab

def update_counts(entMent,typed_counts):
    for name_char in entMent.name_spelling:
        typed_counts['name'][name_char] += 1
    for context_word in entMent.context_string.split(" "):
        typed_counts['context'][context_word] += 1


def process(entMents):
    typed_counts = {'name': defaultdict(int), 'context': defaultdict(int)}
    for m in entMents:
        update_counts(m,typed_counts)
    return typed_counts


def finalize(typed_counts, min_count=5):
    # Add padding and start / end tokens
    base_name = {"PAD": 0, "<OOV>": 1, "<SENT_START>": 2, "<SENT_END>": 3}
    sorted_name = sorted(typed_counts['name'].items(),key=lambda x:(x[1],x[0]),reverse=True)
    name_start = 4
    name_id2w = dict()
    for name,count in sorted_name:
        if count > min_count:
            base_name[name] = name_start
            name_id2w[name_start] = name
            name_start += 1

    base_context = {"PAD": 0, "<OOV>": 1, "<SENT_START>": 2, "<SENT_END>": 3}
    sorted_context = sorted(typed_counts['context'].items(),key=lambda x:(x[1],x[0]),reverse=True)
    context_start = 4
    context_id2w = dict()
    for context,count in sorted_context:
        if count > min_count:
            base_context[context] = context_start
            context_id2w[context_start] = context
            context_start += 1

    name_vocab = Vocab(self_dict={'w2id': base_name, 'id2w': name_id2w, 'size': len(name_id2w), 'max_len': 300})
    context_vocab = Vocab(self_dict={'w2id': base_context, 'id2w': context_id2w, 'size': len(context_id2w), 'max_len': 30})

    typed_vocab = TypedVocab()
    typed_vocab['name'] = name_vocab
    typed_vocab['context'] = context_vocab
    return typed_vocab



if __name__ == "__main__":

    vocab_file = sys.argv[1]
    outfile = sys.argv[2]
    min_count =5
    msl_name = 300
    msl_context = 30

    mentions = load_json_mentions(vocab_file)
    typed_counts = process(mentions)
    tv = finalize(typed_counts)
    with open(outfile,'w') as fout:
        fout.write(tv.to_json())


