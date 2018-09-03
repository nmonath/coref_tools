
from grinch.xdoccoref.Core import EntMent
import sys
import gzip
from grinch.util.IO import lines

class TSVMention(object):

    def __init__(self,line):
        splt = line.split("\t")
        self.mid = splt[0]
        self.docId = splt[1]
        self.url = splt[2]
        self.domain_name = splt[3]
        self.mention_spelling = splt[4]
        self.gt_raw = splt[5]
        self.gt= splt[6] # resolved redirects
        self.left_context = splt[7]
        self.middle_context = splt[8]
        self.right_context = splt[9]
        self.context = " ".join([self.left_context,self.middle_context,self.right_context])

    def to_ent_ment(self):
        entMent = EntMent(self.mid,self.gt)
        entMent.name_spelling = self.mention_spelling
        entMent.context_string = self.context
        # compute the token offsets:
        mention_start = len(self.left_context.split(" "))
        mention_end = mention_start + len(self.middle_context.split(" "))
        entMent.sentence_token_offsets = [x for x in range(mention_start,mention_end)]
        entMent.sentence_char_offset = len(self.left_context) + (1 if len(self.left_context) > 0 else 0)
        entMent.sentence_char_len = len(self.middle_context)
        return entMent


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with gzip.open(output_file,'wt') as fout:
        for idx,line in enumerate(lines(input_file)):
            if idx % 1000 == 0:
                sys.stdout.write("\rProcessed %s lines" % idx)
                sys.stdout.flush()
            tsvMention = TSVMention(line.strip('\n'))
            json_mention = tsvMention.to_ent_ment().to_json()
            fout.write('%s\n' %json_mention)
