
from grinch.xdoccoref.Core import EntMent
import sys
import gzip
from grinch.util.IO import lines

class TSVMention(object):

    def __init__(self,line):
        splt = line.split("\t")
        if len(splt) != 4:
            print('ERROR READING LINE %s' % line)

        self.gt = splt[0]
        self.mid = splt[1]
        self.mention_spelling = splt[2]
        self.context = splt[3].replace("  ", " ").replace("á","a").replace("  ", " ")

    def to_ent_ment(self):
        entMent = EntMent(self.mid,self.gt)
        entMent.name_spelling = self.mention_spelling
        entMent.context_string = self.context
        # compute the token offsets:
        entMent.sentence_char_offset = self.context.lower().find(self.mention_spelling.lower())
        if entMent.sentence_char_offset == -1:
            print('couldnt find the mention adding to the beg of the string')
            print(self.mention_spelling)
            print(self.context)
            self.context = self.mention_spelling + ' ' + self.context
            entMent.context_string = self.context
            entMent.sentence_char_offset = self.context.lower().find(self.mention_spelling.lower())
        entMent.sentence_char_len = 0 if entMent.sentence_char_offset == -1 else len(self.mention_spelling)
        return entMent


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with gzip.open(output_file,'wt') as fout:
        for idx,line in enumerate(lines(input_file)):
            splt = line.split("\t")
            if len(splt) != 4:
                print('ERROR READING LINE %s' % line)
                continue
            if idx % 1000 == 0:
                sys.stdout.write("\rProcessed %s lines" % idx)
                sys.stdout.flush()
            tsvMention = TSVMention(line.strip('\n'))
            json_mention = tsvMention.to_ent_ment().to_json()
            fout.write('%s\n' %json_mention)
