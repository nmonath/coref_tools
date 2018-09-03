
import random

class XDocBatcher(object):
    def __init__(self,config,mentions,pairs,return_one_epoch=False):
        self.config = config
        self.mentions = mentions
        self.pairs = pairs
        self.return_one_epoch = return_one_epoch
        self.id_2_mention = dict()
        self.random = random.Random(self.config.random_seed)
        self.offset = 0

        for m in self.mentions:
            self.id_2_mention[m.mid] = m
        self.shuffle()

    def shuffle(self):
        self.random.shuffle(self.pairs)

    def get_next_batch(self):
        while True:
            if self.offset >= len(self.pairs) and not self.return_one_epoch:
                self.shuffle()
                self.offset = 0
            elif self.offset >= len(self.pairs) and self.return_one_epoch:
                return
            else:
                start = self.offset
                end = min(len(self.pairs),start + self.config.batch_size)
                batch = self.pairs[start:end]
                lefts = [self.id_2_mention[x[0]] for x in batch]
                rights = [self.id_2_mention[x[1]] for x in batch]
                if self.config.produce_sample_pairs:
                    third = [int(x[2]) for x in batch]
                else:
                    third = [self.id_2_mention[x[2]] for x in batch]
                self.offset += end
                yield lefts,rights,third
