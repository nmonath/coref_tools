from grinch.util.IO import lines
from grinch.xdoccoref.Core import EntMent
import json


def load_json_mentions(filename):
    for line in lines(filename):
        yield EntMent.new_from_json(json.loads(line))

def load_mentions_from_file(filename):
    for idx,line in enumerate(lines(filename)):
        if idx % 1000 == 0:
            print('Loaded %s mentions' % idx)
        yield EntMent.new_from_json(json.loads(line))
