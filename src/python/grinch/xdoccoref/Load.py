from grinch.util.IO import lines
from grinch.xdoccoref.Core import EntMent
import json


def load_json_mentions(filename):
    for line in lines(filename):
        yield EntMent.new_from_json(json.loads(line))
