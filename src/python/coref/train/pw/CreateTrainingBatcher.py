"""
Copyright (C) 2018 University of Massachusetts Amherst.
This file is part of "coref_tools"
http://github.com/nmonath/coref_tools
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys

from coref.models import new_model
from coref.util.Config import Config
from coref.train import new_trainer

if __name__ == "__main__":
    config = Config(sys.argv[1])
    model = new_model(config)
    trainer = new_trainer(config,model)

    def mention_pairs(filename):
        with open(filename, 'r') as fin:
            for idx, line in enumerate(fin):
                splt = line.split('\t')
                if len(splt) !=3:
                    print("Error on line %s" % idx)
                    print(line)
                yield splt[0], splt[1], splt[2]

    if trainer is not None:
        trainer.write_training_data(mention_pairs(config.pair_filename),
                                    config.batcher_filename)
    else:
        model.write_training_data(mention_pairs(config.pair_filename),
                                  config.batcher_filename)
    print('[WROTE TRAIN PAIRS.] %s' % config.pair_filename)
    # if trainer is not None:
    #     trainer.write_training_data(mention_pairs(config.dev_pair_filename),
    #                                 config.dev_batcher_filename)
    # else:
    #     model.write_training_data(mention_pairs(config.dev_pair_filename),
    #                               config.dev_batcher_filename)
    # print('[WROTE DEV PAIRS.] %s' % config.dev_pair_filename)
