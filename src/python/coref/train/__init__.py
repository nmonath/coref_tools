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

from coref.train.hac.MergePreTrainer import MergePreTrainer
from coref.train.hac.NoOpTrainer import NoOpTrainer

def new_trainer(config,model):
    """ Create a new trainer based on the trainer_name field of the config
    
    :param config: 
    :param model: 
    :return: 
    """
    if config.trainer_name == "MergePreTrainer":
        trainer = MergePreTrainer(config,None,model)
    elif config.trainer_name == 'NoOpTrainer':
        trainer = NoOpTrainer()
    else:
        raise Exception("Unknown trainer: %s" % config.trainer_name)
    return trainer