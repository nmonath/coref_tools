
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