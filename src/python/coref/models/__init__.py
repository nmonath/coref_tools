
from coref.models.entity.RexaBaseSubEnt import RexaBaseSubEnt
from coref.models.entity.RexaBasePlusName import RexaBasePlusName
from coref.models.entity.RexaBasePlusNamePlusTitle import RexaBasePlusNamePlusTitle
from coref.models.entity.AuthorCorefModel import AuthorCorefModel
from coref.models.entity.OneDModel import OneDModel


def load_model(config):
    """ Load the best model stored in the config file
    
    :param config: 
    :return: 
    """
    if config.model_name == 'AuthorCoref'or config.model_name == 'AuthorCorefModel':
        model = AuthorCorefModel.load(config.best_model)
    elif config.model_name == 'BaseSubEnt' or config.model_name == 'RexaBaseSubEnt':
        model = RexaBaseSubEnt.load(config.best_model)
    elif config.model_name == 'BasePlusName' or config.model_name == 'RexaBasePlusName':
        model = RexaBasePlusName.load(config.best_model)
    elif config.model_name == 'BasePlusNamePlusTitle' or config.model_name == 'RexaBasePlusNamePlusTitle':
        model = RexaBasePlusNamePlusTitle.load(config.best_model)
    elif config.model_name == 'OneDModel':
        model = OneDModel(config,None)
    else:
        raise Exception("Unknown Model: {}".format(config.model_name))
    return model


def new_model(config,vocab=None):
    """ Create a new model object based on the model_name field in the config
    
    :param config: 
    :return: 
    """
    if config.model_name == 'AuthorCoref' or config.model_name == 'AuthorCorefModel':
        model = AuthorCorefModel(config,vocab)
    elif config.model_name == 'BaseSubEnt' or config.model_name == 'RexaBaseSubEnt':
        model = RexaBaseSubEnt(config,vocab)
    elif config.model_name == 'BasePlusName' or config.model_name == 'RexaBasePlusName':
        model =  RexaBasePlusName(config,vocab)
    elif config.model_name == 'BasePlusNamePlusTitle' or config.model_name == 'RexaBasePlusNamePlusTitle':
        model =  RexaBasePlusNamePlusTitle(config,vocab)
    elif config.model_name == 'OneDModel':
        model = OneDModel(config, None)
    else:
        raise Exception("Unknown Model: {}".format(config.model_name))
    return model


