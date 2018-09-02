
from coref.models.nn.NSW import NSW

from geo.models.VecNSW import VecNSW
from geo.models.VecNSWExact import VecNSWExact
# from geo.models.NswNSW import NswNSW

def new_nn_structure(name,config,score_fn, dataset=[]):
    if name == 'nsw':
        return NSW(config,dataset,score_fn,config.nn_k,config.nsw_r,config.random_seed)
    elif name == 'vnsw':
        return VecNSW(config, dataset, score_fn, config.nn_k, config.nsw_r,
                      config.random_seed)
    elif name == 'VecNSWExact':
        return VecNSWExact(config, dataset, score_fn, config.nn_k, config.nsw_r,
                      config.random_seed)
    # elif name == 'nswnsw':
    #     return NswNSW(config, dataset, score_fn, config.nn_k, config.nsw_r,
    #                   config.random_seed)
    else:
        raise Exception('Unknown nn structure %s' % config.nn_structure)