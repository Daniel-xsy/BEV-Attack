"""
Misc Utility functions
"""
import os
import logging
import datetime
import numpy as np

from collections import OrderedDict


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value
    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_logger(logdir):
    logger = logging.getLogger("ptsemseg")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger



#----------------------------------------------------------------------------
# @ Giulio 
# return the correct dictionary depending on the model name
# (needed to simplify strange pretrained model dictionary)
#----------------------------------------------------------------------------

def get_model_state(state, model_name):
    print(model_name)

    if model_name in ['ddrnet23Slim', 'ddrnet23']:
        state = convert_state_dict(state)
        new_state_dict = OrderedDict()
        for k, v in state.items():
            name = k[6:]  
            if name == 'riterion.weight':
                break
            new_state_dict[name] = v
        return new_state_dict

    elif model_name in ['bisenetR18', 'bisenetX39', 'bisenetR101']:
        return convert_state_dict(state)['model']
    
    elif model_name in ['danetR101']:
        print(state['state_dict'].keys())
        return convert_state_dict(state)['state_dict']

    
    elif model_name in ['psanet50', 'psanet101']:

        #print(state.keys())

        state = convert_state_dict(state)['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state.items():
            name = k[7:]  
            new_state_dict[name] = v
        return new_state_dict

    else:
        state = convert_state_dict(state["model_state"])
        return state
