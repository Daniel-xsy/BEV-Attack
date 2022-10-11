import copy
# import torchvision.models as models
from .GLPDepth import GLPDepth
from .unet_adaptive_bins import UnetAdaptiveBins


def get_model(model_dict, version=None):
    name = model_dict["arch"]
    #version = model_dict["version"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")
    
    max_depth = 80 if version in ('kitti', 'carla') else 10
    print("MAX DEPTH: %d" % max_depth)
    kwargs_dict = {
        'is_train': False,
        'max_depth': max_depth
    }
    

    if 'GLPDepth' in name:
        model = model(**kwargs_dict)
    elif 'AdaBins' in name:
        model = UnetAdaptiveBins.build(n_bins=256, min_val=1e-3, max_val=max_depth)
    
    else:
        model = model(**kwargs_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "GLPDepth": GLPDepth,
            "AdaBins": UnetAdaptiveBins
        }[name]
    except:
        raise ("Model {} not available".format(name))
