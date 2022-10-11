import copy
# import torchvision.models as models
import sys
sys.path.append('pt3dod/models/Stereo-RCNN')
from lib.model.stereo_rcnn.resnet import resnet


def get_model(model_dict, version=None):
    name = model_dict["arch"]
    #version = model_dict["version"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")
    
    # max_depth = 80 if version in ('kitti', 'carla') else 10
    # print("MAX DEPTH: %d" % max_depth)
    kwargs_dict = {
        'classes': ('__background__', 'car'),
        'num_layers': 101
    }
    

    if 'stereo-rcnn' in name:
        model = model(**kwargs_dict)
        model.create_architecture()
    
    else:
        model = model(**kwargs_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "stereo-rcnn": resnet,
        }[name]
    except:
        raise ("Model {} not available".format(name))