import copy
import torchvision.models as models

from ptsemseg.models.fcn import fcn8s, fcn16s, fcn32s
from ptsemseg.models.segnet import segnet
from ptsemseg.models.unet import unet
from ptsemseg.models.pspnet import pspnet
from ptsemseg.models.icnet import icnet
from ptsemseg.models.linknet import linknet
from ptsemseg.models.frrn import frrn

from ptsemseg.models.ddrnet_23_slim import DDRNET_23_SLIM
from ptsemseg.models.ddrnet_23 import DDRNET_23
from ptsemseg.models.bisenet_R18 import BISENET_R18
from ptsemseg.models.bisenet_R101 import BISENET_R101
from ptsemseg.models.bisenet_X39 import BISENET_X39
from ptsemseg.models.psanet import PSANet
from ptsemseg.models.danet_R101 import DANET_R101




def get_model(model_dict, n_classes, version=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")
    

    if name in ["frrnA", "frrnB"]:
        model = model(n_classes, **param_dict)

    elif name in ["fcn32s", "fcn16s", "fcn8s"]:
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "segnet":
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "unet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "pspnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "icnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "icnetBN":
        model = model(n_classes=n_classes, **param_dict)

    elif name in ["ddrnet23Slim", "ddrnet23"]:
        model = model(num_classes=n_classes, **param_dict)

    elif name in ["bisenetR18", "bisenetR101", "bisenetX39"]:
        model = model(num_classes=n_classes, **param_dict)
    
    elif name == 'danetR101':
        model = model(nclass=n_classes, **param_dict)

    elif name == "psanet50":
        model = model(layers=50, **param_dict)

    elif name == "psanet101":
        model = model(layers=101, classes=n_classes, **param_dict)

    else:
        model = model(n_classes=n_classes, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "fcn32s": fcn32s,
            "fcn8s": fcn8s,
            "fcn16s": fcn16s,
            "unet": unet,
            "segnet": segnet,
            "pspnet": pspnet,
            "icnet": icnet,
            "icnetBN": icnet,
            "linknet": linknet,
            "frrnA": frrn,
            "frrnB": frrn,

            # additional models
            "ddrnet23Slim": DDRNET_23_SLIM, 
            "ddrnet23": DDRNET_23,
            "bisenetR18": BISENET_R18,
            "bisenetR101": BISENET_R101,
            "bisenetX39": BISENET_X39,
            "danetR101": DANET_R101,
            "psanet50": PSANet,
            "psanet101": PSANet,
        }[name]
    except:
        raise ("Model {} not available".format(name))
