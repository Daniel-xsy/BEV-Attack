import copy
import torchvision.models as models
# from .faster_rcnn import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_320_fpn, fasterrcnn_mobilenet_v3_large_fpn
# from .mask_rcnn import maskrcnn_resnet50_fpn
# from .retinanet import retinanet_resnet50_fpn
# from .ssd import ssd300_vgg16


def get_model(model_dict, n_classes, version=None):
    name = model_dict["arch"]
    #version = model_dict["version"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")
    

    if 'frcnn' in name or 'mask-rcnn' in name or 'retinanet' in name or 'ssd' in name:
        model = model(num_classes=n_classes, pretrained=True, **param_dict)

    
    else:
        model = model(n_classes=n_classes, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "frcnn": models.detection.fasterrcnn_resnet50_fpn,
            # "frcnn-mobilenet320": fasterrcnn_mobilenet_v3_large_320_fpn,
            # "frcnn-mobilenet": fasterrcnn_mobilenet_v3_large_fpn,
            # "mask-rcnn": maskrcnn_resnet50_fpn,
            "retinanet": models.detection.retinanet_resnet50_fpn,
            # "ssd": ssd300_vgg16
        }[name]
    except:
        raise Exception("Model {} not available".format(name))
