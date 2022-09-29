from ptod.loader.coco_loader import CocoDetectionDataset
from ptod.loader.carla_loader import carlaLoader


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        # "pascal": pascalVOCLoader,
        "coco": CocoDetectionDataset,
        "carla": carlaLoader,
        # "folder": folderLoader,
    }[name]