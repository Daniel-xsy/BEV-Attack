import json

from ptsemseg.loader.pascal_voc_loader import pascalVOCLoader
from ptsemseg.loader.coco_loader import COCOSegmentation
# from torchvision.datasets import VOCSegmentation
from ptsemseg.loader.camvid_loader import camvidLoader
from ptsemseg.loader.ade20k_loader import ADE20KLoader
from ptsemseg.loader.mit_sceneparsing_benchmark_loader import MITSceneParsingBenchmarkLoader
from ptsemseg.loader.cityscapes_loader import cityscapesLoader
from ptsemseg.loader.nyuv2_loader import NYUv2Loader
from ptsemseg.loader.sunrgbd_loader import SUNRGBDLoader
from ptsemseg.loader.mapillary_vistas_loader import mapillaryVistasLoader
from ptsemseg.loader.carla_loader import carlaLoader
from ptsemseg.loader.image_folder_loader import folderLoader


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "pascal": pascalVOCLoader,
        "coco": COCOSegmentation,
        "camvid": camvidLoader,
        "ade20k": ADE20KLoader,
        "mit_sceneparsing_benchmark": MITSceneParsingBenchmarkLoader,
        "cityscapes": cityscapesLoader,
        "nyuv2": NYUv2Loader,
        "sunrgbd": SUNRGBDLoader,
        "vistas": mapillaryVistasLoader,
        "carla": carlaLoader,
        "folder": folderLoader,
    }[name]
