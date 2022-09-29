from ptdepth.loader.kitti import kitti
from ptdepth.loader.carla_loader import carlaLoader
from ptdepth.loader.nyudepthv2 import nyudepthv2


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "kitti": kitti,
        'carla': carlaLoader,
        'nyuv2': nyudepthv2
    }[name]