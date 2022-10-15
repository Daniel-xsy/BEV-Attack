import mmcv
import numpy as np

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadImageInfo3D:
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_lidar2img (bool, optional): Whether to load lidar to camera matrix.
            Defaults to True.

    """
    def __init__(self,
                 with_lidar2img=True,
                 with_sensor2lidar=True):

        self.with_lidar2img = with_lidar2img
        self.with_sensor2lidar = with_sensor2lidar

    def _lidar2img(self, results):
        """Private function to load lidar2img.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded lidar2img.
        """
        results['lidar2img'] = results['img_info']['lidar2img']
        return results

    def _sensor2lidar(self, results):
        """Private function to load sensor2lidar.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded sensor2lidar.
        """
        results['sensor2lidar_translation'] = results['img_info']['sensor2lidar_translation']
        results['sensor2lidar_rotation'] = results['img_info']['sensor2lidar_rotation']
        return results

    def __call__(self, results):
        """Call function to load multiple types image info.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded infomation.
        """

        if self.with_lidar2img:
            results = self._lidar2img(results)
            if results is None:
                return None

        if self.with_sensor2lidar:
            results = self._sensor2lidar(results)
            if results is None:
                return None


        return results