import os 
import re
import torch
import numpy as np
# import scipy.misc as m
import json

from pt3dod.loader.kitti import kitti
# from ptsemseg.utils import recursive_glob

class carlaLoader(kitti):
    """
    carlaLoader
    """

    def __init__(
        self,
        root,
        is_transform=False,
        split='train',
        version=None,
        img_size=(370, 1224),
        img_norm=True,
        bgr=False,
        std_version='coco', 
        bottom_crop=0,
        num_patches=1
    ):
#image_set, kitti_path=None, dataset='kitti'
        super(carlaLoader, self).__init__(
            split,
            root,
            dataset='carla',
            # TODO add all the others!,
            bgr=bgr,
            img_norm=img_norm
        )
        self.img_size = img_size
        if root is not None:
            self.rototranslation_file = os.path.join(root, split, 'label_2', 'camera_billboard_info.json')
            self.num_patches = num_patches
            
            with open(self.rototranslation_file, "rb") as f:
                json_dict = json.load(f)
                self.info = json_dict['info']
                self.billboards_camera_positions = json_dict['camera_billboard_positions']
                self.camera_params = json_dict['camera_params']
                self.camera_dx_params = json_dict['camera_dx_params']
                self.build_intrinsics(self.camera_params)
            

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        
        data_left, data_right, im_info, gt_boxes_left_padding, gt_boxes_right_padding, gt_boxes_merge_padding, gt_dim_orien_padding, gt_kpts_padding, num_boxes  = super().__getitem__(index)
        
#         loc, ori = self.read_rototranslation(index_rototrasl)
        billboards_xyzrpy = [np.array(self.billboards_camera_positions[index]['billboard_xyzrpy']), 
                            np.array(self.billboards_camera_positions[index]['billboard2_xyzrpy'])]
        extrinsic, intrinsics = self.build_extrinsics(np.array(self.billboards_camera_positions[index]['camera_xyzrpy']), 
                                          billboards_xyzrpy)

        
        return data_left, data_right, im_info, gt_boxes_left_padding, gt_boxes_right_padding, gt_boxes_merge_padding, gt_dim_orien_padding, gt_kpts_padding, num_boxes, (extrinsic, intrinsics)
    


    def build_extrinsics(self, camera_xyzrpy, billboards_xyzrpy):
        # Build rototranslation matrices: T_CS = T_WS x inv(T_WC) TODO build actual rotation matrix composition (with roll and pitch)
        def build_rotation_matrix(orientation):
            r, p, y = orientation
            return np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        
       
        extr, intr = [], []
        for p in range(self.num_patches):
            billboard_loc = billboards_xyzrpy[p][:3].reshape((3, 1))
            billboard_ori = billboards_xyzrpy[p][-3:]
            camera_loc = camera_xyzrpy[:3].reshape((3, 1))
            camera_ori = camera_xyzrpy[-3:]
            R_WS = build_rotation_matrix(billboard_ori)
            R_WC = build_rotation_matrix(camera_ori)
            
            T_WS = np.block([[R_WS, billboard_loc], [0, 0, 0, 1]])
            T_WC = np.block([[R_WC, camera_loc], [0, 0, 0, 1]])
            T_WC_inv = np.block([[R_WC.T, -R_WC.T @ camera_loc], [0, 0, 0, 1]])
            T_WS_inv = np.block([[R_WS.T, -R_WS.T @ billboard_loc], [0, 0, 0, 1]])

            T_SC = T_WC_inv @ T_WS

            # Intrinsic and extrinsic matrix
            
            cpi = np.cos(np.pi/2)
            spi = np.sin(np.pi/2)
            T_cam = np.transpose(np.array([[cpi, 0, spi, 0], 
                                           [0, 1, 0, 0], 
                                           [-spi, 0, cpi, 0],
                                           [0, 0, 0, 1]]) @ np.array([[cpi, spi, 0, 0], 
                                                                      [-spi, cpi, 0, 0], 
                                                                      [0, 0, 1, 0], 
                                                                      [0, 0, 0, 1]]))
            

            extr.append(torch.Tensor(T_cam) @ torch.Tensor(T_SC))
            intr.append(torch.Tensor(self.intrinsics))
                
        return extr, intr
    
    def build_intrinsics(self, camera_params):
        # focal = self.camera_params['focal']
        fov = camera_params['fov']
        image_h, image_w = self.img_size
        focal = 1/2 * image_w / np.tan(fov/2)
        self.intrinsics = np.array([[focal, 0, self.img_size[1]/2],
                          [0, focal, self.img_size[0]/2], 
                          [0, 0, 1]])
                
    #     print(T_cam)
        
