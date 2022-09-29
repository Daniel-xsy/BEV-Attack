import os 
import re
import torch
import numpy as np
import scipy.misc as m
import json

from ptod.loader.coco_loader import CocoDetectionDataset
# from ptsemseg.utils import recursive_glob

class carlaLoader(CocoDetectionDataset):
    """
    carlaLoader
    """

    def __init__(
        self,
        root,
        is_transform=False,
        split='train',
        version=None,
        img_size=(1333, 800),
        img_norm=True,
        bgr=False,
        std_version='coco', 
        bottom_crop=0,
        num_patches=1
    ):

        super(carlaLoader, self).__init__(
            root,
            is_transform=is_transform,
            split=split,
            version=version,
            img_size=img_size,
            img_norm=img_norm,
            bgr=bgr,
            std_version=std_version, 
            bottom_crop=bottom_crop
        )

        self.n_classes = 91

        if root is not None:
            self.rototranslation_file = os.path.join(root, "annotations", split, 'camera_billboard_info.json')
            self.num_patches = num_patches
            
            with open(self.rototranslation_file, "rb") as f:
                json_dict = json.load(f)
                self.info = json_dict['info']
                self.billboards_camera_positions = json_dict['camera_billboard_positions']
                self.camera_params = json_dict['camera_params']
                self.build_intrinsics()
            

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img, target, img_resize_factors = super().__getitem__(index)
        billboards_xyzrpy = [np.array(self.billboards_camera_positions[index]['billboard_xyzrpy']), 
                            np.array(self.billboards_camera_positions[index]['billboard2_xyzrpy'])]
        extrinsic, intrinsics = self.build_extrinsics(np.array(self.billboards_camera_positions[index]['camera_xyzrpy']), 
                                          billboards_xyzrpy)
        return img, (target, extrinsic, intrinsics), img_resize_factors
    


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
    
    def build_intrinsics(self):
        focal = self.camera_params['focal']
        self.intrinsics = np.array([[focal, 0, self.img_size[1]/2],
                          [0, focal, self.img_size[0]/2], 
                          [0, 0, 1]])
                
    #     print(T_cam)
        
