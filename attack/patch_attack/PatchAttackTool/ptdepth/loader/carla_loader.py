import os 
import re
import torch
import numpy as np
# import scipy.misc as m
import json

from ptdepth.loader.kitti import kitti
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
        img_size=(352, 1216),
        img_norm=True,
        bgr=False,
        std_version='coco', 
        bottom_crop=0,
        is_train=False,
        num_patches=1
    ):
        super(carlaLoader, self).__init__(
            root,
            split=split,
            is_train=False,
            img_size=img_size
            # TODO add all the others!
        )

        if root is not None:
            self.rototranslation_file = os.path.join(root, 'gtFine', split, 'camera_billboard_info.json')
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
        
        img, target = super().__getitem__(index, scale_factor=1)
        
        billboards_xyzrpy = [np.array(self.billboards_camera_positions[index]['billboard_xyzrpy']), 
                            np.array(self.billboards_camera_positions[index]['billboard2_xyzrpy'])]
        
        extrinsic, intrinsics = [torch.zeros((4, 4))], [torch.zeros((3, 3))] #, None
        if self.info['num_billboards'] > 0:
            extrinsic, intrinsics = self.build_extrinsics(np.array(self.billboards_camera_positions[index]['camera_xyzrpy']), 
                                          billboards_xyzrpy)
        
        # if self.augmentations is not None:
            # img, lbl = self.augmentations(img, lbl)

        # if self.is_transform:
            # img, lbl = self.transform(img, lbl)

        return img, (target, extrinsic, intrinsics)
    


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
        fov = self.camera_params['fov']
        image_h, image_w = self.img_size
        # print(image_h, image_w)
        focal = 1/2 * image_w / np.tan(fov/2)
        self.intrinsics = np.array([[focal, 0, self.img_size[1]/2],
                          [0, focal, self.img_size[0]/2],  # Necessary because of the image cropping. 
                          [0, 0, 1]])
                
    #     print(T_cam)
        
