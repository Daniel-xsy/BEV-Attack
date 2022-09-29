import os 
import re
import torch
import numpy as np
import scipy.misc as m
import json

from ptsemseg.loader.base_cityscapes_loader import baseCityscapesLoader
from ptsemseg.utils import recursive_glob

class carlaLoader(baseCityscapesLoader):
    """
    carlaLoader
    """

    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(512, 1024),
        augmentations=None,
        img_norm=True,
        version="cityscapes",
        bgr = True, 
        std_version = "cityscapes",
        bottom_crop = 0,
        num_patches=1,
        # color_transfer=False
    ):

        super(carlaLoader, self).__init__(
            root,
            split=split,
            is_transform=is_transform,
            img_size=img_size,
            augmentations=augmentations,
            img_norm=img_norm,
            version=version,
            bgr = bgr, 
            std_version = std_version,
            bottom_crop = bottom_crop, 
            images_base_set = True,
            # color_transfer = color_transfer
        )

        if self.root is not None:
            self.with_subfolders = len(self.split.split('/')) < 2
            self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
            self.annotations_base = os.path.join(self.root, "gtFine", self.split)
            self.rototranslation_file = os.path.join(self.root, "gtFine", self.split, 'camera_billboard_info.json')
            self.num_patches = num_patches
            
            with open(self.rototranslation_file, "rb") as f:
                json_dict = json.load(f)
                self.info = json_dict['info']
                self.billboards_camera_positions = json_dict['camera_billboard_positions']
                self.camera_params = json_dict['camera_params']
                self.build_intrinsics()
            
#             self.sign_loc, self.sign_ori = self.read_rototranslation(0)
#             for p in range(1, self.num_patches):
#                 if p == 1:
#                     self.sign_loc = [self.sign_loc]
#                     self.sign_ori = [self.sign_ori]
#                 add_sign_loc, add_sign_ori = self.read_rototranslation(p)
#                 self.sign_loc.append(add_sign_loc)
#                 self.sign_ori.append(add_sign_ori)
                
            self.files[split] = sorted(recursive_glob(rootdir=self.images_base, suffix=".png"))

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        
        img_path = self.files[self.split][index].rstrip()

        if self.with_subfolders:
            if 'aachen' in os.path.basename(img_path) or 'tubingen' in os.path.basename(img_path) or 'munster' in os.path.basename(img_path):
                lbl_path = os.path.join(
                self.annotations_base,
                img_path.split(os.sep)[-2],  #NEEDED IF THERE ARE SUB-FOLDERS of different runs.
                os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
            )
            else:
                lbl_path = os.path.join(
                    self.annotations_base,
                    img_path.split(os.sep)[-2],  #NEEDED IF THERE ARE SUB-FOLDERS of different runs.
                    os.path.basename(img_path)[:-4] + "_gtFine_labelIds.png",
                )
        else:
            lbl_path = os.path.join(
                self.annotations_base,
                #img_path.split(os.sep)[-2],  NEEDED IF THERE ARE SUB-FOLDERS of different runs.
                os.path.basename(img_path)[:-4] + "_gtFine_labelIds.png",
            )
#         try:
#             index_rototrasl = int(os.path.basename(img_path)[-6:-4])
#         except:
#             index_rototrasl = int(os.path.basename(img_path)[-5])
#         index_rototrasl = int(os.path.basename(img_path).split('.')[0].split('_')[1]) + self.num_patches
#         print(index, index_rototrasl, img_path)

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        
#         loc, ori = self.read_rototranslation(index_rototrasl)
        billboards_xyzrpy = [np.array(self.billboards_camera_positions[index]['billboard_xyzrpy']), 
                            np.array(self.billboards_camera_positions[index]['billboard2_xyzrpy'])]
        
        extrinsic, intrinsics = [torch.zeros((4, 4))], [torch.zeros((3, 3))] #, None
        if self.info['num_billboards'] > 0:
            extrinsic, intrinsics = self.build_extrinsics(np.array(self.billboards_camera_positions[index]['camera_xyzrpy']), 
                                          billboards_xyzrpy)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, (lbl, extrinsic, intrinsics)
    


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
        
