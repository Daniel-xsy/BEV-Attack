import os 
import re
import torch
import numpy as np
import scipy.misc as m

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
        num_patches=1
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

                                            )

        if self.root is not None:
            self.with_subfolders = len(self.split.split('/')) < 2
            self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
            self.annotations_base = os.path.join(self.root, "gtFine", self.split)
            self.rototranslation_file = os.path.join(self.root, "gtFine", self.split, 'transform_info.txt')
            self.num_patches = num_patches
            
            with open(self.rototranslation_file, "rb") as f:
                self.rototr_lines = f.readlines()
                f.close()
            self.sign_loc, self.sign_ori = self.read_rototranslation(0)
            for p in range(1, self.num_patches):
                if p == 1:
                    self.sign_loc = [self.sign_loc]
                    self.sign_ori = [self.sign_ori]
                add_sign_loc, add_sign_ori = self.read_rototranslation(p)
                self.sign_loc.append(add_sign_loc)
                self.sign_ori.append(add_sign_ori)
                
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
                    os.path.basename(img_path)[:-4] + "gtFine_labelIds.png",
                )
        else:
            lbl_path = os.path.join(
                self.annotations_base,
                #img_path.split(os.sep)[-2],  NEEDED IF THERE ARE SUB-FOLDERS of different runs.
                os.path.basename(img_path)[:-4] + "gtFine_labelIds.png",
            )
#         try:
#             index_rototrasl = int(os.path.basename(img_path)[-6:-4])
#         except:
#             index_rototrasl = int(os.path.basename(img_path)[-5])
        index_rototrasl = int(os.path.basename(img_path).split('.')[0].split('_')[1]) + self.num_patches
#         print(index, index_rototrasl, img_path)

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        
        loc, ori = self.read_rototranslation(index_rototrasl)
        extrinsic, intrinsic = self.build_transformation(loc, ori)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, (lbl, extrinsic, intrinsic)



    def read_rototranslation(self, index):
        
        line = self.rototr_lines[index].decode('utf-8')
        ind_commas = [m.start() for m in re.finditer(', ', line)]
        ind_brackets = [m.start() for m in re.finditer('\)', line)]
        ind_x = [m.end() for m in re.finditer('x=', line)]
        ind_y = [m.end() for m in re.finditer('y=', line)]
        ind_z = [m.end() for m in re.finditer('z=', line)]
        ind_pitch = [m.end() for m in re.finditer('pitch=', line)]
        ind_yaw = [m.end() for m in re.finditer('yaw=', line)]
        ind_roll = [m.end() for m in re.finditer('roll=', line)]
        x = float(line[ind_x[0]:ind_commas[0]])
        y = float(line[ind_y[0]:ind_commas[1]])
        z = float(line[ind_z[0]:ind_brackets[0]])
        pitch = float(line[ind_pitch[0]:ind_commas[3]])
        yaw = float(line[ind_yaw[0]:ind_commas[4]])
        roll = float(line[ind_roll[0]:ind_brackets[1]])
        loc = np.array([[x, y, z]])
        ori = np.array([pitch, yaw, roll])
#         print(loc, ori)
        
        # Transform from Unreal to regular angles and position
        # Change y to -y
        loc[0, 1] = -loc[0, 1]
        
        # Change yaw to -yaw
        ori[1] = -ori[1]
        ori *= np.pi / 180.
        return loc, ori
    


    def build_transformation(self, camera_loc, camera_ori):
        # Build rototranslation matrices: T_CS = T_WS x inv(T_WC) TODO build actual rotation matrix composition (with roll and pitch)
        def build_rotation_matrix(orientation):
            p, y, r = orientation
            return np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        
        if self.num_patches == 1:
        
            R_WS = build_rotation_matrix(self.sign_ori)
            R_WC = build_rotation_matrix(camera_ori)

            T_WS = np.block([[R_WS, self.sign_loc.T], [0, 0, 0, 1]])
            T_WC = np.block([[R_WC, camera_loc.T], [0, 0, 0, 1]])
            T_WC_inv = np.block([[R_WC.T, -R_WC.T @ camera_loc.T], [0, 0, 0, 1]])
            T_WS_inv = np.block([[R_WS.T, -R_WS.T @ self.sign_loc.T], [0, 0, 0, 1]])

            T_SC = T_WC_inv @ T_WS

            # Intrinsic and extrinsic matrix
            fov = np.pi / 2
            image_h, image_w = self.img_size
            focal = 1/2 * image_w / np.tan(fov/2)
            cpi = np.cos(np.pi/2)
            spi = np.sin(np.pi/2)
            T_cam = np.transpose(np.array([[cpi, 0, spi, 0], 
                                           [0, 1, 0, 0], 
                                           [-spi, 0, cpi, 0],
                                           [0, 0, 0, 1]]) @ np.array([[cpi, spi, 0, 0], 
                                                                      [-spi, cpi, 0, 0], 
                                                                      [0, 0, 1, 0], 
                                                                      [0, 0, 0, 1]]))
            K = np.array([[focal, 0, image_w/2],
                          [0, focal, image_h/2], 
                          [0, 0, 1]])
            return torch.Tensor(T_cam) @ torch.Tensor(T_SC), torch.Tensor(K)
        
        else:
            extr, intr = [], []
            for p in range(self.num_patches):
                R_WS = build_rotation_matrix(self.sign_ori[p])
                R_WC = build_rotation_matrix(camera_ori)

                T_WS = np.block([[R_WS, self.sign_loc[p].T], [0, 0, 0, 1]])
                T_WC = np.block([[R_WC, camera_loc.T], [0, 0, 0, 1]])
                T_WC_inv = np.block([[R_WC.T, -R_WC.T @ camera_loc.T], [0, 0, 0, 1]])
                T_WS_inv = np.block([[R_WS.T, -R_WS.T @ self.sign_loc[p].T], [0, 0, 0, 1]])

                T_SC = T_WC_inv @ T_WS

                # Intrinsic and extrinsic matrix
                fov = np.pi / 2
                image_h, image_w = self.img_size
                focal = 1/2 * image_w / np.tan(fov/2)
                cpi = np.cos(np.pi/2)
                spi = np.sin(np.pi/2)
                T_cam = np.transpose(np.array([[cpi, 0, spi, 0], 
                                               [0, 1, 0, 0], 
                                               [-spi, 0, cpi, 0],
                                               [0, 0, 0, 1]]) @ np.array([[cpi, spi, 0, 0], 
                                                                          [-spi, cpi, 0, 0], 
                                                                          [0, 0, 1, 0], 
                                                                          [0, 0, 0, 1]]))
                K = np.array([[focal, 0, image_w/2],
                              [0, focal, image_h/2], 
                              [0, 0, 1]])
                
                extr.append(torch.Tensor(T_cam) @ torch.Tensor(T_SC))
                intr.append(torch.Tensor(K))
                
            return extr, intr
                
    #     print(T_cam)
        
