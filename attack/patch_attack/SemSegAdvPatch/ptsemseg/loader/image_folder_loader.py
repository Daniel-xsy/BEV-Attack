import os 
import re
import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import scipy.misc as m

from torch.utils import data

from ptsemseg.utils import recursive_glob
from ptsemseg.loader.base_cityscapes_loader import baseCityscapesLoader



class folderLoader(baseCityscapesLoader):
    """
    folderLoader - to load images for a general folder
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
        bottom_crop = 0
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """

        super(folderLoader, self).__init__(
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
                                                images_base_set = True
                                            )
        
        print(self.img_size)
        
        self.root = root
        self.files = {}
        if self.root is not None:
            self.images_base = self.root 
            self.files[split] = sorted(recursive_glob(rootdir=self.images_base, suffix=(".png", ".JPG")))
    
    
    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        
        img_path = self.files[self.split][index].rstrip()
        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = img.copy()

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl
    


    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """

        if img.shape[-1] > 3:
            img = img[:, :, :-1]
        
        if(self.bgr):
            img = img[:, :, ::-1]  # RGB -> BGR

        img = img.astype(np.float64)

        if self.img_norm:
            img = img.astype(float) / 255.0

        img -= self.mean
        img /= self.std

        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        
        # TODO. improve cropping method
        #img = transforms.CenterCrop(self.img_size)(img)
        img = self.crop_image(img, None)
        img = F.interpolate(img.unsqueeze(0), size=self.img_size, mode="bilinear", align_corners=True).squeeze(0)

        return img, img
