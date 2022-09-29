from torchvision.datasets import CocoDetection
import torchvision.transforms as tf
import os
import numpy as np
import scipy.misc as m

ANN_FILES = {
    'coco': 'annotations/instances_%s2017.json',
    'apricot': 'Annotations/apricot_%s_coco_annotations.json',
    'apricot-coco': 'Annotations/apricot_%s_coco_annotations.json',
    'apricot-patch': 'Annotations/apricot_%s_patch_annotations.json',
    'carla': 'annotations/%s/coco_annotations.json'
}

IM_DIR = {
    'coco': 'images/%s2017',
    'apricot': 'Images/%s',
    'apricot-coco': 'Images/%s',
    'apricot-patch': 'Images/%s',
    'carla': 'images/%s'
}

MEAN_COCO = {
    'coco': [0.485, 0.456, 0.406],
    'imagenet': [0.48235, 0.45882, 0.40784]
}

STD_COCO = {
    'coco': [0.229, 0.224, 0.225],
    'imagenet': [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
}

class CocoDetectionDataset(CocoDetection):
    def __init__(self,
                 root,
                 is_transform=False,
                 split='train',
                 version=None,
                 img_size=(300, 500),
                 img_norm=True,
                 bgr=False,
                 std_version='coco', 
                 bottom_crop=0) -> None:
        
        self.version = version
        
        self.annFile = os.path.join(root, ANN_FILES[self.version] % split)
        self.root = os.path.join(root, IM_DIR[self.version] % split)
        self.img_norm = img_norm
        self.img_size = img_size
        self.bgr = bgr
        
        if std_version is not None:
            self.mean = np.array(MEAN_COCO[std_version])
            self.std = np.array(STD_COCO[std_version])
        else:
            self.mean = np.array([0, 0, 0])
            self.std = np.array([1, 1, 1])
        self.max_val = (np.array([1, 1, 1]) - self.mean) / self.std
        self.min_val = (np.array([0, 0, 0]) - self.mean) / self.std
        
        

        transform, target_transform = None, None
        if is_transform:
            if img_size[0] is not None:
                transform = tf.Compose([tf.ToTensor(), tf.Normalize(self.mean, self.std), tf.Resize(img_size)]) 
            else:
                transform = tf.ToTensor()
            target_transform = None
        
        super().__init__(self.root, self.annFile, transform, target_transform)
        """ 
        self.ids = [
            "A list of all the file names which satisfy your criteria "
        ] 
        """
        # You can get the above list by applying your filtering logic to
        # this list :list(self.coco.imgs.keys()) So this would only be have
        # to be done only once.
        # Save it to a text file. This file will now contain the names of
        # images that match your criteria
        # Load that file contents in the init function into self.ids
        # the length would automatically be correct



    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        img_resize_factors = None
        if self.resize_factors is not None:
            img_resize_factors = self.resize_factors[index]
        # do whatever you want
        return img, target, img_resize_factors



    def to_image_transform(self, n_img):
        n_img = n_img.transpose(1,2,0)

        n_img *= self.std
        n_img += self.mean

        if self.img_norm:
            n_img *= 255.0

        if self.bgr:
            n_img = n_img[:,:,::-1]
        
        return n_img
    



        


    def image_transform(self, img, resize=True):
        if resize is True:
            img = m.imresize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
           
        # remove alpha channel (if present)
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

        return img
