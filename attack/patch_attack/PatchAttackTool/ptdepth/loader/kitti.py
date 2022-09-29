import os
import cv2
import torch
import numpy as np

from ptdepth.loader.base_dataset import BaseDataset, STDS, MEANS


class kitti(BaseDataset):
    def __init__(self, root, split='test', filenames_path='filenames/', 
                 is_train=False, dataset='kitti', crop_size=(352, 704),
                 scale_size=None,
                 is_transform=True,
                 version='glpdepth',
                 img_size=None,
                 img_norm=True,
                 bgr=False,
                 std_version=None, 
                 bottom_crop=0):
        super().__init__(crop_size)        

        self.scale_size = scale_size
        self.img_size = img_size #(352, 1216)

        self.img_norm = img_norm
        self.bgr = bgr
        self.mean = np.array(MEANS[version])
        self.std = np.array(STDS[version])
        self.max_val = (np.array([1, 1, 1]) - self.mean) / self.std
        self.min_val = (np.array([0, 0, 0]) - self.mean) / self.std
        
        self.is_train = is_train
        self.data_path = root #os.path.join(data_path, 'kitti')

        self.image_path_list = []
        self.depth_path_list = []
        txt_path = os.path.join(root, 'filenames', '%s_list.txt' % split) #, 'eigen_benchmark')
        
        # if is_train:
            # txt_path += '/train_list.txt'
        # else:
            # txt_path += '/test_list.txt'        
        
        self.filenames_list = self.readTXT(txt_path)
        phase = 'train' if is_train else 'test'        
        print("Dataset :", dataset)
        print("# of %s images: %d" % (phase, len(self.filenames_list)))

    # kb cropping
    def cropping(self, img):
        h_im, w_im = img.shape[:2]

        margin_top = int((h_im - 352) / 2)
        margin_left = int((w_im - 1216) / 2)

        img = img[margin_top: margin_top + 352,
                  margin_left: margin_left + 1216]

        return img

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx, scale_factor=256.0):
        img_path = self.data_path + self.filenames_list[idx].split(' ')[0]
        gt_path = self.data_path + self.filenames_list[idx].split(' ')[1]
        filename = img_path.split('/')[-4] + '_' + img_path.split('/')[-1]

        image = cv2.imread(img_path)  # [H x W x C] and C: BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')
        
        image = self.cropping(image)
        depth = self.cropping(depth)

        if self.scale_size:
            image = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))
            depth = cv2.resize(depth, (self.scale_size[0], self.scale_size[1]))

        if self.is_train:
            image, depth = self.augment_training_data(image, depth)
        else:
            image, depth = self.augment_test_data(image, depth)

        depth = depth / scale_factor  # convert in meters
        depth[torch.where(depth>80)] = 0

        return (image, depth) #{'image': image, 'depth': depth, 'filename': filename}


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



    def to_image_transform(self, n_img):
        n_img = n_img.transpose(1,2,0)

        n_img *= self.std
        n_img += self.mean

        if self.img_norm:
            n_img *= 255.0

        if self.bgr:
            n_img = n_img[:,:,::-1]
        
        return n_img
