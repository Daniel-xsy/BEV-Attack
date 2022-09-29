import os
import cv2
import numpy as np

from ptdepth.loader.base_dataset import BaseDataset, MEANS, STDS


class nyudepthv2(BaseDataset):
    def __init__(self, root, split='test', filenames_path='./code/dataset/filenames/',
                 is_train=False, crop_size=(448, 576), scale_size=None,
                 is_transform=True,
                 version='glpdepth',
                 img_size=None,
                 img_norm=True,
                 bgr=False,
                 std_version=None, 
                 bottom_crop=0):
        super().__init__(crop_size)

        self.scale_size = scale_size
        self.img_size = (448, 576)

        self.img_norm = img_norm
        self.bgr = bgr
        self.mean = np.array(MEANS[version])
        self.std = np.array(STDS[version])
        self.max_val = (np.array([1, 1, 1]) - self.mean) / self.std
        self.min_val = (np.array([0, 0, 0]) - self.mean) / self.std

        self.is_train = is_train
        self.data_path = os.path.join(root, 'official_splits', split)

        self.image_path_list = []
        self.depth_path_list = []

        # txt_path = os.path.join(filenames_path, 'nyudepthv2')
        txt_path = os.path.join(root, 'filenames', '%s_list.txt' % split)
        # if is_train:
        #     txt_path += '/train_list.txt'
        # else:
        #     txt_path += '/test_list.txt'
            # self.data_path = self.data_path + '/official_splits/test/'

        self.filenames_list = self.readTXT(txt_path)
        phase = 'train' if is_train else 'test'
        print("Dataset: NYU Depth V2")
        print("# of %s images: %d" % (phase, len(self.filenames_list)))

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        img_path = self.data_path + self.filenames_list[idx].split(' ')[0]
        gt_path = self.data_path + self.filenames_list[idx].split(' ')[1]
        filename = img_path.split('/')[-2] + '_' + img_path.split('/')[-1]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')

        if self.scale_size:
            image = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))
            depth = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))

        if self.is_train:
            image, depth = self.augment_training_data(image, depth)
        else:
            image, depth = self.augment_test_data(image, depth)

        depth = depth / 1000.0  # convert in meters
        # print(depth.shape)
        # print(image.shape)
        return image, depth


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
