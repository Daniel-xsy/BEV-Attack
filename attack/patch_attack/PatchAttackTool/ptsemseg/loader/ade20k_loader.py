import collections
from numpy.lib.twodim_base import mask_indices
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt

from torch.utils import data

from ptsemseg.utils import recursive_glob


class ADE20KLoader(data.Dataset):
    # map 1 to 1 of all the dataset classes

    # 15 - 774, 783
    # 36 - 2985, 533
    # 89 - 246, 247
     
    classes = np.array([
        2978,
        312,
        2420,
        976,
        2855,
        447,
        2131,
        165,
        3055,
        1125,
        350,
        2377,
        1831,
        838,
        774,
        783,
        2684,
        1610,
        1910,
        687,
        471,
        401,
        2994,
        1735,
        2473,
        2329,
        1276,
        2264,
        1564,
        2178,
        913,
        57,
        2272,
        907,
        724,
        2138,
        2985,
        533,
        1395,
        155,
        2053,
        689,
        137,
        266,
        581,
        2380,
        491,
        627,
        2212,
        2388,
        2423,
        943,
        2096,
        1121,
        1788,
        2530,
        2185,
        420,
        1948,
        1869,
        2251,
        2531,
        2128,
        294,
        239,
        212,
        571,
        2793,
        978,
        236,
        1240,
        181,
        629,
        2598,
        744,
        1374,
        591,
        2679,
        223,
        123,
        47,
        1282,
        327,
        2821,
        1451,
        2880,
        2828,
        480,
        77,
        2616,
        246,
        247,
        2733,
        14,
        738,
        38,
        1936,
        1401,
        120,
        868,
        1702,
        249,
        308,
        1969,
        2526,
        2928,
        2337,
        1023,
        609,
        389,
        2989,
        1930,
        2668,
        2586,
        131,
        146,
        3016,
        2739,
        95,
        1563,
        642,
        1708,
        103,
        1002,
        2569,
        2704,
        2833,
        1551,
        1981,
        29,
        187,
        1393,
        747,
        2254,
        206,
        2262,
        1260,
        2243,
        2932,
        2836,
        2850,
        64,
        894,
        1858,
        3109,
        1919,
        1583,
        318,
        2356,
        2046,
        1098,
        530,
        954])
    real_index = list(range(16))
    real_index.extend(range(15,37))
    real_index.extend(range(36, 90))
    real_index.extend(range(89, 150))
    real_index = np.array(real_index)


    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "None": [0.0, 0.0, 0.0],
        "ddrnet_23": [0.485, 0.456, 0.406]
    }  # pascal mean for PSPNet and ICNet pre-trained model

    std = {
        "ddrnet_23": [0.229, 0.224, 0.225],
        "None": [1.0, 1.0, 1.0]
    }


    def __init__(
        self,
        root,
        split="training",
        is_transform=True,
        img_size=512,
        augmentations=None,
        img_norm=True,
        test_mode=False,
        version='pascal',     # THESE ARGUMENTS WERE ADDED.
        bgr=False,
        std_version='pascal',
        bottom_crop=0
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.n_classes = 150
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array(self.mean_rgb[version]) if version is not None else np.array(self.std['None'])
        self.std = np.array(self.std[std_version]) if std_version is not None else np.array(self.std['None'])
        self.files = collections.defaultdict(list)
        self.bgr = bgr

        self.ignore_index = 250

        self.classes_lut = np.zeros((max(self.classes))+500)
        for i, c in enumerate(self.classes):
            self.classes_lut[c] = i

        if not self.test_mode:
            for split in ["training", "validation"]:
                file_list = recursive_glob(
                    rootdir=self.root + "images/" + self.split + "/", suffix=".jpg"
                )
                self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = img_path[:-4] + "_seg.png"

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.int32)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    #-------------------------------------------------------------
    def transform(self, img, lbl):
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        if self.bgr:
            img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        if self.img_norm:
            img = img.astype(float) / 255.0

        img -= self.mean
        img /= self.std
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
       
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)
        assert np.all(classes == np.unique(lbl))

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")
              
        #if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
        #    print("after det", classes, np.unique(lbl))
        #    raise ValueError("Segmentation map contained invalid class values")
     
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl


    def encode_segmap(self, mask):
        # Refer : http://groups.csail.mit.edu/vision/datasets/ADE20K/code/loadAde20K.m
        mask = mask.astype(int)
        #print(mask)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]))
        label_mask = (mask[:, :, 0] / 10) * 256 + mask[:, :, 1]
        label_mask = np.array(label_mask, dtype=np.int16)
        # for i in range(label_mask.shape[0]):
        #     for j in range(label_mask.shape[1]):
        #         map = np.where(self.classes == label_mask[i,j])[0]
        #         if(len(map) == 0):
        #             label_mask[i,j] = 0
        #         else:
        #             label_mask[i,j] = map.item(0)
        label_mask = self.real_index[np.array(self.classes_lut[label_mask], dtype=np.int16)]
        return np.array(label_mask, dtype=np.uint8)


    def decode_segmap(self, temp):
        # TODO:(@meetshah1995)
        # Verify that the color mapping is 1-to-1
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = 10 * (l % 10)
            g[temp == l] = l
            b[temp == l] = 0
        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0

        return rgb
        
        
    def to_image_transform(self, n_img):
        n_img = n_img.transpose(1,2,0)
    
        n_img *= self.std
        n_img += self.mean

        if self.img_norm:
            n_img *= 255.0

        if self.bgr:
            n_img = n_img[:,:,::-1]
        
        return n_img




if __name__ == "__main__":
    local_path = "/Users/meet/data/ADE20K_2016_07_26/"
    dst = ADE20KLoader(local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
            for j in range(4):
                plt.imshow(dst.decode_segmap(labels.numpy()[j]))
                plt.show()
