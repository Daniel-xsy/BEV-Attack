import os
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data
from torchvision.transforms.functional import crop
import torch.nn.functional as F

from ptsemseg.utils import recursive_glob


class baseCityscapesLoader(data.Dataset):

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

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
        split="train",
        is_transform=False,
        img_size=(1024, 2048),
        augmentations=None,
        img_norm=True,
        version=None,
        bgr = True, 
        std_version = None,
        bottom_crop = 0,
        images_base_set = False
    ):
        """__init__
        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.bottom_crop = bottom_crop
        self.root = root
        self.split = split
        self.bgr = bgr
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 19
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        self.mean = np.array(self.mean_rgb[version]) if version is not None else np.array(self.std['None'])
        self.std = np.array(self.std[std_version]) if std_version is not None else np.array(self.std['None'])

        # min and max value of dataset samples
        self.max_val, self.min_val = self.get_boundaries()
        

        self.files = {}
        if self.root is not None and images_base_set is False:
            self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
            self.annotations_base = os.path.join(self.root, "gtFine", self.split)

            self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if self.root is not None and images_base_set is False:
            if not self.files[split]:
                raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

            print("Found %d %s images" % (len(self.files[split]), split))


    def __len__(self):
        """__len__"""
        return len(self.files[self.split])


    def __getitem__(self, index):
        return None

    
    def get_boundaries(self):
        max_val = [1.0]*len(self.mean) if self.img_norm else [255.0]*len(self.mean) 
        min_val = [0.0]*len(self.mean) 

        if(self.bgr):
            max_val = max_val[::-1]
            min_val = min_val[::-1]

        max_val -= self.mean
        max_val /= self.std
        min_val -= self.mean
        min_val /= self.std

     
        return max_val, min_val



    '''
    to adjust no batch input but works if needed for 
    '''
    def crop_image(self, img, label):

        image_ratio = self.img_size[1]/self.img_size[0]
        to_remove_width_per_side = int((self.bottom_crop*image_ratio)/2)

        img = crop(
            img, 
            top = 0, 
            left = to_remove_width_per_side, 
            height = self.img_size[0] - self.bottom_crop,
            width= self.img_size[1] - to_remove_width_per_side
        )
        
        img = F.interpolate(img,
            size=(self.img_size[0], self.img_size[1]))

        label = label.float().unsqueeze(1)

        label = crop(
            label, 
            top = 0, 
            left = to_remove_width_per_side, 
            height = self.img_size[0] - self.bottom_crop,
            width= self.img_size[1] - to_remove_width_per_side
        )
        
        label = F.interpolate(label,
            size=(self.img_size[0], self.img_size[1])).squeeze(1).long()

        return img, label
        


    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        img = self.image_transform(img)
        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()

        #if self.bottom_crop > 0 :
        #    img = self.crop_image(img)

        lbl = torch.from_numpy(lbl).long()

        return img, lbl



    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb



    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask


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


'''
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip(0.5)])

    local_path = "/datasets01/cityscapes/112817/"
    dst = cityscapesLoader(local_path, is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples
        import pdb

        pdb.set_trace()
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()
'''