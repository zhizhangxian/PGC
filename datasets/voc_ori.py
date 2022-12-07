import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np
import torch

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity
from utils import cor_transforms as train_et



DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit/VOC2012'
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': 'TrainVal/VOCdevkit/VOC2011'
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': 'VOCdevkit/VOC2010'
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': 'VOCdevkit/VOC2009'
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': 'VOCdevkit/VOC2008'
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': 'VOCdevkit/VOC2007'
    }
}


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    cmap = voc_cmap()
    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None,
                 num_copys=1):

        is_aug=False
        if year=='2012_aug':
            is_aug = True
            year = '2012'
        
        self.root = os.path.expanduser(root)
        self.year = year
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        self.transform = transform
        self.num_copys = num_copys

        self.image_set = image_set
        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)
        # voc_root = voc_root.replace('/', '\\')
        # print(voc_root)
        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        


        if is_aug and image_set=='train':
            mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
            assert os.path.exists(mask_dir), "SegmentationClassAug not found, please refer to README.md and prepare it manually"
            splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
            
            split_f = os.path.join(splits_dir, 'train_aug.txt')#'./datasets/data/train_aug.txt'
        else:
            mask_dir = os.path.join(voc_root, 'SegmentationClass')
            splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))




    def get_overlaps(self, cur_cors, ori_cors, scales, flips):
        scale = scales[0] / scales[1]
        _flip = flips[0] * flips[1]
        transform = [scale, _flip]
        overlaps = []
        up = max(ori_cors[0][0], ori_cors[1][0])
        left = max(ori_cors[0][1], ori_cors[1][1])
        down = min(ori_cors[0][2], ori_cors[1][2])
        right = min(ori_cors[0][3], ori_cors[1][3])
        up_left = (up, left)
        down_right = (down, right)
        for i in range(self.num_copys):
            flip = flips[i]
            ori_cor = ori_cors[i]
            cur_cor = cur_cors[i]
            # print('cur_cor = {:}'.format(cur_cor))
            if flip == 1:
                size_y, size_x = cur_cor[2] - cur_cor[0], cur_cor[3] - cur_cor[1]
                _up_left = [round(cur_cor[0] + size_y * (up_left[0] - ori_cor[0]) / (ori_cor[2] - ori_cor[0])),
                            round(cur_cor[1] + size_x * (up_left[1] - ori_cor[1]) / (ori_cor[3] - ori_cor[1]))]
                _down_right = [round(cur_cor[0] + size_y * (down_right[0] - ori_cor[0]) / (ori_cor[2] - ori_cor[0])),
                               round(cur_cor[1] + size_x * (down_right[1] - ori_cor[1]) / (ori_cor[3] - ori_cor[1]))]

            elif flip == -1:

                size_y, size_x = cur_cor[2] - cur_cor[0], cur_cor[3] - cur_cor[1]
                _up_left = [round(cur_cor[0] + size_y * (up_left[0] - ori_cor[0]) / (ori_cor[2] - ori_cor[0])),
                            round(cur_cor[1] + size_x * (1 - (down_right[1] - ori_cor[1]) / (ori_cor[3] - ori_cor[1])))]
                _down_right = [round(cur_cor[0] + size_y * (down_right[0] - ori_cor[0]) / (ori_cor[2] - ori_cor[0])),
                               round(cur_cor[1] + size_x * (1 - (up_left[1] - ori_cor[1]) / (ori_cor[3] - ori_cor[1])))]
            overlaps.append([_up_left, _down_right])
        # print(overlaps)
        # exit()
        return overlaps, scale, transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.transform is not None:
            if self.image_set == 'train':
                if self.num_copys > 1:
                    imgs = []
                    targets = []
                    cur_cors = []
                    ori_cors = []
                    scales = []
                    flips = []
                    for i in range(self.num_copys):

                        _img, _target, _cordinate = self.transform(img, target)
                        # y_min, x_min, y_max, x_max, Y_min, X_min, Y_max, X_max, scale, flip = _cordinate
                        # print('y_min = {:}, x_min = {:}, y_max = {:},'
                        #       'x_max = {:}, Y_min = {:}, X_min = {:},'
                        #       ' Y_max = {:}, X_max = {:}, scale = {:}, flip = {:}'.format(y_min, x_min, y_max, x_max, Y_min, X_min, Y_max, X_max, scale, flip))
                        imgs.append(_img)
                        targets.append(_target)
                        cur_cors.append(_cordinate[:4])
                        ori_cors.append(_cordinate[4:8])
                        scales.append(_cordinate[-2])
                        flips.append(_cordinate[-1])
                    overlaps, scale, transform = self.get_overlaps(cur_cors, ori_cors, scales, flips)

                    return imgs, targets, overlaps, transform
                else:
                    img, target, cor = self.transform(img, target)
            else:
                img, target = self.transform(img, target)
        return img, target


    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=root)

def collate_fn2(batchs):
    _imgs = []
    _targets = []
    _overlaps = []
    flips = []
    for index, batch in enumerate(batchs):
        _overlaps.append([])
        imgs, targets, overlaps, transform = batch
        flips.append(transform[-1])
        for i in range(len(imgs)):
            # print('index: {:}, i: {:}'.format(index, i))
            _imgs.append(torch.unsqueeze(imgs[i], 0))
            _targets.append(torch.unsqueeze(targets[i], 0))
            _overlaps[index].append(overlaps[i])
    return torch.cat(_imgs, dim=0), torch.cat(_targets, dim=0), _overlaps, flips




def main():
    train_transform = train_et.ExtCompose([
        # et.ExtResize(size=opts.crop_size),
        # train_et.ExtRandomScale((0.5, 2.0)),
        train_et.New_ExtRandomCrop(size=(513, 513), pad_if_needed=True),
        train_et.ExtRandomHorizontalFlip(),
        train_et.ExtToTensor(),
        train_et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    dataset = VOCSegmentation(root=r'D:\datasets\VOC2012\VOCtrainval_11-May-2012', transform=train_transform, num_copys=2)
    return dataset
