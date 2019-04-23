import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt

from torch.utils import data
from ptsemseg.augmentations import *

import cv2 as cv
from torchvision import transforms


class myLoader(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=512,
        augmentations=None,
        img_norm=True,
    ):
        self.root = root
        self.split = split
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.mean = np.array([115.3165639, 83.02458143, 81.95442675])
        self.n_classes = 6
        self.files = collections.defaultdict(list)

        for split in ["train", "test", "val"]:
            file_list = os.listdir(root + "/" + split)
            self.files[split] = file_list

        # self.tf = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.45222182, 0.32558659, 0.32138991],
        #                                                    [0.21074223, 0.14708663, 0.14242824])])
        # self.tf_no_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.45222182, 0.32558659, 0.32138991],
        #                                                                           [1,1,1])])
        self.tf = transforms.ToTensor()
        self.tf_no_train = transforms.ToTensor()

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + "/" + self.split + "/" + img_name
        lbl_path = self.root + "/" + self.split + "_labels/" + img_name

        img = cv.cvtColor(cv.imread(img_path, -1), cv.COLOR_BGR2RGB)
        lbl = cv.imread(lbl_path, -1)
        # im = Image.open(im_path)
        # lbl = Image.open(lbl_path)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        if self.img_size == ('same', 'same'):
            pass
        else:
            #opencv resize,(width,heigh)
            img=cv.resize(img,(self.img_size[1],self.img_size[0]))
            lbl = cv.resize(lbl, (self.img_size[1], self.img_size[0]))

            # img = img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
            # lbl = lbl.resize((self.img_size[0], self.img_size[1]))
        if self.split=="train":
            img = self.tf(img)
        else:
            img=self.tf_no_train(img)
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def decode_segmap(self, temp, plot=False):
        Imps = [255,255,255]
        Building = [0,0,255]
        Lowvg = [0,255,255]
        Tree = [0,255,0]
        Car= [255,255,0]
        bg = [255,0,0]

        label_colours = np.array(
            [
                Imps,
                Building,
                Lowvg,
                Tree,
                Car,
                bg,
            ]
        )
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]
        # rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb = np.zeros((temp.shape[0], temp.shape[1], 3), dtype=np.uint8)
        # rgb[:, :, 0] = r / 255.0
        # rgb[:, :, 1] = g / 255.0
        # rgb[:, :, 2] = b / 255.0
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        return rgb


