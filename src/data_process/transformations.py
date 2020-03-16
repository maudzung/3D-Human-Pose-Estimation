import copy
import numpy as np
import sys
from functools import partial, reduce
import random

import cv2
import torch
import albumentations as albu

sys.path.append('../')


def denormalize_img(img):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img = (img * std + mean) * 255
    img = img.astype(np.uint8)

    return img


def get_img_transformations(configs, is_train_mode=True):
    img_transformations = None

    if configs.apply_img_transformations:
        if is_train_mode:
            img_transformations = albu.Compose([
                albu.RandomBrightnessContrast(brightness_limit=configs.train.brightness_limit,
                                              contrast_limit=configs.train.contrast_limit,
                                              p=configs.train.brightness_contrast_prob),
                # albu.HueSaturationValue(hue_shift_limit=configs.train.hue_shift_limit,
                #                         sat_shift_limit=configs.train.sat_shift_limit,
                #                         val_shift_limit=configs.train.val_shift_limit,
                #                         p=configs.train.hue_prob),
                # albu.ImageCompression(quality_lower=48, quality_upper=50, p=configs.train.jpeg_compression_prob),
                # albu.OneOf([albu.ToGray()], p=configs.train.togray_prob),
                albu.Resize(configs.inp_res, configs.inp_res, interpolation=cv2.INTER_CUBIC, p=1.),
                albu.Normalize(p=1.),
            ])
        else:
            img_transformations = albu.Compose([
                albu.Resize(configs.inp_res, configs.inp_res, interpolation=cv2.INTER_CUBIC, p=1.),
                albu.Normalize(p=1.)
            ])

    return img_transformations

