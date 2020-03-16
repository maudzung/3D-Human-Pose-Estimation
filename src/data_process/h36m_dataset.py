import copy
import os
import numpy as np
import sys
from glob import glob
import time
import random

import cv2
from torch.utils.data import Dataset

sys.path.append('../')

from data_process.utils import load_raw_img
from utils.misc import make_folder


class H36M_Dataset(Dataset):
    def __init__(self, dataset_dir, subjects, cameras, act_names, act_ids, protocol_name, n_images=None, inp_res=368,
                 is_fixed_cropsize=False, fixed_cropsize=440, scale_range=None, img_transformations=None,
                 is_train_mode=False, is_debug_mode=False):
        self.dataset_dir = dataset_dir
        self.subjects = subjects
        self.cameras = cameras
        self.act_names = act_names
        self.act_ids = act_ids
        self.protocol_name = protocol_name

        self.inp_res = inp_res
        self.is_fixed_cropsize = is_fixed_cropsize
        self.fixed_cropsize = fixed_cropsize
        self.scale_range = scale_range
        self.img_transformations = img_transformations
        self.is_train_mode = is_train_mode
        self.is_debug_mode = is_debug_mode

        print('loading image paths...preparing')
        self.image_paths = self.load_image_paths()
        print('loading image paths...done')
        print('number of images: {}'.format(len(self.image_paths)))

        print('loading annotations...preparing')
        self.annos = self.load_annos()
        print('loading annotations...done')

        random.shuffle(self.image_paths)
        # self.image_paths = sorted(self.image_paths)
        if n_images:
            self.image_paths = self.image_paths[:n_images]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        # img = load_raw_img(img_path)
        # img_h, img_w = img.shape[0], img.shape[1]
        img_h, img_w = 1000, 1000

        img_path_parts = img_path.split('/')
        img_fn = img_path_parts[-1][:-4]
        cmr = img_path_parts[-2]
        act_fullname = img_path_parts[-4]
        sbj = img_path_parts[-5]
        img_annos = self.annos[sbj][act_fullname][img_fn]
        org_bbox = img_annos['bboxes'][cmr] # poses_3d, poses_3d_mono, poses_3d_mono_uni, poses_2d_mono
        pose_3d_mono_annos = img_annos['poses_3d_mono'][cmr]
        poses_2d_mono_annos = img_annos['poses_2d_mono'][cmr]

        if self.is_fixed_cropsize:
            human_center = self.get_human_center(org_bbox)
            croped_bbox = self.get_human_bbox_fixed_cropsize(human_center, fixed_cropsize=self.fixed_cropsize)
            xmin, ymin, xmax, ymax = croped_bbox
        else:
            xmin, ymin, xmax, ymax = org_bbox
            croped_bbox = self.align_bbox_to_square(org_bbox, img_h, img_w) # Align the w, h of bbox to get the square image (ignore)


        pose_3d_mono_annos = pose_3d_mono_annos - pose_3d_mono_annos[0,:] #Pelvis is the root joint
        pose_3d_mono_annos = pose_3d_mono_annos[1:, :].reshape(-1)
        # pose_3d_mono_annos = pose_3d_mono_annos[1:, :] # Take all joint, except the Pelvis joint (First joint)
        poses_2d_mono_annos = poses_2d_mono_annos - np.array([xmin, ymin]).reshape(1, 2) # 2D keypoints in the images)
        poses_2d_mono_annos = poses_2d_mono_annos / np.array([xmax-xmin, ymax-ymin], dtype=np.float).reshape(1,2)
        poses_2d_mono_annos = poses_2d_mono_annos.reshape(-1)

        # img = self.crop_human_bbox(img, xmin, ymin, xmax, ymax) # Crop the image based on the bbox
        # if self.img_transformations:
        #     img = self.img_transformations(img)


        # return img_path, img, croped_bbox, poses_2d_mono_annos, pose_3d_mono_annos
        return img_path, croped_bbox, poses_2d_mono_annos, pose_3d_mono_annos

    def load_image_paths(self):
        image_paths = []
        for cmr in self.cameras:
            for sbj in self.subjects:
                for act_name in self.act_names:
                    for act_id in self.act_ids:
                        act_fullname = '{}_{}'.format(act_name, act_id)
                        image_paths += glob(os.path.join(self.dataset_dir, 'human36M', self.protocol_name, sbj,
                                                         act_fullname, 'imageSequence', cmr, '*.jpg'))
        return image_paths

    def load_annos(self):
        annos = {}
        for sbj in self.subjects:
            annos[sbj] = {}
            for act_name in self.act_names:
                for act_id in self.act_ids:
                    act_fullname = '{}_{}'.format(act_name, act_id)
                    annos_path = os.path.join(self.dataset_dir, 'human36M', self.protocol_name, sbj,
                                              act_fullname, 'annos', 'selected_annotations.npy')
                    annos[sbj][act_fullname] = np.load(annos_path, allow_pickle=True).item()
        return annos

    def get_human_center(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        h, w = ymax - ymin, xmax - xmin
        x_center, y_center = xmin + int(w / 2), ymin + int(h / 2)

        return [x_center, y_center]

    def get_human_bbox_fixed_cropsize(self, human_center, fixed_cropsize=440):
        fixed_cropsize_half = int(fixed_cropsize / 2)
        xmin = human_center[0] - fixed_cropsize_half
        ymin = human_center[1] - fixed_cropsize_half
        xmax = human_center[0] + fixed_cropsize_half
        ymax = human_center[1] + fixed_cropsize_half
        return [xmin, ymin, xmax, ymax]

    def align_bbox_to_square(self, bbox, img_h, img_w):
        """
        Align bbox to a same dimension
        :param bbox:
        :param max_hw:
        :return:
        """
        xmin, ymin, xmax, ymax = bbox
        h, w = ymax - ymin, xmax - xmin
        x_center, y_center = xmin + int(w / 2), ymin + int(h / 2)
        if h > w:
            # Choose h dimention, change xmin, xmax
            xmin = x_center - int(h / 2)
            xmax = x_center + int(h / 2)
            if xmin < 0:
                xmin = 0
            if xmax > img_w:
                xmax = img_w
        elif h < w:
            # Choose w dimention, change ymin, ymax
            ymin = y_center - int(w / 2)
            ymax = y_center + int(w / 2)
            if ymin < 0:
                ymin = 0
            if ymax > img_h:
                ymax = img_h
        bbox = [xmin, ymin, xmax, ymax]
        # assert (bbox[2] - bbox[0]) == (bbox[3] - bbox[1])  # only correct when xmin, ymin > 0

        return bbox

    def crop_human_bbox(self, img, xmin, ymin, xmax, ymax):
        return img[ymin:ymax, xmin:xmax, :]


if __name__ == '__main__':
    from config.configs import get_configs

    configs = get_configs()

    h36m_dataset = H36M_Dataset(configs.datasets_dir, configs.train.subjects, configs.train.cameras,
                                configs.train.act_names, configs.train.act_ids, configs.protocol_name,
                                configs.n_images, configs.inp_res, configs.fixed_cropsize,
                                configs.train.scale_range, img_transformations=None,
                                is_train_mode=False, is_debug_mode=False)
