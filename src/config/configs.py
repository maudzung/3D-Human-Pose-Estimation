import torch
import os
import numpy as np
import datetime

from easydict import EasyDict as edict

import sys
sys.path.append('../')

from utils.misc import make_folder
from data_process.h36m_properties import h36m_keypoints


#note for re-produce the results with seed random

def get_configs():
    configs = edict()

    ####################################################################
    ##############     Model configs            ###################
    ####################################################################
    configs.model = edict()
    configs.model.backbone = 'densenet121'
    configs.model.use_bn = True
    configs.model.pretrained = True
    configs.model.dropout_p = 0.5

    ####################################################################
    ############## General configurations ############################
    ############## Data, log and Checkpoint ############################
    ####################################################################
    configs.task = '3D_hpe'

    configs.saved_fn = '{}_{}'.format(configs.task, configs.model.backbone)

    configs.working_dir = '../../'
    configs.checkpoints_dir = os.path.join(configs.working_dir, 'checkpoints', configs.task)
    configs.logs_dir = os.path.join(configs.working_dir, 'logs', configs.task)
    configs.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs.n_gpus = torch.cuda.device_count()
    # configs.pin_memory = True if torch.cuda.is_available() else False
    configs.pin_memory = False

    ####################################################################
    ##############     Data configs            ###################
    ####################################################################
    configs.datasets_dir = os.path.join('/media/nmdung/SSD_4TB_Disk_1/kpts_works/hpe_3D', 'datasets')
    configs.n_keypoints = len(h36m_keypoints)
    configs.protocol_name = 'protocol_1'
    ####################################################################
    ##############     Losses configs            ###################
    ####################################################################
    configs.loss = edict()
    configs.loss.loss_type = 'RMSE'  # MSE or SmoothL1Loss or MAE

    ####################################################################
    ##############     Running configs            ###################
    ####################################################################
    configs.n_images = None
    configs.n_workers = 8
    configs.batch_size = 32 * configs.n_gpus
    configs.drop_last = True
    configs.inp_res = 256
    configs.is_fixed_cropsize = False
    configs.fixed_cropsize = 440
    configs.print_freq = 200
    configs.verbose = True
    configs.apply_img_transformations = True

    ####################################################################
    ##############     Training strategy            ###################
    ####################################################################
    configs.train = edict()

    configs.train.subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
    configs.train.cameras = ['54138969', '55011271', '60457274', '58860488']
    configs.train.act_names = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
                                'Sitting', 'SittingDown', 'Smoking', 'TakingPhoto', 'Waiting', 'Walking', 'WalkingDog',
                                'WalkingTogether']
    configs.train.act_ids = ['1', '2']

    # configs.train.subjects = ['S1']
    # configs.train.cameras = ['55011271']
    # configs.train.act_names = ['Walking']
    # configs.train.act_ids = ['1', '2']

    configs.train.n_epochs_total = 200
    configs.train.n_epochs_warm = 0

    configs.train.warmup_lr = 1e-3
    configs.train.warmup_momentum = 0.9
    configs.train.warmup_weight_decay = 0.
    configs.train.warmup_nesterov = False
    configs.train.warmup_optimizer_type = 'adam'

    configs.train.lr = 1e-2
    configs.train.minimum_lr = 1e-7
    configs.train.momentum = 0.9
    configs.train.weight_decay = 0.
    configs.train.nesterov = True
    configs.train.optimizer_type = 'adam'  # or sgd or adam

    configs.train.lr_milestones = [i for i in range(5, configs.train.n_epochs_total, 5)]
    configs.train.lr_type = 'multi_step_lr'  # or 'multi_step_lr' or plateau
    configs.train.lr_factor = 0.5
    configs.train.lr_patience = 3
    configs.train.earlystop_patience = None


    # Augment data
    configs.train.hflip_prob = 0.
    configs.train.scale_range = None # (1.0, 1.2)
    configs.train.fill_values = (124, 116, 104)
    configs.train.limited_angle = 10.
    configs.train.rotate_prob = 0.05

    configs.train.brightness_contrast_prob = 0.2
    configs.train.brightness_limit = 0.1
    configs.train.contrast_limit = 0.1

    configs.train.hue_prob = 0.5
    configs.train.hue_shift_limit = 10
    configs.train.sat_shift_limit = 10
    configs.train.val_shift_limit = 10

    configs.train.jpeg_compression_prob = 0.1
    configs.train.togray_prob = 0.01

    ####################################################################
    ##############     Validation configurations     ###################
    ####################################################################
    configs.val = edict()
    configs.val.subjects = ['S9', 'S11']
    configs.val.cameras = ['54138969', '55011271', '60457274', '58860488']
    configs.val.act_names = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
                             'Sitting', 'SittingDown', 'Smoking', 'TakingPhoto', 'Waiting', 'Walking', 'WalkingDog',
                             'WalkingTogether']
    configs.val.act_ids = ['1', '2']

    # Augment data
    configs.val.hflip_prob = 0.
    configs.val.scale_range = None
    configs.val.fill_values = (124, 116, 104)

    ####################################################################
    ##############     Evaluation configurations     ###################
    ####################################################################
    configs.use_best_checkpoint = True
    if configs.use_best_checkpoint:
        configs.saved_weight_name = os.path.join(configs.checkpoints_dir, '{}_best.pth'.format(configs.saved_fn))
        configs.results_dir = os.path.join(configs.working_dir, 'Results', configs.task, configs.model.backbone,
                                           'best_checkpoint')
    else:
        configs.saved_weight_name = os.path.join(configs.checkpoints_dir, '{}.pth'.format(configs.saved_fn))
        configs.results_dir = os.path.join(configs.working_dir, 'Results', configs.task, configs.model.backbone,
                                           'last_checkpoint')

    configs.is_debug = False

    make_folder(configs.checkpoints_dir)
    make_folder(configs.logs_dir)

    return configs


if __name__ == "__main__":
    configs = get_configs()
    print(configs)

    print(datetime.date.today())
    print(datetime.datetime.now().year)
