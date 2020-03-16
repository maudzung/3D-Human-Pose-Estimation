import argparse
import time
import os
import numpy as np
import sys
import copy
import random

import torch
from torchsummary import summary

sys.path.append('../')

from data_process.h36m_dataloader import H36m_dataloader
from train.h36m_train_utils import get_model, get_optimizer
from losses.losses import compute_loss
from utils.misc import AverageMeter, save_checkpoint, adjust_lr
from utils.logger import create_logger
from config.configs import get_configs


seed = 2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train(train_loader, model, optimizer, epoch, configs, logger):
    time_infor = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    start_time = time.time()
    for b_idx, (img_path, croped_bbox, poses_2d_mono_annos, pose_3d_mono_annos) in enumerate(train_loader):
        # zero the parameter gradients
        optimizer.zero_grad()
        b_size = poses_2d_mono_annos.size(0)
        # img = img.to(configs.device, non_blocking=True)
        # img = img.to(configs.device)

        poses_2d_mono_annos = poses_2d_mono_annos.to(configs.device).float()
        pose_3d_mono_annos = pose_3d_mono_annos.to(configs.device).float()

        # compute output
        with torch.set_grad_enabled(True):
            pose_3d_preds = model(poses_2d_mono_annos)
            # total_loss = compute_loss(pose_3d_preds, pose_3d_mono_annos, configs.loss.loss_type, configs.device)
            total_loss = compute_loss(pose_3d_preds, pose_3d_mono_annos, b_size)

            # compute gradient and do SGD step
            total_loss.backward()
            optimizer.step()
        losses.update(total_loss.item(), b_size)
        # measure elapsed time
        time_infor.update(time.time() - start_time)

        if (b_idx + 1) % configs.print_freq == 0:
            print_string = '>>> Epoch: [{}/{}][{}/{}]\t'.format(epoch + 1, configs.train.n_epochs_total, b_idx,
                                                                len(train_loader))
            print_string += 'Loss value: {:.2f}mm, avg: {:.2f}mm\t'.format(losses.val, losses.avg)
            print_string += 'time: {:.2f}s\t'.format(time_infor.val)
            # print(print_string)
            logger.info(print_string)
    logger.info('>>> End train, avg loss {:.2f}mm --- time: {:.2f}s'.format(losses.avg, time.time() - start_time))
    return losses.avg


def validate(val_loader, model, epoch, configs, logger):
    time_infor = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    start_time = time.time()
    for b_idx, (img_path, croped_bbox, poses_2d_mono_annos, pose_3d_mono_annos) in enumerate(val_loader):
        b_size = poses_2d_mono_annos.size(0)

        poses_2d_mono_annos = poses_2d_mono_annos.to(configs.device).float()
        pose_3d_mono_annos = pose_3d_mono_annos.to(configs.device).float()

        # compute output
        with torch.set_grad_enabled(False):
            pose_3d_preds = model(poses_2d_mono_annos)
            # total_loss = compute_loss(pose_3d_preds, pose_3d_mono_annos, configs.loss.loss_type, configs.device)
            total_loss = compute_loss(pose_3d_preds, pose_3d_mono_annos, b_size)

        losses.update(total_loss.item(), b_size)
        time_infor.update(time.time() - start_time)

        # measure elapsed time
        if (b_idx + 1) % configs.print_freq == 0:
            print_string = '>>> Epoch: [{}/{}][{}/{}]\t'.format(epoch +  1, configs.train.n_epochs_total, b_idx,
                                                                len(val_loader))
            print_string += 'Loss value: {:.2f}mm, avg: {:.2f}mm\t'.format(losses.val, losses.avg)
            print_string += 'time: {:.2f}s\t'.format(time_infor.val)
            logger.info(print_string)
            # print(print_string)
    logger.info('>>> End validate, avg loss {:.2f}mm --- time: {:.2f}s'.format(losses.avg, time.time() - start_time))

    return losses.avg


def main():
    configs = get_configs()
    logger = create_logger(configs.logs_dir, configs.saved_fn)
    logger.info('>>> Created a new logger')
    logger.info('>>> configs: {}'.format(configs))

    logger.info(">>> Loading dataset...")
    # Create dataloader
    train_loader, val_loader = H36m_dataloader(configs)

    # model
    model = get_model(configs).to(configs.device)

    if configs.n_gpus > 1:
        model = torch.nn.DataParallel(model)
    # load pretrained
    if configs.train.n_epochs_warm > 0:
        optimizer = get_optimizer(configs, model, is_warm_up=True)

        for epoch in range(configs.train.n_epochs_warm):
            # train for one epoch
            logger.info(
                '>>> warm-up training with the frozen base model...epoch: {}/{}'.format(epoch + 1,
                                                                                        configs.train.n_epochs_warm))
            train_loss = train(train_loader, model, optimizer, epoch, configs, logger)

            # evaluate on validation set
            val_loss = validate(val_loader, model, epoch, configs, logger)
            saved_state = {
                'epoch': epoch,
                'configs': configs,
                'loss': val_loss,
                'optimizer': copy.deepcopy(optimizer.state_dict()),
            }
            if configs.n_gpus > 1:
                saved_state['state_dict'] = copy.deepcopy(model.module.state_dict())
            else:
                saved_state['state_dict'] = copy.deepcopy(model.state_dict())

            save_checkpoint(configs.checkpoints_dir, configs.saved_fn, saved_state, is_best=False, logger=None)

    # Release all weights
    if configs.n_gpus > 1:
        for param in model.module.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

    # summary(model.cuda(), (3, 224, 224))

    optimizer = get_optimizer(configs, model, is_warm_up=False)

    best_val_loss = np.inf
    lr = configs.train.lr
    lr_patience_count = 0
    earlystop_count = 0
    logger.info('>>> Train full network')
    for epoch in range(configs.train.n_epochs_warm, configs.train.n_epochs_total):
        logger.info('{}'.format('*-' * 40))
        logger.info('{} {}/{} {}'.format('=' * 35, epoch + 1, configs.train.n_epochs_total, '=' * 35))
        logger.info('{}'.format('*-' * 40))

        logger.info('>>> learning rate: {}'.format(lr))
        # train for one epoch
        train_loss = train(train_loader, model, optimizer, epoch, configs, logger)

        # evaluate on validation set
        val_loss = validate(val_loader, model, epoch, configs, logger)

        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        lr_patience_count = 0 if is_best else (lr_patience_count + 1)
        earlystop_count = 0 if is_best else (earlystop_count + 1)

        print_string = '>>> train_loss: {:.2f}mm, val_loss: {:.2f}mm, best_val_loss: {:.2f}mm\t'.format(train_loss, val_loss,
                                                                                                  best_val_loss)
        print_string += '||| \t lr_patience_count: {}, earlystop_count: {}'.format(lr_patience_count, earlystop_count)

        logger.info(print_string)
        saved_state = {
            'epoch': epoch,
            'configs': configs,
            'loss': val_loss,
            'lr': lr,
            'optimizer': copy.deepcopy(optimizer.state_dict()),
        }
        if configs.n_gpus > 1:
            saved_state['state_dict'] = copy.deepcopy(model.module.state_dict())
        else:
            saved_state['state_dict'] = copy.deepcopy(model.state_dict())
        save_checkpoint(configs.checkpoints_dir, configs.saved_fn, saved_state, is_best=is_best, logger=None)

        lr, lr_patience_count = adjust_lr(optimizer, epoch, lr, configs, lr_patience_count)

        if configs.train.earlystop_patience:
            if configs.train.earlystop_patience <= earlystop_count:
                logger.info('>>> Early stopping!!!')
                break
            else:
                logger.info('>>> Continue training..., earlystop_count: {}'.format(earlystop_count))


if __name__ == '__main__':
    main()
