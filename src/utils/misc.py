import os
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # or os.makedirs(folder_name, exist_ok=True)


def save_checkpoint(checkpoints_dir, saved_fn, saved_state, is_best, logger=None):
    if is_best:
        save_path = os.path.join(checkpoints_dir, '{}_best.pth'.format(saved_fn))
    else:
        save_path = os.path.join(checkpoints_dir, '{}.pth'.format(saved_fn))

    torch.save(saved_state, save_path)
    if logger:
        logger.info('save the checkpoint at {}'.format(save_path))


def adjust_lr(optimizer, epoch, lr, configs, lr_patience_count):
    if configs.train.lr_type == 'fixed':
        pass
    elif configs.train.lr_type == 'multi_step_lr':
        """Sets the learning rate to the initial LR decayed by schedule"""
        if epoch in configs.train.lr_milestones:
            lr *= configs.train.lr_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    elif configs.train.lr_type == 'plateau':
        if lr_patience_count >= configs.train.lr_patience:
            lr *= configs.train.lr_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            lr_patience_count = 0
    else:
        assert False, "No lr adjust defined"
    if lr < configs.train.minimum_lr:
        print('lr reaches the minimum value')
        lr = configs.train.minimum_lr

    return lr, lr_patience_count


