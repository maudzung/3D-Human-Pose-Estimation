import torch
import sys

sys.path.append('../')

from models.models import simple_model

def get_model(configs):
    model = simple_model(n_keypoints=17, hidden_neurons=1024)
    return model


def get_optimizer(configs, model, is_warm_up):
    # trainable_vars = [param for param in model.module.parameters() if param.requires_grad]
    if is_warm_up:
        lr = configs.train.warmup_lr
        momentum = configs.train.warmup_momentum
        weight_decay = configs.train.warmup_weight_decay
        nesterov = configs.train.warmup_nesterov
        optimizer_type = configs.train.warmup_optimizer_type
    else:
        lr = configs.train.lr
        momentum = configs.train.momentum
        weight_decay = configs.train.weight_decay
        nesterov = configs.train.nesterov
        optimizer_type = configs.train.optimizer_type
    if configs.n_gpus > 1:
        if optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(model.module.parameters(), lr=lr, momentum=momentum,
                                        weight_decay=weight_decay, nesterov=nesterov)
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(model.module.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            assert False, "Unknown optimizer type"
    else:
        if optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                        weight_decay=weight_decay, nesterov=nesterov)
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            assert False, "Unknown optimizer type"

    return optimizer


if __name__ == '__main__':
    from config.configs import get_configs
    configs = get_configs()
    model = get_model(configs)