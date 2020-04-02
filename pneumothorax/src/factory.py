import torch
from torch import nn
import torch.optim
from torch.optim import lr_scheduler

import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.torch import ToTensor
# from torchvision.transforms import ToTensor

from .mixed_loss import MixedLoss
from .utils.logger import log


def get_transforms(tfms):
    def get_object(transform):
        if hasattr(albu, transform.name):
            return getattr(albu, transform.name)
        else:
            return eval(transform.name)

    transforms = [get_object(transform)(*transform.args, **transform.params) for transform in tfms]
    return albu.Compose(transforms)


def get_model(cfg):
    log(f'Model: {cfg.model.name}')
    log(f'Pretrained: {cfg.model.pretrained}')
    if cfg.model.name == 'unet_resnet34':
        model = smp.Unet("resnet34",
                         encoder_weights="imagenet", activation=None)

    return model


def get_optimizer(cfg, parameters):
    optim = getattr(torch.optim, cfg.optim.name)(parameters,
                                                 **cfg.optim.params)
    log(f'Optimizer: {cfg.optim.name}')
    return optim


def get_loss(cfg):
    log('Loss: %s' % cfg.loss.name)

    if cfg.loss.name == 'MixedLoss':
        loss = MixedLoss(cfg.loss.params.alpha, cfg.loss.params.gamma)
        log(f'alpha: {cfg.loss.params.alpha}, gamma: {cfg.loss.params.gamma}')
    else:
        loss = getattr(nn, cfg.loss.name)(**cfg.loss.params)

    return loss


def get_scheduler(cfg, optim, last_epoch):
    log(f'Scheduler: {cfg.scheduler.name}')
    if cfg.scheduler.name == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optim,
            **cfg.scheduler.params,
        )
        scheduler.last_epoch = last_epoch
    else:
        scheduler = getattr(lr_scheduler, cfg.scheduler.name)(
            optim,
            last_epoch=last_epoch,
            **cfg.scheduler.params,
        )

    log(f'Last_epoch: {last_epoch}')
    return scheduler
