import time
import warnings

import torch
from torch import nn
import torch.optim
from torch.optim import lr_scheduler
#from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.optim as optim
import torch.backends.cudnn as cudnn

warnings.filterwarnings("ignore")
import segmentation_models_pytorch as smp

from .mixed_loss import MixedLoss
from .metrics import Meter
from .metrics import epoch_log
from .dataset import provider
from .utils.logger import log

# ---> Need to fix paths
#input_dir_path='../input/1024-s2/'
#data_folder = input_dir_path+"train"
#test_data_folder = input_dir_path+"test"
#sample_submission_path = '../input/stage_2_sample_submission.csv'
#train_rle_path = '../input/stage_2_train.csv' #'train-rle.csv' #


def get_model(cfg):
    log(f'model: {cfg.model.name}')
    log(f'pretrained: {cfg.model.pretrained}')
    if cfg.model.name=='unet_resnet34':
        model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)

    return model


def get_optim(cfg, parameters):
    optim = getattr(torch.optim, cfg.optim.name)(parameters, **cfg.optim.params)
    log(f'optim: {cfg.optim.name}')
    return optim


def get_loss(cfg):
    log('loss: %s' % cfg.loss.name)

    if cfg.loss.name=='MixedLoss':
        loss = MixedLoss(cfg.loss.params.alpha, cfg.loss.params.gamma)
        log(f'alpha: {cfg.loss.params.alpha}, gamma: {cfg.loss.params.gamma}')
    else:
        loss = getattr(nn, cfg.loss.name)(**cfg.loss.params)
    
    return loss  


def get_scheduler(cfg, optim, last_epoch):
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

    log(f'last_epoch: {last_epoch}')
    return scheduler


class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self, cfg, model, optimizer, scheduler, criterion, gpu):
        self.num_workers = cfg.num_workers
        self.workdir=cfg.workdir
        self.fold = cfg.fold
        self.size = cfg.img_size
        self.accumulation_steps = cfg.data.train.n_grad_acc
        self.lr = cfg.optim.params.lr
        self.num_epochs = cfg.epochs
        self.batch_size = {"train": cfg.batch_size, "valid": cfg.data.valid.loader.batch_size}
        self.best_loss = float("inf")
        self.phases = ["train", "valid"]
        self.device = torch.device(gpu)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.criterion = criterion
        self.optimizer = optimizer #.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = scheduler
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                fold=self.fold,
                total_folds=cfg.n_fold,
                data_folder=cfg.data.train.imgdir,
                df_path=cfg.data.train.train_rle_path,
                phase=phase,
                size=self.size,
                mean=cfg.normalize.mean, #(0.485, 0.456, 0.406),
                std=cfg.normalize.std,  #(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        
    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
#         tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
#             tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            val_loss = self.iterate(epoch, "valid")
            self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, self.workdir+"/model_{}_{}.pth".format(self.size,self.fold))
            print()