import argparse
from .utils.config import Config

import torch
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
from .mixed_loss import MixedLoss

from . import factory


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'valid', 'test'])
    parser.add_argument('config')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--snapshot')
    parser.add_argument('--output') 
    return parser.parse_args()


def main():

    args = get_args()
    cfg = Config.fromfile(args.config)

    # copy command line args to cfg
    cfg.mode = args.mode
    cfg.debug = args.debug
    cfg.fold = args.fold
    #cfg.snapshot = args.snapshot
    cfg.output = args.output
    cfg.gpu = args.gpu

    torch.cuda.set_device(cfg.gpu)


    model=factory.get_model(cfg)
    model.cuda() #need to do this consistently

    if cfg.mode == 'train':
        train(cfg, model)
    elif cfg.mode == 'valid':
        valid(cfg, model)
    elif cfg.mode == 'test':
        test(cfg, model)

def train(cfg, model):
    
    if cfg.resume_from:
        #input_model_file="model_512_{}.pth".format(fold)
        #old_model=torch.load(cfg.resume_from)
        model.load_state_dict(torch.load(cfg.resume_from)["state_dict"])
    
    #size=1024 #512
    #fold=4
    #gpu_temp='cuda:'+str(cfg.gpu)
    #lr=5e-4
    #optimizer=optim.Adam(model.parameters(), lr=lr)
    optim = factory.get_optim(cfg, model.parameters())
    scheduler = ReduceLROnPlateau(optim, mode="min", patience=3, verbose=True)
    #criterion = factory.get_loss(cfg)
    criterion=MixedLoss(10, 2)
    #epochs=50
    #bs=4 #16
    #acc_steps=4 #32//bs

    # data
    model_trainer = factory.Trainer(cfg, model, optim, scheduler, criterion, gpu='cuda:'+str(cfg.gpu))
    model_trainer.start()




if __name__ == '__main__':

    #torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = True

    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard Interrupted')