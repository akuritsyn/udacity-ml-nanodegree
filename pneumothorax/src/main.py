import argparse
from .utils.config import Config
from .utils.logger import logger, log

import torch
#import torch.optim as optim

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

    logger.setup(cfg.workdir, name='%s_fold%d' % (cfg.mode, cfg.fold))
    torch.cuda.set_device(cfg.gpu) # <----

    log(f'mode: {cfg.mode}')
    log(f'workdir: {cfg.workdir}')
    log(f'fold: {cfg.fold}')
    log(f'batch size: {cfg.batch_size}')
    #log(f'acc: {cfg.data.train.n_grad_acc}')


    model=factory.get_model(cfg)
    model.cuda() #need to do this consistently

    if cfg.mode == 'train':
        train(cfg, model)
    elif cfg.mode == 'valid':
        valid(cfg, model)
    elif cfg.mode == 'test':
        test(cfg, model)

def train(cfg, model):
 
    criterion = factory.get_loss(cfg)
    optimizer = factory.get_optim(cfg, model.parameters())

    best = {
        'loss': float('inf'),
        'score': 0.0,
        'epoch': -1,
    }

   
    if cfg.resume_from:
        model.load_state_dict(torch.load(cfg.resume_from, map_location=lambda storage, location: storage)["state_dict"])

    #scheduler = ReduceLROnPlateau(optim, mode="min", patience=3, verbose=True)
    scheduler = factory.get_scheduler(cfg, optimizer, best['epoch'])

    model_trainer = factory.Trainer(cfg, model, optimizer, scheduler, criterion, gpu='cuda:'+str(cfg.gpu))
    model_trainer.start()


if __name__ == '__main__':

    #torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = True

    main()
    # try:
    #     main()
    # except KeyboardInterrupt:
    #     print('Keyboard Interrupted')