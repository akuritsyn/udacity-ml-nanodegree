import argparse
from .utils.config import Config
from .utils.logger import logger, log

import torch
import pandas as pd
from tqdm import tqdm
import pickle
import gc
import os

from . import factory
from . import predictor
from .rle_functions import run_length_encode


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('config')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--fold', type=int) #, required=True
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output') 
    return parser.parse_args()


def main():

    args = get_args()
    cfg = Config.fromfile(args.config)

    # copy command line args to cfg
    cfg.mode = args.mode
    cfg.debug = args.debug
    cfg.fold = args.fold
    cfg.output = args.output
    cfg.gpu = args.gpu

    if cfg.mode == 'train':
        logger.setup(cfg.workdir, name='%s_fold%d' % (cfg.mode, cfg.fold))
    elif cfg.mode == 'test':
        logger.setup(cfg.workdir, name='%s' % (cfg.mode))

    log(f'mode: {cfg.mode}')
    log(f'workdir: {cfg.workdir}')
    log(f'fold: {cfg.fold}')
    log(f'batch size: {cfg.batch_size}')
    log(f'acc: {cfg.data.train.n_grad_acc}')

    torch.cuda.set_device(cfg.gpu) 
    model=factory.get_model(cfg)
    model.cuda() #need to do this consistently
    #device = torch.device("cuda:"+str(cfg.gpu))
    #model.to(device)

    if cfg.mode == 'train':
        train(cfg, model)
    elif cfg.mode == 'test':
        test(cfg, model)


def test(cfg, model):
    model.eval()

    df=pd.read_csv(cfg.data.test.sample_submission_file)
    if cfg.debug:
        df=df.head(30)
        log('Debug mode: loading first %d records of test data' % df.shape[0])

    loader_test=predictor.get_dataloader(cfg.data.test, df)
    log('Test data: loaded %d records' % len(loader_test.dataset))
    pixel_probs = predictor.get_pixel_probabilities(cfg, model, loader_test)
    
    loader_test=predictor.get_dataloader(cfg.data.test, df, hflip=True)
    log('H-flipped test data for TTA: loaded %d records' % len(loader_test.dataset))
    pixel_probs_hflipped = predictor.get_pixel_probabilities(cfg, model, loader_test, hflip=True)
    
    for i in range(len(pixel_probs)):
        pixel_probs[i]=(pixel_probs[i]+pixel_probs_hflipped[i])/2.
        
    del pixel_probs_hflipped
    gc.collect()

    if cfg.data.test.output_file_probabilty_name:
        with open(os.path.join(cfg.workdir,cfg.data.test.output_file_probabilty_name), 'wb') as f:
            pickle.dump(pixel_probs, f)

    # Generate submission file
    encoded_pixels = []
    for probability in pixel_probs:
        predict, num_predict = predictor.post_process(cfg.data.test, probability)
        if num_predict == 0:
            encoded_pixels.append('-1')
        else: 
            r = run_length_encode(predict)
            encoded_pixels.append(r)
    df['EncodedPixels'] = encoded_pixels
    df.to_csv(os.path.join(cfg.workdir,cfg.data.test.submission_file_name), columns=['ImageId', 'EncodedPixels'], index=False)
    log(f'Saved predictions to {cfg.data.test.submission_file_name}')


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

    scheduler = factory.get_scheduler(cfg, optimizer, best['epoch'])

    model_trainer = factory.Trainer(cfg, model, optimizer, scheduler, criterion, gpu='cuda:'+str(cfg.gpu))
    model_trainer.start()


if __name__ == '__main__':

    #torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = True

    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard Interrupted')