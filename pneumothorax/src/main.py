import argparse
import pandas as pd
# from tqdm import tqdm
import pickle
import gc
import os
import time

import torch
import albumentations as albu

from .utils.config import Config
from .utils.logger import logger, log
from .utils import util
from . import factory
from . import predictor
from .rle_functions import run_length_encode
from .dataset import provider
from .metrics import Meter
from .metrics import epoch_log

import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('config')
    parser.add_argument('--debug', type=int)  # action='store_true'
    parser.add_argument('--fold', type=int)   # , required=True
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

    log(f'Mode: {cfg.mode}')
    log(f'Workdir: {cfg.workdir}')
    log(f'Fold: {cfg.fold}')

    util.set_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.set_device(cfg.gpu)
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    model = factory.get_model(cfg)
    model.cuda()
    # device = torch.device("cuda:"+str(cfg.gpu))
    # model.to(device)

    if cfg.mode == 'train':
        train(cfg, model)
    elif cfg.mode == 'test':
        test(cfg, model)


def test(cfg, model):
    model.eval()

    log(f'Batch size: {cfg.data.test.loader.batch_size}')

    df = pd.read_csv(cfg.data.test.sample_submission_file)
    if cfg.debug:
        df = df.head(cfg.debug)  # 30
        log('Debug mode: loading first %d records of test data' % df.shape[0])

    loader_test = predictor.get_dataloader_test(cfg.data.test, df)
    log('Test data: loaded %d records' % len(loader_test.dataset))
    pixel_probs = predictor.get_pixel_probabilities(cfg, model, loader_test)

    loader_test = predictor.get_dataloader_test(cfg.data.test, df, hflip=True)
    log(f'H-flipped test data for TTA: loaded \
        {len(loader_test.dataset)} records')
    pixel_probs_hflipped = predictor.get_pixel_probabilities(
        cfg, model, loader_test, hflip=True)

    for i in range(len(pixel_probs)):
        pixel_probs[i] = (pixel_probs[i]+pixel_probs_hflipped[i])/2.

    del pixel_probs_hflipped
    gc.collect()

    if cfg.data.test.output_file_probabilty_name:
        with open(os.path.join(cfg.workdir,
                  cfg.data.test.output_file_probabilty_name), 'wb') as f:
            pickle.dump(pixel_probs, f)

    # Generate submission file
    encoded_pixels = []
    for probability in pixel_probs:
        predict, num_predict = predictor.post_process(
                            probability, cfg.data.test.prob_threshold,
                            cfg.data.test.min_object_size)
        if num_predict == 0:
            encoded_pixels.append('-1')
        else:
            r = run_length_encode(predict)
            encoded_pixels.append(r)
    df['EncodedPixels'] = encoded_pixels
    df.to_csv(os.path.join(cfg.workdir, cfg.data.test.submission_file_name),
              columns=['ImageId', 'EncodedPixels'], index=False)
    log(f'Saved predictions to {cfg.data.test.submission_file_name}')


def train(cfg, model):

    log(f'Batch size: {cfg.batch_size}')
    log(f'Gradient accumulation: {cfg.n_grad_acc}')

    criterion = factory.get_loss(cfg)
    optimizer = factory.get_optimizer(cfg, model.parameters())

    best = {
        'loss': float('inf'),
        'score': 0,
        'epoch': -1,
    }

    loader_train = provider(cfg, phase='train')
    loader_valid = provider(cfg, phase='valid')

    if cfg.resume_from:
        state = torch.load(cfg.resume_from,
                           map_location=lambda storage, location: storage)
        model.load_state_dict(state["state_dict"])
        best['epoch'] = state['epoch']
        best['loss'] = state['best_loss']
        best['score'] = state['best_score']
        log(f'Resuming training from {cfg.resume_from} - starting epoch {best["epoch"]+1}')
        optimizer.load_state_dict(state['optimizer'])
        log('Loading optimizer weights too')
        # model.cuda()

    if cfg.retrain_from:
        model.load_state_dict(torch.load(cfg.retrain_from,
                              map_location=lambda storage,
                              location: storage)["state_dict"])
        log(f'Using weights from {cfg.retrain_from} - starting epoch {best["epoch"]+1}')
        # model.cuda()

    scheduler = factory.get_scheduler(cfg, optimizer, best['epoch'])

    phases = ['train', 'valid']

    training_summary = dict(
        epoch=[],
        losses={phase: [] for phase in phases},
        dice_scores={phase: [] for phase in phases},
        iou_scores={phase: [] for phase in phases}
    )

    for epoch in range(best['epoch']+1, cfg.epochs):

        start = time.strftime("%H:%M:%S")
        log(f"Starting epoch: {epoch} | phase: \'train\' | ⏰: {start}")
        result = run_nn(cfg.data.train, 'train', model, loader_train,
                        criterion, optimizer)

        state = {
            "epoch": epoch,
            "best_loss": best['loss'],
            "best_score": best['score'],
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        training_summary['epoch'] = epoch
        training_summary['losses']['train'].append(result['epoch_loss'])
        training_summary['dice_scores']['train'].append(result['dice_score'])
        training_summary['iou_scores']['train'].append(result['iou_score'])

        start = time.strftime("%H:%M:%S")
        log(f"Starting epoch: {epoch} | phase: \'valid\' | ⏰: {start}")
        with torch.no_grad():
            result = run_nn(cfg.data.valid, 'valid',
                            model, loader_valid, criterion)

        # val_loss = result['epoch_loss']
        # !!!!!!
        # val_loss = result['dice_score']

        training_summary['losses']['valid'].append(result['epoch_loss'])
        training_summary['dice_scores']['valid'].append(result['dice_score'])
        training_summary['iou_scores']['valid'].append(result['iou_score'])

        scheduler.step(result['epoch_loss'])

        # if result['epoch_loss'] <= best['loss']:
        if result['dice_score'] >= best['score']:
            log("******** New optimal found, saving state ********")
            log("")
            state["best_loss"] = best['loss'] = result['epoch_loss']
            state["best_score"] = best['score'] = result['dice_score']
            torch.save(state, os.path.join(cfg.workdir,
                       "model_{}_{}.pth".format(cfg.imgsize, cfg.fold)))

    with open(os.path.join(cfg.workdir,
              "training_summary_{}_{}.pkl".format(cfg.imgsize, cfg.fold)),
              'wb') as f:
        pickle.dump(training_summary, f)


def run_nn(cfg, mode, model, loader, criterion, optimizer=None):

    meter = Meter(cfg)

    if mode == 'train':
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    running_loss = 0.0
    total_batches = len(loader)
#         tk0 = tqdm(enumerate(loader), total=total_batches)

    for i, (inputs, targets) in enumerate(loader):

        inputs = inputs.cuda()    # inputs == images
        targets = targets.cuda()  # targets == masks
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # loss = loss / cfg.n_grad_acc

        if mode == "train":
            loss.backward()
            if (i + 1) % cfg.n_grad_acc == 0:
                optimizer.step()
                optimizer.zero_grad()

        with torch.no_grad():
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets.cpu(), outputs)
#             tk0.set_postfix(loss=(running_loss / ((itr + 1))))

    epoch_loss = running_loss / total_batches  # * cfg.n_grad_acc
    dice, iou = epoch_log(epoch_loss, meter)

    torch.cuda.empty_cache()  # <---- ?

    result = {
        'epoch_loss': epoch_loss,
        'dice_score': dice,
        'iou_score': iou,
    }

    return result


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard Interrupted')
