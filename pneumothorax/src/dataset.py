from torch.utils.data import DataLoader, Dataset
import pandas as pd
import pickle

import albumentations as albu
from albumentations.torch import ToTensor

# from sklearn.model_selection import StratifiedKFold
import cv2
import os
import numpy as np

from .rle_functions import run_length_decode
from .utils.logger import log
from .factory import get_transforms


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.root = cfg.imgdir
        self.transforms = get_transforms(cfg.transforms)
        self.gb = df.groupby('ImageId')
        self.fnames = list(self.gb.groups.keys())

    def __getitem__(self, idx):
        image_id = self.fnames[idx]
        df = self.gb.get_group(image_id)
        annotations = df['EncodedPixels'].tolist()
        image_path = os.path.join(self.root, image_id + ".png")
        image = cv2.imread(image_path)
        mask = np.zeros([1024, 1024])
        if annotations[0] != '-1':
            for rle in annotations:
                mask += run_length_decode(rle)
        mask = (mask >= 1).astype('float32')  # for overlap cases
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        return image, mask

    def __len__(self):
        return len(self.fnames)


def provider(cfg, phase):

    df = pd.read_csv(cfg.train_rle_path)

    with open(cfg.train_folds, 'rb') as f:
        folds_list = pickle.load(f)

    train_idx, val_idx = folds_list[cfg.fold]
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    df = train_df if phase == "train" else val_df
    # n_fold=5 -> train/val : 80%/20%

    if phase == 'train':

        if cfg.sample_classes:
            df_with_mask = df[df["EncodedPixels"] != "-1"]
            # df_with_mask['has_mask'] = 1
            df_without_mask = df[df["EncodedPixels"] == "-1"]
            # df_without_mask['has_mask'] = 0
            df_without_mask_sampled = df_without_mask.sample(len(df_with_mask.drop_duplicates('ImageId')), random_state=cfg.seed)

            if cfg.debug:
                df = pd.concat([df_with_mask.head(cfg.debug//2),
                                df_without_mask_sampled.head(cfg.debug//2)])
                log('Debug mode: reding first %d records with class sampler' % df.shape[0])
            else:
                log(f'Using class sampler: with mask - {df_with_mask.shape[0]}, without mask - {df_without_mask_sampled.shape[0]}')
                df = pd.concat([df_with_mask, df_without_mask_sampled])

    elif phase == 'valid':

        if cfg.debug:
            df = df.head(cfg.debug)
            log('Debug mode: reading first %d records' % df.shape[0])

    if phase == 'train':
        cfg_to_pass = cfg.data.train
    elif phase == 'valid':
        cfg_to_pass = cfg.data.valid

    dataset = TrainDataset(cfg_to_pass, df)
    loader = DataLoader(dataset, **cfg_to_pass.loader)

    log(f'{phase} data: loaded {len(loader.dataset)} records')
    return loader
