from torch.utils.data import DataLoader, Dataset
import pandas as pd

import albumentations as albu
from albumentations.torch import ToTensor
# from torchvision.transforms import ToTensor
from sklearn.model_selection import StratifiedKFold
import cv2
import os
import numpy as np
from .rle_functions import run_length_decode
from .utils.logger import log


class SIIMDataset(Dataset):
    def __init__(self, df, data_folder, size, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.size = size
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, size, mean, std)
        self.gb = self.df.groupby('ImageId')
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


def get_transforms(phase, size, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                albu.HorizontalFlip(),
                albu.OneOf([
                    albu.RandomContrast(),
                    albu.RandomGamma(),
                    albu.RandomBrightness(),
                    ], p=0.3),
                albu.OneOf([
                    albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    albu.GridDistortion(),
                    albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                    ], p=0.3),
                albu.ShiftScaleRotate(),
                # GaussNoise(),
            ]
        )
    list_transforms.extend(
        [
            albu.Normalize(mean=mean, std=std, p=1),
            albu.Resize(size, size),
            ToTensor(),
        ]
    )

    list_trfms = albu.Compose(list_transforms)
    return list_trfms


def provider(
    cfg,
    phase,
    batch_size=8,
):
    # Think about saving this part into make_folds.py and just loading it here from cache
    df = pd.read_csv(cfg.data.train.train_rle_path)
#     df = df.drop_duplicates('ImageId')
    df_with_mask = df[df["EncodedPixels"] != "-1"]
    df_with_mask['has_mask'] = 1
    df_without_mask = df[df["EncodedPixels"] == "-1"]
    df_without_mask['has_mask'] = 0
    df_without_mask_sampled = df_without_mask.sample(len(df_with_mask.drop_duplicates('ImageId')))
    df = pd.concat([df_with_mask, df_without_mask_sampled])

    kfold = StratifiedKFold(cfg.n_fold, shuffle=True, random_state=cfg.seed)
    train_idx, val_idx = list(kfold.split(
        df["ImageId"], df["has_mask"]))[cfg.fold]
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    df = train_df if phase == "train" else val_df
    # NOTE: n_fold=5 -> train/val : 80%/20%

    if cfg.debug:
        df = df.head(cfg.debug)
        log('Debug mode: loading first %d records' % df.shape[0])

    image_dataset = SIIMDataset(df, cfg.data.train.imgdir, cfg.imgsize, cfg.normalize.mean, cfg.normalize.std, phase)

    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    log(f'{phase} data: loaded {len(dataloader.dataset)} records')
    return dataloader
