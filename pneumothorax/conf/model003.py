workdir = './model/model003'
seed = 69

n_fold = 5
epochs = 20
sample_classes = False
resume_from = None  # './model/model001/model_1024_0.pth'
retrain_from = './predict/model_1024_1.pth'

batch_size = 4
n_grad_acc = 4
num_workers = 4
imgsize = 1024

model = dict(
    name='unet_resnet34',
    pretrained='imagenet',
)

optim = dict(
    name='Adam',
    params=dict(
        lr=1e-5,  # lr=5e-4
    ),
)

loss = dict(
    name='MixedLoss',
    params=dict(
        alpha=10,
        gamma=2,
    ),
)

scheduler = dict(
    name='ReduceLROnPlateau',
    params=dict(
        mode="min",
        patience=3,
        verbose=True,
    ),
)

prob_threshold = 0.5
min_object_size = 3500  # pixels 

normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


# crop = dict(name='RandomResizedCrop', params=dict(height=imgsize[0], width=imgsize[1], scale=(0.7,1.0), p=1.0))
resize = dict(name='Resize', params=dict(height=imgsize, width=imgsize))
hflip = dict(name='HorizontalFlip', params=dict(p=1.))
# vflip = dict(name='VerticalFlip', params=dict(p=0.5,))
# contrast = dict(name='RandomBrightnessContrast', params=dict(brightness_limit=0.08, contrast_limit=0.08, p=0.5))
totensor = dict(name='ToTensor', params=dict(normalize=normalize))
# rotate = dict(name='Rotate', params=dict(limit=30, border_mode=0), p=0.7)

data = dict(
    train=dict(
        # dataset_type='CustomDataset',
        # annotations='./cache/train_folds.pkl',
        imgdir='./input/1024-s2/train',
        train_rle_path='./input/stage_2_train.csv',
        imgsize=imgsize,
        n_grad_acc=n_grad_acc,
        loader=dict(
            shuffle=True,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        prob_threshold=prob_threshold,
        min_object_size=None,
        # transforms=[crop, hflip, rotate, contrast, totensor],
    ),
    valid=dict(
        # dataset_type='CustomDataset',
        # annotations='./cache/train_folds.pkl',
        imgdir='./input/1024-s2/train',
        imgsize=imgsize,
        n_grad_acc=n_grad_acc,
        loader=dict(
            shuffle=False,
            batch_size=2,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        prob_threshold=prob_threshold,
        min_object_size=None  # min_object_size,
        # transforms=[crop, hflip, rotate, contrast, totensor],
    ),
    test=dict(
        # dataset_type='CustomDataset',
        # annotations='./cache/test.pkl',
        normalize=normalize,
        imgdir='./input/1024-s2/test',
        # sample_submission_file = './input/stage_2_sample_submission.csv',
        sample_submission_file='./predict/submission_pytorch_5fold_ave_Wflip_0p55th.csv',
        trained_models=workdir+'/'+'model_1024_*.pth',
        imgsize=imgsize,
        loader=dict(
            shuffle=False,
            batch_size=1,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        transforms=[resize, totensor],
        transforms_and_hflip=[hflip, resize, totensor],
        prob_threshold=0.55,
        min_object_size=3500,
        output_file_probabilty_name='pixel_probabilities_1024.pkl',
        submission_file_name='submission_pytorch_5fold_ave_Wflip_0p55th.csv',
    ),
)
