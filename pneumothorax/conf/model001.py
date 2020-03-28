workdir = './model/model001'
seed = 69

n_fold = 5
epochs = 50
resume_from = './model/model001/model_512_1.pth'

batch_size = 4
#acc_steps = 4
num_workers = 4
imgsize = 1024

model = dict(
    name='unet_resnet34',
    pretrained='imagenet',
)

optim = dict(
    name='Adam',
    params=dict(
        lr=5e-4,
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

normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],}


# crop = dict(name='RandomResizedCrop', params=dict(height=imgsize[0], width=imgsize[1], scale=(0.7,1.0), p=1.0))
resize = dict(name='Resize', params=dict(height=imgsize, width=imgsize))
hflip = dict(name='HorizontalFlip', params=dict(p=1.))
# vflip = dict(name='VerticalFlip', params=dict(p=0.5,))
# contrast = dict(name='RandomBrightnessContrast', params=dict(brightness_limit=0.08, contrast_limit=0.08, p=0.5))
totensor = dict(name='ToTensor', params=dict(normalize=normalize))
# rotate = dict(name='Rotate', params=dict(limit=30, border_mode=0), p=0.7)

data = dict(
    train=dict(
        #dataset_type='CustomDataset',
        #annotations='./cache/train_folds.pkl',
        imgdir='./input/1024-s2/train',
        train_rle_path = './input/stage_2_train.csv',
        imgsize=imgsize,
        n_grad_acc=4,
        loader=dict(
            shuffle=True,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        #transforms=[crop, hflip, rotate, contrast, totensor],
        #dataset_policy='all',
        #window_policy=window_policy,
    ),
    valid = dict(
        #dataset_type='CustomDataset',
        #annotations='./cache/train_folds.pkl',
        imgdir='./input/1024-s2/train',
        imgsize=imgsize,
        loader=dict(
            shuffle=False,
            batch_size=2,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        #transforms=[crop, hflip, rotate, contrast, totensor],
        #dataset_policy='all',
        #window_policy=window_policy,
    ),
    test = dict(
        #dataset_type='CustomDataset',
        #annotations='./cache/test.pkl',
        normalize=normalize,
        imgdir='./input/1024-s2/test',
 #       sample_submission_file = './input/stage_2_sample_submission.csv',
        sample_submission_file = './predict/submission_pytorch_5fold_ave_Wflip_0p55th.csv',
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
        prob_threshold = 0.55,
        min_object_size = 3500, #pixels
        output_file_probabilty_name = 'pixel_probabilities_1024.pkl',
        submission_file_name = 'submission_pytorch_5fold_ave_Wflip_0p55th.csv',
        #dataset_policy='all',
        #window_policy=window_policy,
    ),
)
