import albumentations as albu

workdir = './model/model004'
seed = 69

n_fold = 5
epochs = 50
sample_classes = True
resume_from = None  # './model/model001/model_1024_0.pth'
retrain_from = None  # './model/model002/model_1024_0.pth'

train_rle_path = './input/stage_2_train.csv'
#train_imgdir = './input/1024-s2/train'
train_imgdir = './input/512-s2/train'

batch_size = 16
n_grad_acc = 2
num_workers = 4
imgsize = 512

model = dict(
    name='unet_resnet34',
    pretrained='imagenet',
)

optim = dict(
    name='Adam',
    params=dict(
        lr=5e-4,  # lr=5e-4
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

hflip = dict(name='HorizontalFlip', args=[], params=dict())

oneof_contrast = dict(name='OneOf', args=[[
    albu.RandomContrast(),
    albu.RandomGamma(),
    albu.RandomBrightness()]], params=dict(p=0.3))

oneof_transform = dict(name='OneOf', args=[[
    albu.ElasticTransform(alpha=120, sigma=120*0.05, alpha_affine=120*0.03),
    albu.GridDistortion(),
    albu.OpticalDistortion(distort_limit=2, shift_limit=0.5)]], params=dict(p=0.3))

shiftscalerotate = dict(name='ShiftScaleRotate', args=[], params=dict())

# normalize_old = dict(name='Normalize', args=[], params=dict(mean=normalize['mean'], std=normalize['std'], p=1))
# totensor_old = dict(name='ToTensor', args=[], params=dict())

resize = dict(name='Resize', args=[], params=dict(height=imgsize, width=imgsize))
totensor = dict(name='ToTensor', args=[], params=dict(normalize=normalize))

hflip1 = dict(name='HorizontalFlip', args=[], params=dict(p=1.))


data = dict(
    train=dict(
        phase='train',
        imgdir=train_imgdir,
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
        transforms=[hflip, oneof_contrast, oneof_transform, shiftscalerotate, resize, totensor] 
    ),
    valid=dict(
        phase='valid',
        imgdir=train_imgdir,
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
        min_object_size=None,  # min_object_size,
        transforms=[resize, totensor],
        # transforms=[normalize_old, resize, totensor_old],
    ),
    test=dict(
        imgdir='./input/512-s2/test',
        # sample_submission_file = './input/stage_2_sample_submission.csv',
        sample_submission_file='./predict/submission_pytorch_5fold_ave_Wflip_0p55th.csv',
        trained_models=workdir+'/'+'model_512_*.pth',
        imgsize=imgsize,
        loader=dict(
            shuffle=False,
            batch_size=1,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        transforms=[resize, totensor],
        transforms_and_hflip=[hflip1, resize, totensor],
        prob_threshold=0.55,
        min_object_size=3500,
        output_file_probabilty_name='pixel_probabilities_512.pkl',
        submission_file_name='submission_pytorch_5fold_ave_Wflip_0p55th_FineTunedOnAllImages.csv',
    ),
)
