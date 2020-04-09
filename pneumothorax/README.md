# Pneumothorax Segmentation in Chest X-ray Images

This is the Capstone Project for Udacity Machine Learning Engineer Nanodegree based on 
[Pneumothorax Segmentation in Chest X-ray Images](
www.kaggle.com/c/siim-acr-pneumothorax-segmentation) hosted on Kaggle in 2019.
- [Project proposal](https://github.com/akuritsyn/udacity-ml-nanodegree/blob/master/pneumothorax/capstone-proposal.pdf)
- [Final project report](https://github.com/akuritsyn/udacity-ml-nanodegree/blob/master/pneumothorax/capstone-final-report.pdf)


## Requirements

- Python 3.7.5
- [Pytorch](https://pytorch.org/) 1.3.1
- [Segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) 0.1
- [Albumentations](https://github.com/albumentations-team/albumentations) 0.3.0


## Preparation

Please put `./input` directory in the root level and unzip the files downloaded from Kaggle there (
[512x512](https://www.kaggle.com/iafoss/siimacr-pneumothorax-segmentation-data-512),
[1024x1024](https://www.kaggle.com/iafoss/siimacr-pneumothorax-segmentation-data-1024),
[Annotations](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/data)
). All other directories such as `./cache`, `./model` will be created automatically when needed.


## Preprocessing

Please make sure you run the script from parent directory of `./bin`.

~~~
$ sh ./bin/preprocess.sh
~~~

[preprocess.sh](https://github.com/akuritsyn/udacity-ml-nanodegree/blob/master/pneumothorax/bin/preprocess.sh) is used to make 5 folds for training data using StratifiedKFold.



## Training

~~~
$ sh ./bin/train00[X].sh
~~~

- [train006.sh](https://github.com/akuritsyn/udacity-ml-nanodegree/blob/master/pneumothorax/bin/train006.sh) performs traing on 512x512 images using Focal+Dice loss. 
- [train007.sh](https://github.com/akuritsyn/udacity-ml-nanodegree/blob/master/pneumothorax/bin/train007.sh) upscales previous model to 1024x1024 images using Focal+Dice loss. 
- [train008.sh](https://github.com/akuritsyn/udacity-ml-nanodegree/blob/master/pneumothorax/bin/train008.sh) upscales previous model to 1024x1024 images using BCE+Dice loss.

To train all the folds [0...4], change fold numbers inside .sh files and names of the pre-trained models in the corresponing config files located in `./conf`.

All 3 steps for one fold probably take ~10.5 hours to train with a single 1080ti GPU.


## Inference

~~~
$ sh ./bin/test00[X].sh
~~~
[test008.sh](https://github.com/akuritsyn/udacity-ml-nanodegree/blob/master/pneumothorax/bin/test008.sh) does the predictions for the final model.

You can [upload the result to Kaggle](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/submissions) for scoring either manually or do it through a Kaggle API (registered account is required):
~~~
$  kaggle competitions submit -c siim-acr-pneumothorax-segmentation -f submission.scv -m "my submission"
~~~