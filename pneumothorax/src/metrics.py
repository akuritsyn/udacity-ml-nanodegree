import numpy as np
import torch
import cv2
from .utils.logger import log
from .predictor import post_process


def predict_masks(X, prob_threshold=0.5):
    X_p = np.copy(X)
    preds = (X_p >= prob_threshold).astype('uint8')
    return preds


def metric(probability, truth, imgsize, prob_threshold, min_object_size):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    with torch.no_grad():

        # probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)  # torch.Size([4, 1048576])
        # assert(probability.shape == truth.shape)
        t = (truth > 0.5).float()
        # print(probability.shape, truth.shape)

        if min_object_size:
            probability = probability.numpy()[:, 0, :, :]  # torch.Size([4, 1, 1024, 1024]) --> [4, 1024, 1024]
            for i, prob in enumerate(probability):
                predict, num_predict = post_process(prob, prob_threshold, min_object_size)
                if num_predict == 0:
                    probability[i, :, :] = 0
                else:
                    probability[i, :, :] = predict
            p = torch.from_numpy(probability)
            p = p.view(batch_size, -1).float()  # torch.Size([4, 1048576])
        else:
            probability = probability.view(batch_size, -1)
            p = (probability > prob_threshold).float()

        EPS = 1e-6
        intersection = torch.sum(p * t, dim=1)
        union = torch.sum(p, dim=1) + torch.sum(t, dim=1) + EPS
        dice = (2*(intersection + EPS) / union).mean()
        if dice > 1:
            dice = 1

    return dice


class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, cfg):
        self.imgsize = cfg.imgsize
        self.prob_threshold = cfg.prob_threshold
        self.min_object_size = cfg.min_object_size
        self.base_dice_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice = metric(probs, targets, self.imgsize, self.prob_threshold, self.min_object_size)
        self.base_dice_scores.append(dice)
        preds = predict_masks(probs, self.prob_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        iou = np.nanmean(self.iou_scores)
        return dice, iou


def epoch_log(epoch_loss, meter):
    '''logging the metrics at the end of an epoch'''
    dice, iou = meter.get_metrics()
    log("Loss: %0.4f | dice: %0.4f | IoU: %0.4f" % (epoch_loss, dice, iou))
    return dice, iou


def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs)  # copy is imp
    labels = np.array(labels)  # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou
