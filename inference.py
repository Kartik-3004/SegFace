import os
import pdb
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict

def get_confusion_matrix(gt_label, pred_label, num_classes):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param num_classes: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * num_classes + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i_label in range(num_classes):
        for i_pred_label in range(num_classes):
            cur_index = i_label * num_classes + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


def fast_histogram(a, b, na, nb):
    '''
    fast histogram calculation
    ---
    * a, b: non negative label ids, a.shape == b.shape, a in [0, ... na-1], b in [0, ..., nb-1]
    '''
    assert a.shape == b.shape
    assert np.all((a >= 0) & (a < na) & (b >= 0) & (b < nb))
    # k = (a >= 0) & (a < na) & (b >= 0) & (b < nb)
    hist = np.bincount(
        nb * a.reshape([-1]).astype(int) + b.reshape([-1]).astype(int),
        minlength=na * nb).reshape(na, nb)
    assert np.sum(hist) == a.size
    return hist

def _merge(*list_pairs):
    a = []
    b = []
    for al, bl in list_pairs:
        a += al
        b += bl
    return a, b 

def validation(cfg, model, val_dataloaders, val_names, log_root):
    if val_names[0] == "LaPa":
        LABELS = ['background', 'face_lr_rr', 'lb', 'rb', 'le', 're', 'nose', 'ul', 'im', 'll', 'hair']
        gt_label_names = ['background', 'face_lr_rr', 'lb', 'rb', 'le', 're', 'nose', 'ul', 'im', 'll', 'hair']
        pred_label_names = ['background', 'face_lr_rr', 'lb', 'rb', 'le', 're', 'nose', 'ul', 'im', 'll', 'hair']
        num_classes = 11
    elif val_names[0] == "CelebAMaskHQ":
        LABELS = ['background', 'neck', 'skin', 'cloth', 'l_ear', 'r_ear', 'l_brow', 'r_brow','l_eye', 'r_eye', 'nose', 'mouth', 'l_lip', 'u_lip', 'hair','eye_g', 'hat', 'ear_r', 'neck_l']
        gt_label_names = ['background', 'neck', 'skin', 'cloth', 'l_ear', 'r_ear', 'l_brow', 'r_brow','l_eye', 'r_eye', 'nose', 'mouth', 'l_lip', 'u_lip', 'hair','eye_g', 'hat', 'ear_r', 'neck_l']
        pred_label_names = ['background', 'neck', 'skin', 'cloth', 'l_ear', 'r_ear', 'l_brow', 'r_brow','l_eye', 'r_eye', 'nose', 'mouth', 'l_lip', 'u_lip', 'hair','eye_g', 'hat', 'ear_r', 'neck_l']
        num_classes = 19
    elif val_names[0] == "HELEN":
        LABELS = ['bg', 'face', 'lb', 'rb', 'le', 're', 'nose', 'ulip', 'imouth', 'llip', 'hair']
        gt_label_names = ['bg', 'face', 'lb', 'rb', 'le', 're', 'nose', 'ulip', 'imouth', 'llip', 'hair']
        pred_label_names = ['bg', 'face', 'lb', 'rb', 'le', 're', 'nose', 'ulip', 'imouth', 'llip', 'hair']
        num_classes = 11
    confusion_matrix = np.zeros((num_classes, num_classes))

    hists = []
    loader_names = val_names
    model.eval()
    for itr, val_dataloader in enumerate(val_dataloaders):
        with torch.no_grad():
            val_pbar = tqdm(val_dataloader, leave=True)
            for batch in val_pbar:
                images, labels, datasets = batch["image"], batch["label"], batch["dataset"]
                images = images.cuda()
                for k in labels.keys():
                    labels[k] = labels[k].cuda()
                datasets = datasets.cuda()
                seg_output = model(images, labels, datasets)
                mask = F.interpolate(seg_output, size=(cfg.input_resolution,cfg.input_resolution), mode='bilinear', align_corners=False)
                mask = mask.softmax(dim=1)
                preds = torch.argmax(mask, dim=1)
                preds = preds.cpu().numpy()
                gt = labels['segmentation'].cpu().numpy()
                for pred, gt in zip(preds, gts):
                    gt = np.asarray(gt, dtype=np.int32)
                    pred = np.asarray(pred, dtype=np.int32)
                    ignore_index = gt != 255
                    gt = gt[ignore_index]
                    pred = pred[ignore_index]

                    hist = fast_histogram(gt, pred, len(gt_label_names), len(pred_label_names))
                    hists.append(hist)

                    confusion_matrix += get_confusion_matrix(gt, pred, num_classes)

    hist_sum = np.sum(np.stack(hists, axis=0), axis=0)    
    eval_names = dict()
    for label_name in gt_label_names:
        gt_ind = gt_label_names.index(label_name)
        pred_ind = pred_label_names.index(label_name)
        eval_names[label_name] = ([gt_ind], [pred_ind])
    if val_names[0] == "HELEN":
        if 'le' in eval_names and 're' in eval_names:
            eval_names['eyes'] = _merge(eval_names['le'], eval_names['re'])
        if 'lb' in eval_names and 'rb' in eval_names:
            eval_names['brows'] = _merge(eval_names['lb'], eval_names['rb'])
        if 'ulip' in eval_names and 'imouth' in eval_names and 'llip' in eval_names:
            eval_names['mouth'] = _merge(
                eval_names['ulip'], eval_names['imouth'], eval_names['llip'])
        if 'eyes' in eval_names and 'brows' in eval_names and 'nose' in eval_names and 'mouth' in eval_names:
            eval_names['overall'] = _merge(
                eval_names['eyes'], eval_names['brows'], eval_names['nose'], eval_names['mouth'])


    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()
    log_root.info('Pixel accuracy: %f' % pixel_accuracy)
    log_root.info('Mean accuracy: %f' % mean_accuracy)
    mIoU_value = []
    f1_value = []
    mf1_value = []

    for i, (label, iou) in enumerate(zip(LABELS, IoU_array)):
        mIoU_value.append((label, iou))

    mIoU_value.append(('Pixel accuracy', pixel_accuracy))
    mIoU_value.append(('Mean accuracy', mean_accuracy))
    mIoU_value.append(('Mean IU', mean_IoU))
    mIoU_value = OrderedDict(mIoU_value)
    for key in mIoU_value.keys():
        log_root.info(f"{key} : {mIoU_value[key]}")
    log_root.info("---"*15)

    for eval_name, (gt_inds, pred_inds) in eval_names.items():
        A = hist_sum[gt_inds, :].sum()
        B = hist_sum[:, pred_inds].sum()
        intersected = hist_sum[gt_inds, :][:, pred_inds].sum()
        f1 = 2 * intersected / (A + B)

        if eval_name in gt_label_names[1:]:
            mf1_value.append(f1)
        f1_value.append((eval_name, f1))

    f1_value.append(('Mean_F1', np.array(mf1_value).mean()))
    f1_value = OrderedDict(f1_value)
    log_root.info('Mean_F1: %f \n' % f1_value['Mean_F1'])
    for key in f1_value.keys():
        log_root.info(f"{key} : {f1_value[key]}")
    log_root.info(f'Validation Done')
    return None      
 
