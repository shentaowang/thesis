from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import argparse
import torch
import json
import time
import os
import cv2

from sklearn import metrics
from scipy import interpolate
import numpy as np
from torchvision.transforms import transforms as T
from models.model import create_model, load_model
from datasets.dataset.jde import DetDataset, collate_fn
from utils.utils import xywh2xyxy, ap_per_class, bbox_iou
from fcos_opts import opts
from utils.post_process import ctdet_post_process

from utils.image import transform_preds

from mmdet.core import multiclass_nms, distance2bbox


def post_process(opt, dets, meta):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], opt.num_classes)
    for j in range(1, opt.num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
    return dets[0]


def merge_outputs(opt, detections):
    results = {}
    for j in range(1, opt.num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

    scores = np.hstack(
        [results[j][:, 4] for j in range(1, opt.num_classes + 1)])
    if len(scores) > 128:
        kth = len(scores) - 128
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, opt.num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results


def fcos_decode(opt, scores, bbox_pred, centerness, nms_pre=5000, score_thr=0.15, max_per_img=500):
    h, w = opt.output_h, opt.output_w
    scores = scores[0].permute(1, 2, 0).reshape(-1, opt.num_classes)
    padding = scores.new_zeros(scores.shape[0], 1)
    scores = torch.cat([scores, padding], dim=1)
    centerness = centerness[0].permute(1, 2, 0).reshape(-1)
    bbox_pred = bbox_pred[0].permute(1, 2, 0).reshape(-1, 4)
    x_range = torch.arange(0, w * opt.down_ratio, opt.down_ratio, dtype=torch.float)
    y_range = torch.arange(0, h * opt.down_ratio, opt.down_ratio, dtype=torch.float)
    y, x = torch.meshgrid(y_range, x_range)
    points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1) + opt.down_ratio // 2
    points = points.detach().cuda(bbox_pred.device)
    if nms_pre > 0 and scores.shape[0] > nms_pre:
        max_scores, _ = (scores * centerness[:, None]).max(dim=1)
        # max_scores, _ = (scores).max(dim=1)
        _, topk_inds = max_scores.topk(nms_pre)
        points = points[topk_inds, :]
        bbox_pred = bbox_pred[topk_inds, :]
        scores = scores[topk_inds, :]
        centerness = centerness[topk_inds]
    bboxes = distance2bbox(points, bbox_pred)
    det_bboxes, det_labels = multiclass_nms(
        bboxes,
        scores,
        score_thr,
        dict(type='nms', iou_thr=0.4, class_agnostic=False),
        max_per_img,
        centerness
    )
    results = {}
    for i in range(0, opt.num_classes):
        inds = (det_labels == i)
        results[i+1] = det_bboxes[inds]

    return results


def test_det(
        opt,
        batch_size=12,
        img_size=(1088, 608),
        iou_thres=0.5,
        print_interval=40):
    data_cfg = opt.data_cfg
    f = open(data_cfg)
    data_cfg_dict = json.load(f)
    f.close()
    nC = 10
    test_path = data_cfg_dict['test']
    dataset_root = data_cfg_dict['root']
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    #model = torch.nn.DataParallel(model)
    model = model.to(opt.device)
    model.eval()

    # Get dataloader
    transforms = T.Compose([T.ToTensor()])
    dataset = DetDataset(dataset_root, test_path, img_size, augment=False, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=8, drop_last=False, collate_fn=collate_fn)
    mean_mAP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    outputs, mAPs, mR, mP, TP, confidence, pred_class, target_class, jdict = \
        [], [], [], [], [], [], [], [], []
    AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)
    for batch_i, (imgs, targets, paths, shapes, targets_len) in enumerate(dataloader):
        t = time.time()
        #seen += batch_size

        output = model(imgs.cuda())[-1]
        origin_shape = shapes[0]
        width = origin_shape[1]
        height = origin_shape[0]
        inp_height = img_size[1]
        inp_width = img_size[0]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height,
                'out_width': inp_width}
        hm = output['hm'].sigmoid_()
        bbox = output['bbox']
        centerness = output['centerness'].sigmoid_()
        opt.K = 200
        # detections, inds = mot_decode(hm, wh, reg=reg, cat_spec_wh=opt.cat_spec_wh, K=opt.K)
        # Compute average precision for each sample
        targets = [targets[i][:int(l)] for i, l in enumerate(targets_len)]
        for si, labels in enumerate(targets):
            seen += 1
            dets = fcos_decode(opt, hm[si:si + 1], bbox[si:si + 1], centerness[si:si + 1], score_thr=0.5)
            # path = paths[si]
            # img0 = cv2.imread(path)
            pred_class = []
            for i in range(10):
                pred_class += [i] * len(dets[i + 1])
            dets = torch.cat(
                [dets[1], dets[2], dets[3], dets[4], dets[5], dets[6], dets[7], dets[8], dets[9], dets[10]])
            assert len(pred_class) == len(dets)
            dets[:, 0] = dets[:, 0].clamp(min=0, max=opt.input_w - 1)
            dets[:, 1] = dets[:, 1].clamp(min=0, max=opt.input_h - 1)
            dets[:, 2] = dets[:, 2].clamp(min=0, max=opt.input_w - 1)
            dets[:, 3] = dets[:, 3].clamp(min=0, max=opt.input_h - 1)

            dets = dets.detach().cpu().numpy()

            dets[:, :2] = transform_preds(
                dets[:, 0:2], meta['c'], meta['s'], (meta['out_width'], meta['out_height']))
            dets[:, 2:4] = transform_preds(
                dets[:, 2:4], meta['c'], meta['s'], (meta['out_width'], meta['out_height']))

            scores = dets[:, 4]
            dets = dets[scores.argsort()[-opt.K*10:][::-1]]

            # remain_inds = dets[:, 4] > opt.det_thres
            # dets = dets[remain_inds]
            if dets is None:
                # If there are labels but no detections mark as zero AP
                if labels.size(0) != 0:
                    mAPs.append(0), mR.append(0), mP.append(0)
                continue

            # If no labels add number of detections as incorrect
            correct = []
            if labels.size(0) == 0:
                # correct.extend([0 for _ in range(len(detections))])
                mAPs.append(0), mR.append(0), mP.append(0)
                continue
            else:
                target_cls = labels[:, 0]

                # Extract target boxes as (x1, y1, x2, y2)
                target_boxes = xywh2xyxy(labels[:, 2:6])
                target_boxes[:, 0] *= width
                target_boxes[:, 2] *= width
                target_boxes[:, 1] *= height
                target_boxes[:, 3] *= height

                # path = paths[si]
                # img0 = cv2.imread(path)
                # img1 = cv2.imread(path)
                # for t in range(len(target_boxes)):
                #     x1 = target_boxes[t, 0]
                #     y1 = target_boxes[t, 1]
                #     x2 = target_boxes[t, 2]
                #     y2 = target_boxes[t, 3]
                #     cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 4)
                # cv2.imwrite('gt.jpg', img0)
                # for t in range(len(dets)):
                #     x1 = dets[t, 0]
                #     y1 = dets[t, 1]
                #     x2 = dets[t, 2]
                #     y2 = dets[t, 3]
                #     cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 4)
                # cv2.imwrite('pred.jpg', img1)
                # abc = ace
                detected = []
                assert len(dets) == len(pred_class)
                for idx, (*pred_bbox, conf) in enumerate(dets):
                    obj_pred = pred_class[idx]
                    # obj_pred = pred_class[idx]
                    pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                    # Compute iou with target boxes
                    iou = bbox_iou(pred_bbox, target_boxes, x1y1x2y2=True)[0]
                    # Extract index of largest overlap
                    best_i = np.argmax(iou)
                    # If overlap exceeds threshold and classification is correct mark as correct
                    if iou[best_i] > iou_thres and obj_pred == labels[best_i, 0] and best_i not in detected:
                        correct.append(1)
                        detected.append(best_i)
                    else:
                        correct.append(0)

            # Compute Average Precision (AP) per class
            # print(pred_class)
            # print(target_cls)
            AP, AP_class, R, P = ap_per_class(tp=correct,
                                              conf=dets[:, 4],
                                              pred_cls=pred_class,
                                              target_cls=target_cls)

            # Accumulate AP per class
            AP_accum_count += np.bincount(AP_class, minlength=nC)
            AP_accum += np.bincount(AP_class, minlength=nC, weights=AP)

            # Compute mean AP across all classes in this image, and append to image list
            mAPs.append(AP.mean())
            mR.append(R.mean())
            mP.append(P.mean())

            # Means of all images
            mean_mAP = np.sum(mAPs) / (AP_accum_count + 1E-16)
            mean_R = np.sum(mR) / (AP_accum_count + 1E-16)
            mean_P = np.sum(mP) / (AP_accum_count + 1E-16)
        if batch_i % print_interval == 0:
            # Print image mAP and running mean mAP
            print(('%11s%11s' + '%11.3g' * 4 + 's') %
                  (seen, dataloader.dataset.nF, np.mean(mean_P), np.mean(mean_R), np.mean(mean_mAP), time.time() - t))
    # Print mAP per class
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    for i in range(10):
        print('AP: %-.4f\n\n' % (AP_accum[i] / (AP_accum_count[i] + 1E-16)))

    # Return mAP
    return mean_mAP, mean_R, mean_P


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()
    opt.load_model = '../exp/mot/0829-hrnet-detction/model_last.pth'
    with torch.no_grad():
        map = test_det(opt, batch_size=12)
