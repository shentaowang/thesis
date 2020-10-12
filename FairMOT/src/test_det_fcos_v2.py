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


def fcos_decode(opt, scores, bbox_pred, centerness, nms_pre=5000, score_thr=0.15, max_per_img=1000):
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
        max_per_img
    )
    print("before nms cnt: {}, after nms cnt: {}".format(len(bboxes), len(det_bboxes)))
    results = {}
    for i in range(0, opt.num_classes):
        inds = (det_labels == i)
        results[i+1] = det_bboxes[inds]

    return results


def test_det(
        opt,
        detections_dir,
        groundtruths_dir,
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
    for batch_i, (imgs, targets, paths, shapes, targets_len) in enumerate(dataloader):
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
        # detections, inds = mot_decode(hm, wh, reg=reg, cat_spec_wh=opt.cat_spec_wh, K=opt.K)
        # Compute average precision for each sample
        targets = [targets[i][:int(l)] for i, l in enumerate(targets_len)]
        for si, labels in enumerate(targets):
            file_name = paths[si].split('/')[-1]
            file_name = file_name.replace('jpg', 'txt')
            dets = fcos_decode(opt, hm[si:si + 1], bbox[si:si + 1], centerness[si:si + 1], score_thr=opt.det_thres)
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

            # write the detect result
            with open(os.path.join(detections_dir, file_name), 'w') as fin:
                for i in range(len(pred_class)):
                    fin.write("{} {:.4f} {:.0f} {:.0f} {:.0f} {:.0f}\n".format(pred_class[i], dets[i][4], dets[i][0],
                                                                               dets[i][1], dets[i][2], dets[i][3]))

            with open(os.path.join(groundtruths_dir, file_name), 'w') as fin:
                for i in range(len(labels)):
                    target_boxes = xywh2xyxy(labels[:, 2:6])
                    target_boxes[:, 0] *= width
                    target_boxes[:, 2] *= width
                    target_boxes[:, 1] *= height
                    target_boxes[:, 3] *= height
                    fin.write("{:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n".format(labels[i][0], target_boxes[i][0],
                                                                            target_boxes[i][1], target_boxes[i][2],
                                                                            target_boxes[i][3]))
            # Extract target boxes as (x1, y1, x2, y2)
            target_boxes = xywh2xyxy(labels[:, 2:6])
            target_boxes[:, 0] *= width
            target_boxes[:, 2] *= width
            target_boxes[:, 1] *= height
            target_boxes[:, 3] *= height

            path = paths[si]
            img0 = cv2.imread(path)
            img1 = cv2.imread(path)
            for t in range(len(target_boxes)):
                x1 = target_boxes[t, 0]
                y1 = target_boxes[t, 1]
                x2 = target_boxes[t, 2]
                y2 = target_boxes[t, 3]
                cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.imwrite('../gt_images/{}'.format(paths[si].split('/')[-1]), img0)
            for t in range(len(dets)):
                x1 = dets[t, 0]
                y1 = dets[t, 1]
                x2 = dets[t, 2]
                y2 = dets[t, 3]
                cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.imwrite('../pred_images/{}'.format(paths[si].split('/')[-1]), img1)
            # abc = ace


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    opt.gpus = [0]
    exp_path = '../exp/mot/0831-hrnet-detction-2/'
    opt.load_model = os.path.join(exp_path, 'model_last.pth')
    opt.det_thres = 0.05
    detections_dir = os.path.join(exp_path, 'detections005')
    if not os.path.exists(detections_dir):
        os.makedirs(detections_dir)
    groundtruths_dir = os.path.join(exp_path, 'groundtruths005')
    if not os.path.exists(groundtruths_dir):
        os.makedirs(groundtruths_dir)
    if not os.path.exists(detections_dir):
        os.makedirs(detections_dir)
    if not os.path.exists(groundtruths_dir):
        os.makedirs(groundtruths_dir)
    with torch.no_grad():
        test_det(opt, batch_size=1, detections_dir=detections_dir, groundtruths_dir=groundtruths_dir)
