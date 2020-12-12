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
import matplotlib.pyplot as plt
from torchvision.transforms import transforms as T
from models.model import create_model, load_model
import datasets.dataset.jde as datasets
from utils.utils import xywh2xyxy, ap_per_class, bbox_iou
from fcos_opts import opts
from utils.post_process import ctdet_post_process
from tracking_utils.timer import Timer

from utils.image import transform_preds

from mmdet.core import multiclass_nms, distance2bbox
from lib.utils.nms_wrapper import nms_detections


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


def fcos_decode(opt, scores, bbox_pred, centerness, nms_pre=5000, score_thr=0.5, max_per_img=500):
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


def main(
        opt,
        data_root,
        seqs,
        output_dir,
        batch_size=12,
        img_size=(1088, 608),
        iou_thres=0.5,
        print_interval=40):
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    model = model.to(opt.device)
    model.eval()
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        fout = open(os.path.join(output_dir, seq+'.txt'), 'w')
        dataloader = datasets.LoadImages(os.path.join(data_root, seq), opt.img_size)
        timer = Timer()
        for path, img, img0 in dataloader:
            timer.tic()
            frame = int(path.split('/')[-1].split('.')[-2])
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
            output = model(blob)[0]
            width = img0.shape[1]
            height = img0.shape[0]
            inp_height = blob.shape[2]
            inp_width = blob.shape[3]
            c = np.array([width / 2., height / 2.], dtype=np.float32)
            s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
            meta = {'c': c, 's': s,
                    'out_height': inp_height,
                    'out_width': inp_width}
            hm = output['hm'].sigmoid_()
            bbox = output['bbox']
            centerness = output['centerness'].sigmoid_()
            opt.K = 200
            dets = fcos_decode(opt, hm, bbox, centerness, score_thr=0.3)
            dets = torch.cat([dets[1], dets[3], dets[4], dets[5], dets[6]])
            dets[:, 0] = dets[:, 0].clamp(min=0, max=inp_width - 1)
            dets[:, 1] = dets[:, 1].clamp(min=0, max=inp_height - 1)
            dets[:, 2] = dets[:, 2].clamp(min=0, max=inp_width - 1)
            dets[:, 3] = dets[:, 3].clamp(min=0, max=inp_height - 1)
            dets = dets.detach().cpu().numpy()

            len_det = dets.shape[0]

            rois = dets[:, :4]
            scores = dets[:, 4]
            keep = nms_detections(rois, scores.reshape(-1), nms_thresh=0.4)
            mask = np.zeros(len(rois), dtype=bool)
            mask[keep] = True
            # mask = np.ones(len(rois), dtype=bool)
            keep = np.where(mask & (scores >= 0.15))[0]
            dets = [dets[i] for i in keep]
            dets = np.array(dets)

            if dets.shape[0] == 0:
                continue

            dets[:, :2] = transform_preds(dets[:, 0:2], meta['c'], meta['s'], (meta['out_width'], meta['out_height']))
            dets[:, 2:4] = transform_preds(dets[:, 2:4], meta['c'], meta['s'], (meta['out_width'], meta['out_height']))

            timer.toc()

            for i in range(dets.shape[0]):
                x, y, w, h, score = dets[i][0], dets[i][1], \
                                    dets[i][2] - dets[i][0], dets[i][3] - dets[i][1], dets[i][4]
                fout.write("{}, {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format(frame, frame, x, y, w, h, score))
            #     cv2.rectangle(img0, (int(dets[i, 0]), int(dets[i, 1])),
            #                   (int(dets[i, 2]), int(dets[i, 3])), (0, 255, 0), 2)
            # plt.figure(figsize=(16, 9))
            # plt.imshow(img0)
            # plt.show()
        timer_avgs.append(timer.average_time)
        timer_calls.append(timer.calls)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    print(all_time)
    print(avg_time)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    opt = opts().init()
    opt.load_model = '../exp/mot/1204-fcos-litedla-576-b8/model_last.pth'
    data_root = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/testc5/'
    output_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/detections-1204"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    val_seqs_str = """
                    uav0000086_00000_v
                    uav0000117_02622_v
                    uav0000137_00458_v
                    uav0000182_00000_v
                    uav0000268_05773_v
                    uav0000305_00000_v
                    uav0000339_00001_v
    """
    test_seqs_str = """
                    uav0000009_03358_v
                    uav0000073_00600_v
                    uav0000073_04464_v
                    uav0000077_00720_v
                    uav0000088_00290_v
                    uav0000119_02301_v
                    uav0000120_04775_v
                    uav0000161_00000_v
                    uav0000188_00000_v
                    uav0000201_00000_v
                    uav0000249_00001_v
                    uav0000249_02688_v
                    uav0000297_00000_v
                    uav0000297_02761_v
                    uav0000306_00230_v
                    uav0000355_00001_v
                    uav0000370_00001_v
    """
    seqs = test_seqs_str
    seqs = [seq.strip() for seq in seqs.split()]
    with torch.no_grad():
        main(opt, data_root, seqs, output_dir, batch_size=1)
