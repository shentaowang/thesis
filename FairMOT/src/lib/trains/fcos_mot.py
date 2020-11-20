from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.decode import mot_decode
from models.losses import FocalLossBase
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss, IoULoss
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process

from mmdet.core import distance2bbox
import sys
sys.path.append('../../../../mechanical/code/')
from mmdetection.mmdet.models.losses import CrossEntropyLoss

from .base_trainer import BaseTrainer


class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = FocalLossBase()
        self.reg = IoULoss()
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.CenternessLoss = CrossEntropyLoss(use_sigmoid=True)
        # self.TriLoss = TripletLoss()
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, bbox_loss, centerness_loss, id_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])
            # print(output['hm'].shape, batch['hm'].shape)
            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks

            if opt.bbox_weght > 0:
                pos_bbox_targets = batch['bbox'][batch['pos_ind'] > 0]
                output['bbox'] = output['bbox'].permute(0, 2, 3, 1).reshape(output['bbox'].shape[0], -1, 4)
                pos_bbox_preds = output['bbox'][batch['pos_ind'] > 0]
                h, w = self.opt.output_h, self.opt.output_w
                x_range = torch.arange(0, w*opt.down_ratio, opt.down_ratio, dtype=torch.int64)
                y_range = torch.arange(0, h*opt.down_ratio, opt.down_ratio, dtype=torch.int64)
                y, x = torch.meshgrid(y_range, x_range)
                points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1) + opt.down_ratio//2
                points = points.unsqueeze(0).repeat(batch['pos_ind'].shape[0], 1, 1)
                self.pos_points = points[batch['pos_ind'] > 0]
                # need to optimize, clear the code
                self.pos_points = self.pos_points.detach().cuda(pos_bbox_preds.device).float()
                pos_decoded_bbox_preds = distance2bbox(self.pos_points, pos_bbox_preds,
                                                       (self.opt.input_w, self.opt.input_h))
                pos_decoded_bbox_targets = distance2bbox(self.pos_points, pos_bbox_targets,
                                                         (self.opt.input_w, self.opt.input_h))

                bbox_loss = self.reg(pos_decoded_bbox_preds, pos_decoded_bbox_targets)

            if opt.centerness_weight > 0:
                # output['centerness'] = _sigmoid(output['centerness'])
                pos_centerness_targets = batch['centerness'][batch['pos_ind'] > 0]
                pos_centerness_preds = output['centerness'].permute(0, 2, 3, 1).\
                    reshape(batch['centerness'].shape[0], -1)
                pos_centerness_preds = pos_centerness_preds[batch['pos_ind'] > 0]
                centerness_loss = self.CenternessLoss(pos_centerness_preds.unsqueeze(1),
                                                      pos_centerness_targets.unsqueeze(1))

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = id_head[batch['reg_mask'] > 0].contiguous()
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch['ids'][batch['reg_mask'] > 0]
                id_output = self.classifier(id_head).contiguous()
                id_loss += self.IDLoss(id_output, id_target)
                # id_loss += self.IDLoss(id_output, id_target) + self.TriLoss(id_head, id_target)

        # loss = opt.hm_weight * hm_loss + opt.bbox_weght * bbox_loss + opt.bbox_weght * centerness_loss \
        #        + opt.id_weight * id_loss

        det_loss = opt.hm_weight * hm_loss + opt.bbox_weght * bbox_loss + opt.centerness_weight * centerness_loss

        loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
        loss *= 0.5

        # print(loss, hm_loss, wh_loss, off_loss, id_loss)

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'bbox_loss': bbox_loss, 'centerness_loss': centerness_loss, 'id_loss': id_loss}
        return loss, loss_stats


class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'bbox_loss', 'centerness_loss', 'id_loss']
        loss = MotLoss(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = None

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          nms_pre=500,
                          score_thr=0.5,
                          nms=dict(type='nms', iou_thr=0.5),
                          max_per_img=100,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1)

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                max_scores, _ = (scores).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            score_thr,
            nms,
            max_per_img,
            score_factors=mlvl_centerness)
        # det_bboxes, det_labels = multiclass_nms(
        #     mlvl_bboxes,
        #     mlvl_scores,
        #     cfg.score_thr,
        #     cfg.nms,
        #     cfg.max_per_img)
        return det_bboxes, det_labels