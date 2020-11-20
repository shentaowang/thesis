from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from models import *
from models.decode import mot_decode
from models.model import create_model, load_model
from models.utils import _tranpose_and_gather_feat
from tracker import matching
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.log import logger
from tracking_utils.utils import *
from utils.post_process import ctdet_post_process
from utils.image import transform_preds

from .basetrack import BaseTrack, TrackState

from mmdet.core import multiclass_nms, distance2bbox


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, buffer_size=30):
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = opt.K
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter()

    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def update(self, im_blob, img0):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.opt.reg_offset else None
            dets, inds = mot_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])
        # dets = np.concatenate([dets[1], dets[2], dets[3], dets[4], dets[5], dets[6]])

        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]

        # vis
        '''
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img0, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
        cv2.imshow('dets', img0)
        cv2.waitKey(0)
        id0 = id0-1
        '''

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        # for strack in strack_pool:
        #     strack.predict()
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        # dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.opt.embedding_thres)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks


def tlwh(mean):
    """Get current position in bounding box format `(top left x, top left y,
            width, height)`.
    """
    ret = mean[:4].copy()
    ret[2] *= ret[3]
    ret[:2] -= ret[2:] / 2
    return ret


def tlwh_to_xyah(tlwh):
    """Convert bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`.
    """
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    ret[2] /= ret[3]
    return ret


def affine_transform(stracks, affine_mat):
    # if len(stracks) < 1:
    #     return
    # bbox_infer = [tlwh(i.mean) for i in stracks]
    # bbox_infer = np.array(bbox_infer)
    # bbox_infer[:, 2:] += bbox_infer[:, :2]
    # bbox_tl = np.concatenate([bbox_infer[:, :2], np.ones((bbox_infer.shape[0], 1))], axis=1).T
    # bbox_br = np.concatenate([bbox_infer[:, 2:], np.ones((bbox_infer.shape[0], 1))], axis=1).T
    # bbox_tr = np.concatenate([bbox_infer[:, 2:3], bbox_infer[:, 1:2], np.ones((bbox_infer.shape[0], 1))], axis=1).T
    # bbox_bl = np.concatenate([bbox_infer[:, 0:1], bbox_infer[:, 3:4], np.ones((bbox_infer.shape[0], 1))], axis=1).T
    # bbox_tl = np.dot(affine_mat, bbox_tl).T
    # bbox_br = np.dot(affine_mat, bbox_br).T
    # bbox_tr = np.dot(affine_mat, bbox_tr).T
    # bbox_bl = np.dot(affine_mat, bbox_bl).T
    # bbox_tl = bbox_tl/bbox_tl[:, 2:]
    # bbox_br = bbox_br/bbox_br[:, 2:]
    # bbox_tr = bbox_tr/bbox_tr[:, 2:]
    # bbox_bl = bbox_bl/bbox_bl[:, 2:]
    # bbox_infer[:, 0] = np.min(np.concatenate([bbox_tl[:, 0:1], bbox_bl[:, 0:1]], axis=1), axis=1)
    # bbox_infer[:, 1] = np.min(np.concatenate([bbox_tl[:, 1:2], bbox_tr[:, 1:2]], axis=1), axis=1)
    # bbox_infer[:, 2] = np.max(np.concatenate([bbox_br[:, 0:1], bbox_tr[:, 0:1]], axis=1), axis=1)
    # bbox_infer[:, 3] = np.max(np.concatenate([bbox_br[:, 1:2], bbox_bl[:, 1:2]], axis=1), axis=1)
    # bbox_infer[:, 2:] -= bbox_infer[:, :2]
    # for i in range(len(stracks)):
    #     stracks[i].mean[:4] = tlwh_to_xyah(bbox_infer[i])

    for i in range(len(stracks)):
        mean = stracks[i].mean.copy()
        bbox_infer = tlwh(mean)
        bbox_infer[2:] += bbox_infer[:2]
        bbox_expand = np.ones((3, 4))
        bbox_expand[:2, 0] = bbox_infer[:2]
        bbox_expand[:2, 1] = bbox_infer[2:]
        # tr
        bbox_expand[:2, 2] = bbox_infer[2], bbox_infer[1]
        # bl
        bbox_expand[:2, 3] = bbox_infer[0], bbox_infer[3]
        bbox_expand = np.dot(affine_mat, bbox_expand)
        for t in range(bbox_expand.shape[1]):
            bbox_expand[:2, t] /= bbox_expand[2, t]
        # get the out bounding bbox
        bbox_infer[0] = min(bbox_expand[0, :])
        bbox_infer[1] = min(bbox_expand[1, :])
        bbox_infer[2] = max(bbox_expand[0, :])
        bbox_infer[3] = max(bbox_expand[1, :])
        # bbox_infer[:2] = bbox_expand[:2, 0]
        # bbox_infer[2:] = bbox_expand[:2, 1]
        bbox_infer[2:] -= bbox_infer[:2]
        # print('before', mean)
        mean[:4] = tlwh_to_xyah(bbox_infer)
        # print('after', mean)
        stracks[i].mean = mean


class FcosJDETracker(JDETracker):
    def __init__(self, opt, frame_rate=30):
        super(FcosJDETracker, self).__init__(opt, frame_rate)

        self.des_last, self.kp_last = None, None
        self.des_cur, self.kp_cur = None, None
        self.affine_mat = np.eye(3, 3)
        self.min_match_count = 4

    def affine_estimate(self, img0):
        img_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        orb.setNLevels(1)
        kp, des = orb.detectAndCompute(img_gray, None)
        if self.kp_cur is None:
            self.kp_cur, self.des_cur = kp, des
        else:
            self.kp_last, self.des_last = self.kp_cur, self.des_cur
            self.kp_cur, self.des_cur = kp, des

        if self.kp_last is None:
            self.affine_mat = np.eye(3, 3)
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matchs = bf.match(self.des_last, self.des_cur)
            if len(matchs) > self.min_match_count:
                src_pts = np.float32([self.kp_last[m.queryIdx].pt for m in matchs]).reshape(-1, 1, 2)
                dst_pts = np.float32([self.kp_cur[m.trainIdx].pt for m in matchs]).reshape(-1, 1, 2)
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, maxIters=200)
                self.affine_mat = M

    def fcos_decode(self, scores, bbox_pred, centerness, nms_pre=5000, score_thr=0.6,
                    nms=dict(type='nms', iou_thr=0.4), max_per_img=1000):
        h, w = self.opt.output_h, self.opt.output_w
        scores = scores[0].permute(1, 2, 0).reshape(-1, self.opt.num_classes)
        padding = scores.new_zeros(scores.shape[0], 1)
        scores = torch.cat([scores, padding], dim=1)
        centerness = centerness[0].permute(1, 2, 0).reshape(-1)
        bbox_pred = bbox_pred[0].permute(1, 2, 0).reshape(-1, 4)
        x_range = torch.arange(0, w * self.opt.down_ratio, self.opt.down_ratio, dtype=torch.float)
        y_range = torch.arange(0, h * self.opt.down_ratio, self.opt.down_ratio, dtype=torch.float)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1) + self.opt.down_ratio // 2
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
            nms,
            max_per_img,
            centerness
        )

        results = {}
        for i in range(0, self.opt.num_classes):
            inds = (det_labels == i)
            results[i+1] = det_bboxes[inds]

        return results

    def update(self, im_blob, img0):
        self.affine_estimate(img0)
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height,
                'out_width': inp_width}
        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            bbox = output['bbox']
            id_feature = output['id']
            centerness = output['centerness'].sigmoid_()
            id_feature = F.normalize(id_feature, dim=1)
            id_feature = id_feature[0].permute(1, 2, 0).view(-1, self.opt.reid_dim)
            # use conf threshold
            dets = self.fcos_decode(hm, bbox, centerness, max_per_img=self.opt.K)
            dets_copy = dets.copy()
            # dets_copy = torch.cat([dets_copy[1], dets_copy[2], dets_copy[3], dets_copy[4], dets_copy[5]])
            # dets_copy = dets_copy.detach().cpu().numpy()
            # # only track pedestrian(1) car(4), van(5), truck(6), bus(9)
            dets = torch.cat([dets[1], dets[3], dets[4], dets[5], dets[6]])
            # dets = torch.cat([dets[1], dets[2], dets[3], dets[4], dets[5], dets[6],
            #                   dets[7], dets[8], dets[9], dets[10]])
            dets[:, 0] = dets[:, 0].clamp(min=0, max=inp_width - 1)
            dets[:, 1] = dets[:, 1].clamp(min=0, max=inp_height - 1)
            dets[:, 2] = dets[:, 2].clamp(min=0, max=inp_width - 1)
            dets[:, 3] = dets[:, 3].clamp(min=0, max=inp_height - 1)
            # first extract the feature
            center_idx = (dets[:, 0:1] + dets[:, 2:3]) / (2 * self.opt.down_ratio)
            center_idx = center_idx.long()
            center_idy = (dets[:, 1:2] + dets[:, 3:4]) / (2 * self.opt.down_ratio)
            center_idy = center_idy.long()
            ind = center_idy * self.opt.output_w + center_idx
            id_feature = id_feature[ind]
            id_feature = id_feature.squeeze(1)

            dets = dets.detach().cpu().numpy()

            dets[:, :2] = transform_preds(dets[:, 0:2], meta['c'], meta['s'], (meta['out_width'], meta['out_height']))
            dets[:, 2:4] = transform_preds(dets[:, 2:4], meta['c'], meta['s'], (meta['out_width'], meta['out_height']))

            id_feature = id_feature.cpu().numpy()

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        for strack in strack_pool:
            strack.predict()
        # STrack.multi_predict(strack_pool)

        # use affine to update the pos for strack_pool
        affine_transform(strack_pool, self.affine_mat)

        # # vis
        # if self.frame_id > 32:
        #     for i in range(1, 6):
        #         dets_copy[i] = dets_copy[i].detach().cpu().numpy()
        #         dets_copy[i][:, :2] = transform_preds(dets_copy[i][:, :2], meta['c'], meta['s'], (meta['out_width'],
        #                                                                                           meta['out_height']))
        #         dets_copy[i][:, 2:4] = transform_preds(dets_copy[i][:, 2:4], meta['c'], meta['s'], (meta['out_width'],
        #                                                                                             meta['out_height']))
        #     plt.figure(self.frame_id, figsize=(16, 9))
        #     plt.subplot(2, 2, 1)
        #     img_sub1 = img0.copy()
        #     for label in range(1, 6):
        #         det = dets_copy[label]
        #         for i in range(0, det.shape[0]):
        #             bbox = det[i][0:4]
        #             cv2.rectangle(img_sub1, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        #             cv2.putText(img_sub1, str(label), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        #     img_sub1 = cv2.cvtColor(img_sub1, cv2.COLOR_BGR2RGB)
        #     plt.imshow(img_sub1)
        #     plt.subplot(2, 2, 2)
        #     centerness_map = centerness[0].permute(1, 2, 0)
        #     centerness_map = centerness_map * 255
        #     centerness_map = centerness_map.cpu().numpy().astype(np.uint8)[:, :, 0]
        #     plt.imshow(centerness_map, cmap='gray')
        #     plt.subplot(2, 2, 3)
        #     plt.imshow(centerness_map, cmap='gray')
        #     plt.plot(center_idx.cpu().numpy(), center_idy.cpu().numpy(), 'ro')
        #     plt.subplot(2, 2, 4)
        #     # show the affine pos
        #     img_sub4 = img0.copy()
        #     for track in strack_pool:
        #         mean = track.mean.copy()
        #         bbox = tlwh(mean)
        #         bbox = [int(i) for i in bbox]
        #         cv2.rectangle(img_sub4, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
        #     affine_transform(strack_pool, affine_mat)
        #     for track in strack_pool:
        #         mean = track.mean.copy()
        #         bbox = tlwh(mean)
        #         bbox = [int(i) for i in bbox]
        #         cv2.rectangle(img_sub4, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 0), 2)
        #     img_sub4 = cv2.cvtColor(img_sub4, cv2.COLOR_BGR2RGB)
        #     plt.imshow(img_sub4)
        #     plt.show()

        dists = matching.embedding_distance(strack_pool, detections)
        # dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.opt.embedding_thres)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
