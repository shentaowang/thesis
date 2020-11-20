import sys
import numpy as np
sys.path.append('../FairMOT/src/lib/tracking_utils')
from kalman_filter import KalmanFilter
import matplotlib.pyplot as plt
import os
from cython_bbox import bbox_overlaps as bbox_ious
import cv2
import pickle
from collections import defaultdict

w, h = 1088, 608

# registration image between two


def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def tlwh_to_xyah(tlwh):
    """Convert bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`.
    """
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    ret[2] /= ret[3]
    return ret


def tlwh(mean):
    """Get current position in bounding box format `(top left x, top left y,
            width, height)`.
    """
    ret = mean[:4].copy()
    ret[2] *= ret[3]
    ret[:2] -= ret[2:]/2
    return ret


def tlwh_to_tlbr(tlwh):
    """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
    `(top left, bottom right)`.
    """
    ret = tlwh.copy()
    ret[2:] += ret[:2]
    return ret


def tlbr_to_tlwh(tlbr):
    """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
    `(top left, width, height)`.
    """
    ret = tlbr.copy()
    ret[2:] -= ret[:2]
    return ret


def get_affine():
    """
    get the affine matrix between two frames, note that the resolutionn is 1088*608
    :return:
    """
    root_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/"
    seq_dir = root_dir + "sequences/"
    affine_dir = root_dir + "affine/"
    if not os.path.exists(affine_dir):
        os.makedirs(affine_dir)
    MIN_MATCH_COUNT = 10
    # 1088 is more accurate
    for seq in os.listdir(seq_dir):
        print(seq)
        seq_files = os.listdir(os.path.join(seq_dir, seq))
        seq_files = sorted(seq_files, key=lambda x: int(x[:-4]))
        affine_dict = {}
        for i in range(len(seq_files)-1):
            print(i)
            image0 = cv2.imread(os.path.join(seq_dir, seq, seq_files[i]))
            image1 = cv2.imread(os.path.join(seq_dir, seq, seq_files[i+1]))
            image0, _, _, _ = letterbox(image0)
            image1, _, _, _ = letterbox(image1)
            image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            surf = cv2.xfeatures2d.SURF_create()
            kp0, des0 = surf.detectAndCompute(image0, None)
            kp1, des1 = surf.detectAndCompute(image1, None)
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matchs = flann.knnMatch(des0, des1, k=2)

            # store all the good matchs as per Lowe's ratio test
            good = []
            for m, n in matchs:
                if m.distance < 0.7 * n.distance:
                    good.append(m)
            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            else:
                M = np.eye(3, 3)
            affine_dict[seq_files[i]] = M
        with open(os.path.join(seq_dir, affine_dir, seq+'.pickle'), 'wb') as fout:
            pickle.dump(affine_dict, fout)


def get_affine_orig():
    """
    get the affine matrix between two frames, note that the resolutionn is original size
    :return:
    """
    root_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/"
    seq_dir = root_dir + "sequences/"
    affine_dir = root_dir + "affine_orig/"
    if not os.path.exists(affine_dir):
        os.makedirs(affine_dir)
    MIN_MATCH_COUNT = 10
    # 1088 is more accurate
    for seq in os.listdir(seq_dir):
        print(seq)
        seq_files = os.listdir(os.path.join(seq_dir, seq))
        seq_files = sorted(seq_files, key=lambda x: int(x[:-4]))
        affine_dict = {}
        for i in range(len(seq_files)-1):
            print(i)
            image0 = cv2.imread(os.path.join(seq_dir, seq, seq_files[i]))
            image1 = cv2.imread(os.path.join(seq_dir, seq, seq_files[i+1]))
            image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            surf = cv2.xfeatures2d.SURF_create()
            kp0, des0 = surf.detectAndCompute(image0, None)
            kp1, des1 = surf.detectAndCompute(image1, None)
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=10)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matchs = flann.knnMatch(des0, des1, k=2)

            # store all the good matchs as per Lowe's ratio test
            good = []
            for m, n in matchs:
                if m.distance < 0.7 * n.distance:
                    good.append(m)
            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            else:
                M = np.eye(3, 3)
            affine_dict[seq_files[i]] = M
        with open(os.path.join(seq_dir, affine_dir, seq+'.pickle'), 'wb') as fout:
            pickle.dump(affine_dict, fout)


def get_affine_orig_v2():
    """
    get the affine matrix between two frames, note that the resolutionn is original size
    filter the bbox area
    :return:
    """
    root_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/"
    seq_dir = root_dir + "sequences/"
    annotations_dir = root_dir + 'annotations/'
    affine_dir = root_dir + "affine_orig_v2/"
    if not os.path.exists(affine_dir):
        os.makedirs(affine_dir)
    MIN_MATCH_COUNT = 10
    # 1088 is more accurate
    seqs_sample = '''
                  uav0000249_00001_v
                  uav0000249_02688_v
                  '''
    seqs_str = seqs_sample
    # seqs = [seq.strip() for seq in seqs_str.split()]
    seqs = os.listdir(seq_dir)
    for seq in seqs:
        print(seq)
        # sort the seq files
        seq_files = os.listdir(os.path.join(seq_dir, seq))
        seq_files = sorted(seq_files, key=lambda x: int(x[:-4]))
        image0 = cv2.imread(os.path.join(seq_dir, seq, seq_files[0]))
        height, width = image0.shape[0], image0.shape[1]
        print("height: {}, width: {}".format(height, width))
        # first load the bbox annotations
        frame_mask = get_frame_mask(annotations_dir, seq+'.txt', width=width, height=height)
        affine_dict = {}
        for i in range(len(seq_files)-1):
            print(i)
            image0 = cv2.imread(os.path.join(seq_dir, seq, seq_files[i]))
            image1 = cv2.imread(os.path.join(seq_dir, seq, seq_files[i+1]))
            image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            surf = cv2.xfeatures2d.SURF_create()
            kp0, des0 = surf.detectAndCompute(image0, None)
            kp1, des1 = surf.detectAndCompute(image1, None)
            # filter the kp0 and des0, kp1 and des1 by mask0 and mask1
            mask0 = frame_mask[i]
            mask1 = frame_mask[i+1]
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=10)
            point_mask0 = [1 if mask0[int(i.pt[1]), int(i.pt[0])] == 1 else 0 for i in kp0]
            point_mask1 = [1 if mask1[int(i.pt[1]), int(i.pt[0])] == 1 else 0 for i in kp1]
            kp0 = [i for idx, i in enumerate(kp0) if point_mask0[idx] == 1]
            des0 = [i for idx, i in enumerate(des0) if point_mask0[idx] == 1]
            des0 = np.array(des0)
            kp1 = [i for idx, i in enumerate(kp1) if point_mask1[idx] == 1]
            des1 = [i for idx, i in enumerate(des1) if point_mask1[idx] == 1]
            des1 = np.array(des1)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matchs = flann.knnMatch(des0, des1, k=2)

            # store all the good matchs as per Lowe's ratio test
            good = []
            for m, n in matchs:
                if m.distance < 0.7 * n.distance:
                    good.append(m)
            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            else:
                M = np.eye(3, 3)
            affine_dict[seq_files[i]] = M
        with open(os.path.join(seq_dir, affine_dir, seq+'.pickle'), 'wb') as fout:
            pickle.dump(affine_dict, fout)


def get_frame_bbox(annotations_dir, seq):
    """
    get one seq, bbox, frame id
    :param annotations_dir:
    :param seq:
    :return:
    x, y, w, h
    """
    root_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/"
    bbox = []
    frame_id = []
    trace_bbox = []
    trace_frame_id = []
    annotations_dir = root_dir + 'annotations/'
    gt_txt = os.path.join(annotations_dir, seq)
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
    tid_last = gt[0][1]
    for fid, tid, x, y, w, h, mark, label, _, _ in gt:
        if int(mark) == 0 or int(label) == 0 or int(label) == 11:
            continue
        if tid == tid_last:
            trace_bbox.append([x, y, w, h])
            trace_frame_id.append(fid)
        else:
            tid_last = tid
            if len(trace_bbox) > 1:
                bbox.append(np.array(trace_bbox))
                frame_id.append(np.array(trace_frame_id))
            trace_bbox = [[x, y, w, h]]
            trace_frame_id = [fid]

    return bbox, frame_id


def get_frame_mask(annotations_dir, seq, width, height):
    """
    get the frame mask, where 1 is registration valid, 0 is object
    Args:
        annotations_dir:
        seq:
        width:
        height:

    Returns:

    """
    frame_bbox = defaultdict(list)
    gt_txt = os.path.join(annotations_dir, seq)
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
    for fid, tid, x, y, w, h, mark, label, _, _ in gt:
        fid = int(fid)
        frame_bbox[fid].append([x, y, w, h])
    frame_mask = {}
    for key in frame_bbox:
        mask = np.ones((height, width))
        for bbox in frame_bbox[key]:
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            mask[x: x+w, y: y+h] = 0
        frame_mask[key] = mask
    # frame id is begin width 1
    frame_mask[0] = frame_mask[1]
    return frame_mask


def eval_pos():
    """
    input one long track, show if the kalman is accurate for predicting
    :return:
    """
    annotations_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/annotations"
    all_iou = []
    seqs_sample = '''
                  uav0000249_00001_v
                  uav0000249_02688_v
                  '''
    seqs_str = seqs_sample
    seqs = [seq.strip() for seq in seqs_str.split()]
    for seq in seqs:
        print(seq)
        bbox, frame_id = get_frame_bbox(annotations_dir, seq + '.txt')
        predict_bbox = []
        for idx in range(len(bbox)):
            kalman_filter = KalmanFilter()
            trace_bbox = bbox[idx]
            trace_predict_bbox = []
            mean, covariance = kalman_filter.initiate(tlwh_to_xyah(trace_bbox[0]))
            for i in range(1, trace_bbox.shape[0]):
                mean, covariance = kalman_filter.predict(mean, covariance)
                trace_predict_bbox.append(tlwh(mean))
                mean, covariance = kalman_filter.update(mean, covariance, tlwh_to_xyah(trace_bbox[i]))

            trace_predict_bbox = np.array(trace_predict_bbox)
            for i in range(trace_predict_bbox.shape[0]):
                trace_predict_bbox[i] = tlwh_to_tlbr(trace_predict_bbox[i])
            for i in range(trace_bbox.shape[0]):
                trace_bbox[i] = tlwh_to_tlbr(trace_bbox[i])

            predict_bbox.append(trace_predict_bbox)
            bbox[idx] = bbox[idx][1:]
            frame_id[idx] = frame_id[idx][1:]
            assert bbox[idx].shape[0] == predict_bbox[idx].shape[0]
        iou = []
        for i in range(len(bbox)):
            trace_iou = []
            trace_bbox = bbox[i]
            trace_predict_bbx = predict_bbox[i]
            for j in range(trace_bbox.shape[0]):
                iou_val = bbox_ious(np.ascontiguousarray(trace_bbox[j][np.newaxis, :], dtype=np.float),
                                    np.ascontiguousarray(trace_predict_bbx[j][np.newaxis, :], dtype=np.float))
                trace_iou.append(iou_val)
            iou.append(np.array(trace_iou))
        iou = [int(np.mean(i)*100) for i in iou]
        all_iou += iou
    bins = np.zeros(101)
    for i in all_iou:
        bins[i] += 1
    plt.bar(np.arange(101), bins)
    plt.ylabel('num')
    plt.xlabel('IoU*100')
    plt.show()


def eval_pos_affine():
    """
    eval the iou with kalman and registration
    :return:
    """
    root_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/"
    seq_dir = root_dir + "sequences/"
    annotations_dir = root_dir + 'annotations/'
    affine_dir = root_dir + "affine_orig/"
    all_iou = []
    seqs_sample = '''
                  uav0000249_00001_v
                  uav0000249_02688_v
                  '''
    seqs_str = seqs_sample
    seqs = [seq.strip() for seq in seqs_str.split()]
    for seq in seqs:
        image_file = os.listdir(os.path.join(seq_dir, seq))[0]
        image = cv2.imread(os.path.join(seq_dir, seq, image_file))
        orig_h, orig_w = image.shape[:2]

        with open(os.path.join(affine_dir, seq+'.pickle'), 'rb') as fin:
            affine_dict = pickle.load(fin)

        bbox, frame_id = get_frame_bbox(annotations_dir, seq + '.txt')
        predict_bbox = []
        for i in range(len(bbox)):
            # convert to std resolution
            bbox[i][:, 0] = bbox[i][:, 0]
            bbox[i][:, 1] = bbox[i][:, 1]
            bbox[i][:, 2] = bbox[i][:, 2]
            bbox[i][:, 3] = bbox[i][:, 3]

            # for j in range(bbox[i].shape[0]):
            #     bbox[i][j] = tlwh_to_tlbr(bbox[i][j])
        for idx in range(len(bbox)):
            kalman_filter = KalmanFilter()
            trace_bbox = bbox[idx]
            trace_predict_bbox = []
            mean, covariance = kalman_filter.initiate(tlwh_to_xyah(trace_bbox[0]))
            for i in range(1, trace_bbox.shape[0]):
                # i-1 to i M
                frame_name = "{:07d}.jpg".format(int(frame_id[idx][i-1]))
                M = affine_dict[frame_name]
                bbox_infer = tlwh(mean)
                bbox_infer = tlwh_to_tlbr(bbox_infer)
                bbox_expand = np.ones((3, 4))
                bbox_expand[:2, 0] = bbox_infer[:2]
                bbox_expand[:2, 1] = bbox_infer[2:]
                # tr
                bbox_expand[:2, 2] = bbox_infer[2], bbox_infer[1]
                # bl
                bbox_expand[:2, 3] = bbox_infer[0], bbox_infer[3]
                bbox_expand = np.dot(M, bbox_expand)
                for t in range(bbox_expand.shape[1]):
                    bbox_expand[:2, t] /= bbox_expand[2, t]
                # bbox_infer[:2] = bbox_expand[:2, 0]
                # bbox_infer[2:] = bbox_expand[:2, 1]
                # get the out bounding bbox
                bbox_infer[0] = min(bbox_expand[0, :])
                bbox_infer[1] = min(bbox_expand[1, :])
                bbox_infer[2] = max(bbox_expand[0, :])
                bbox_infer[3] = max(bbox_expand[1, :])
                bbox_infer = tlbr_to_tlwh(bbox_infer)
                # print(bbox_infer)
                trace_predict_bbox.append(bbox_infer)
                # move = mean[:4] - tlwh_to_xyah(bbox_infer)
                # if np.sum(np.square(move)[:2]) > 32*32:
                #     print(move)
                #     print(idx, frame_name)
                # print(mean)
                mean[:4] = tlwh_to_xyah(bbox_infer)
                # print(mean)
                mean, covariance = kalman_filter.predict(mean, covariance)
                mean, covariance = kalman_filter.update(mean, covariance, tlwh_to_xyah(trace_bbox[i]))

            trace_predict_bbox = np.array(trace_predict_bbox)
            for i in range(trace_predict_bbox.shape[0]):
                trace_predict_bbox[i] = tlwh_to_tlbr(trace_predict_bbox[i])
            for i in range(trace_bbox.shape[0]):
                trace_bbox[i] = tlwh_to_tlbr(trace_bbox[i])

            predict_bbox.append(trace_predict_bbox)
            bbox[idx] = bbox[idx][1:]
            frame_id[idx] = frame_id[idx][1:]
            assert bbox[idx].shape[0] == predict_bbox[idx].shape[0]
        iou = []
        for i in range(len(bbox)):
            trace_iou = []
            trace_bbox = bbox[i]
            trace_predict_bbx = predict_bbox[i]
            for j in range(trace_bbox.shape[0]):
                iou_val = bbox_ious(np.ascontiguousarray(trace_bbox[j][np.newaxis, :], dtype=np.float),
                                    np.ascontiguousarray(trace_predict_bbx[j][np.newaxis, :], dtype=np.float))
                trace_iou.append(iou_val)
            iou.append(np.array(trace_iou))
        iou = [int(np.mean(i) * 100) for i in iou]
        all_iou += iou
    bins = np.zeros(101)
    for i in all_iou:
        bins[i] += 1
    plt.bar(np.arange(101), bins)
    plt.ylabel('num')
    plt.xlabel('iou(*100)')
    plt.show()


def test_affine():
    """
    test the affine matrix, check the point transformed by M is correct
    :return:
    """
    root_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/"
    seq_dir = root_dir + "sequences/"
    affine_dir = root_dir + "affine_orig_v2/"
    MIN_MATCH_COUNT = 10
    # 1088 is more accurate
    seqs_sample = '''
                  uav0000249_00001_v
                  uav0000249_02688_v
                  '''
    seqs_str = seqs_sample
    seqs = [seq.strip() for seq in seqs_str.split()]
    for seq in seqs:
        print(seq)
        with open(os.path.join(affine_dir, seq+'.pickle'), 'rb') as fin:
            affine_dict = pickle.load(fin)
        seq_files = os.listdir(os.path.join(seq_dir, seq))
        seq_files = sorted(seq_files, key=lambda x: int(x[:-4]))
        for i in range(34, len(seq_files)-1):
            frame_name = "{:07d}.jpg".format(i)
            M = affine_dict[frame_name]
            print(i)
            image0 = cv2.imread(os.path.join(seq_dir, seq, seq_files[i]))
            image1 = cv2.imread(os.path.join(seq_dir, seq, seq_files[i+1]))
            image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            # surf = cv2.xfeatures2d.SURF_create()
            # kp0, des0 = surf.detectAndCompute(image0, None)
            # kp1, des1 = surf.detectAndCompute(image1, None)
            # FLANN_INDEX_KDTREE = 0
            # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            # search_params = dict(checks=10)
            #
            # flann = cv2.FlannBasedMatcher(index_params, search_params)
            # matchs = flann.knnMatch(des0, des1, k=2)
            #
            # # store all the good matchs as per Lowe's ratio test
            # good = []
            # for m, n in matchs:
            #     if m.distance < 0.7 * n.distance:
            #         good.append(m)
            # if len(good) > MIN_MATCH_COUNT:
            #     src_pts = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            #     dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            #     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # else:
            #     M = np.eye(3, 3)

            image0_transform = cv2.warpPerspective(image0, M, (image0.shape[1], image0.shape[0]))
            bbox = np.array([540, 540, 600, 1079])
            bbox_expand = np.ones((3, 2))
            bbox_expand[:2, 0] = bbox[:2]
            bbox_expand[:2, 1] = bbox[2:]
            bbox_expand = np.dot(M, bbox_expand)
            bbox_transform = np.concatenate([bbox_expand[:2, 0], bbox_expand[:2, 1]])
            bbox_transform = bbox_transform.astype(np.uint64)

            # show the images
            plt.figure(i, figsize=(16, 9))
            plt.subplot(2, 2, 1)
            cv2.rectangle(image0, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.rectangle(image0, (bbox_transform[0], bbox_transform[1]), (bbox_transform[2], bbox_transform[3]),
                          (0, 0, 255), 2)
            plt.imshow(image0)
            plt.subplot(2, 2, 2)
            cv2.rectangle(image1, (bbox_transform[0], bbox_transform[1]), (bbox_transform[2], bbox_transform[3]),
                          (0, 255, 0), 2)
            plt.imshow(image1)
            plt.subplot(2, 2, 3)
            cv2.rectangle(image0_transform, (bbox_transform[0], bbox_transform[1]), (bbox_transform[2], bbox_transform[3]),
                          (0, 255, 0), 2)
            plt.imshow(image0_transform)
            plt.show()


if __name__ == '__main__':
    eval_pos()
    # eval_pos_affine()
    # get_affine()
    # get_affine_orig()
    # test_affine()
    # get_affine_orig_v2()
