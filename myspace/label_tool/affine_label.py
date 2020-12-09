import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt


def affine2label():
    """
    convert affine matrix to label
    use transform point by grid matrix, calculate the distance from point to transformed point
    use orb
    :return:
    """
    root_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/"
    seq_dir = root_dir + "sequences/"
    annotations_dir = root_dir + 'annotations/'
    seqs_sample = '''
                  uav0000249_00001_v
                  uav0000249_02688_v
                  '''
    affine_dir = root_dir + "affine_label_ratio2/"
    if not os.path.exists(affine_dir):
        os.makedirs(affine_dir)
    MIN_MATCH_COUNT = 10
    # 1088 is more accurate
    # seqs = [seq.strip() for seq in seqs_sample.split()]
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
        affine_dict = {}
        for i in range(0, len(seq_files) - 2, 2):
            print(i)
            image0 = cv2.imread(os.path.join(seq_dir, seq, seq_files[i]))
            image1 = cv2.imread(os.path.join(seq_dir, seq, seq_files[i + 2]))
            image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create()
            kp0, des0 = orb.detectAndCompute(image0, None)
            kp1, des1 = orb.detectAndCompute(image1, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des0, des1)
            # store all the good matchs as per Lowe's ratio test
            matches = sorted(matches, key=lambda x: x.distance)
            good = matches[:1000]
            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            else:
                M = np.eye(3, 3)
            affine_dict[seq_files[i]] = M
            x = np.linspace(0, 639, 640)
            y = np.linspace(0, 479, 480)
            [x, y] = np.meshgrid(x, y)
            x, y = x.reshape(1, -1), y.reshape(1, -1)
            pts = np.concatenate([x, y, np.ones(x.shape)], axis=0)
            trans_pts = np.dot(M, pts)
            trans_pts = trans_pts/trans_pts[2, :]
            dist = np.linalg.norm(trans_pts - pts, axis=0)
            assert dist.shape[0] == pts.shape[1]
            print(np.max(dist))
            # save the mat and label
            data = {}
            data["M"] = M
            data["dist"] = np.max(dist)
            affine_dict[seq_files[i]] = data
            # show the affine image
            # plt.figure(i, figsize=(8, 4))
            # plt.subplot(1, 3, 1)
            # plt.imshow(image0, cmap='gray')
            # plt.subplot(1, 3, 2)
            # plt.imshow(image1, cmap='gray')
            # plt.subplot(1, 3, 3)
            # result = cv2.warpPerspective(image0, M, (image0.shape[1], image0.shape[0]))
            # plt.imshow(result, cmap='gray')
            # plt.show()
        with open(os.path.join(seq_dir, affine_dir, seq + '.pickle'), 'wb') as fout:
            pickle.dump(affine_dict, fout)


def affine2label_v2():
    """
    convert affine matrix to label
    use transform point by grid matrix, calculate the distance from point to transformed point
    use surf
    :return:
    """
    root_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/"
    seq_dir = root_dir + "sequences/"
    annotations_dir = root_dir + 'annotations/'
    seqs_sample = '''
                  uav0000249_00001_v
                  uav0000249_02688_v
                  '''
    affine_dir = root_dir + "affine_label_surf_ratio2/"
    if not os.path.exists(affine_dir):
        os.makedirs(affine_dir)
    MIN_MATCH_COUNT = 10
    # 1088 is more accurate
    # seqs = [seq.strip() for seq in seqs_sample.split()]
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
        affine_dict = {}
        for i in range(0, len(seq_files) - 2, 2):
            ratio = np.min([604/width, 480/height])
            height_resize, width_resize = int(height*ratio), int(width*ratio)
            print(i)
            image0 = cv2.imread(os.path.join(seq_dir, seq, seq_files[i]))
            image1 = cv2.imread(os.path.join(seq_dir, seq, seq_files[i + 2]))
            image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            image0_resize = cv2.resize(image0, (width_resize, height_resize))
            image1_resize = cv2.resize(image1, (width_resize, height_resize))
            surf = cv2.xfeatures2d.SURF_create()
            kp0, des0 = surf.detectAndCompute(image0_resize, None)
            kp1, des1 = surf.detectAndCompute(image1_resize, None)
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des0, des1, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)
            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp0[m.queryIdx].pt/ratio for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp1[m.trainIdx].pt/ratio for m in good]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            else:
                M = np.eye(3, 3)
            affine_dict[seq_files[i]] = M
            x = np.linspace(0, 639, 640)
            y = np.linspace(0, 479, 480)
            [x, y] = np.meshgrid(x, y)
            x, y = x.reshape(1, -1), y.reshape(1, -1)
            pts = np.concatenate([x, y, np.ones(x.shape)], axis=0)
            trans_pts = np.dot(M, pts)
            trans_pts = trans_pts/trans_pts[2, :]
            dist = np.linalg.norm(trans_pts - pts, axis=0)
            assert dist.shape[0] == pts.shape[1]
            print(np.max(dist))
            # save the mat and label
            data = {}
            data["M"] = M
            data["dist"] = np.max(dist)
            affine_dict[seq_files[i]] = data
            # # show the affine image
            # if np.max(dist) > 40:
            #     plt.figure(i, figsize=(8, 4))
            #     plt.subplot(1, 3, 1)
            #     plt.imshow(image0, cmap='gray')
            #     plt.subplot(1, 3, 2)
            #     plt.imshow(image1, cmap='gray')
            #     plt.subplot(1, 3, 3)
            #     result = cv2.warpPerspective(image0, M, (image0.shape[1], image0.shape[0]))
            #     plt.imshow(result, cmap='gray')
            #     plt.show()
        with open(os.path.join(seq_dir, affine_dir, seq + '.pickle'), 'wb') as fout:
            pickle.dump(affine_dict, fout)


def get_count():
    annotation_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/affine_label_ratio2/"
    thre, stable_cnt, unstable_cnt = 30, 0, 0
    for file in os.listdir(annotation_dir):
        with open(os.path.join(annotation_dir, file), 'rb') as fin:
            affine_mat = pickle.load(fin)
        for key in affine_mat:
            print(key)
            data = affine_mat[key]
            if data['dist'] < thre:
                stable_cnt += 1
            else:
                unstable_cnt += 1
    print("stable cnt: {}, unstable cnt: {}".format(stable_cnt, unstable_cnt))


def cal_accuracy_by_dists():
    affine_label_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-val/affine_label_ratio2/"
    tracker_predict_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-val/tracker_iou_dists_2/"
    # affine_label_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/affine_label_ratio2/"
    # tracker_predict_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/tracker_iou_dists_2/"
    affine_thre, tracker_thre, stable_cnt, unstable_cnt = 40, 0.4, 0, 0
    for file in os.listdir(affine_label_dir):
        fin = open(os.path.join(affine_label_dir, file), 'rb')
        affine_label = pickle.load(fin)
        fin = open(os.path.join(tracker_predict_dir, file), 'rb')
        tracker_predict = pickle.load(fin)
        gt_labels = []
        predict_labels = []
        tp, fp, tn, fn = 0, 0, 0, 0
        for key in affine_label:
            if affine_label[key]['dist'] > affine_thre:
                gt_labels.append(1)
            else:
                gt_labels.append(0)
        for key in tracker_predict:
            dists = tracker_predict[key]
            if dists.shape[0] == 0 or dists.shape[1] == 0:
                predict_labels.append(0)
            elif np.mean(np.min(dists, axis=1)) <= tracker_thre:
                predict_labels.append(0)
            else:
                predict_labels.append(1)
        predict_labels = predict_labels[1:]
        print(len(gt_labels))
        print(len(predict_labels))
        for i in range(len(gt_labels)):
            if gt_labels[i] == 1 and predict_labels[i] == 1:
                tp += 1
            elif gt_labels[i] == 1 and predict_labels[i] == 0:
                fn += 1
            elif gt_labels[i] == 0 and predict_labels[i] == 1:
                fp += 1
            else:
                tn += 1
        print("tp: {}, fn: {}, fp: {}, tn: {}".format(tp, fn, fp, tn))
        print((tp + tn)/(tp + fn + fp + tn))


def analysis_affine_dists():
    tracker_predict_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/tracker_iou_dists_2_det0.4/"
    track_cnt, object_cnt = [], []
    for file in os.listdir(tracker_predict_dir):
        fin = open(os.path.join(tracker_predict_dir, file), 'rb')
        affine_dists = pickle.load(fin)
        for key in affine_dists:
            track_cnt.append(affine_dists[key].shape[0])
            object_cnt.append(affine_dists[key].shape[1])
        print("track min: {}, mean:{}, max:{}".format(np.min(track_cnt), np.mean(track_cnt), np.max(track_cnt)))
        print("object min: {}, mean:{}, max:{}".format(np.min(object_cnt), np.mean(object_cnt), np.max(object_cnt)))


if __name__ == "__main__":
    # affine2label()
    affine2label_v2()
    # get_count()
    # analysis_affine_dists()
    # cal_accuracy_by_dists()
