import numpy as np
import os
import pickle
from cython_bbox import bbox_overlaps as bbox_ious
import cv2
import matplotlib.pyplot as plt


"""
analysis centerness, scores for classification
"""

def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def eval_precision(predict, gt):
    tp, fp = 0, 0
    for i, j in zip(predict, gt):
        if i == 1 and j == 1:
            tp += 1
        elif i == 1 and j == 0:
            fp += 1
    return tp/(fp+tp+1)


def eval_recall(predict, gt):
    tp, fn = 0, 0
    for i, j in zip(predict, gt):
        if j == 1 and i == 1:
            tp += 1
        elif j == 1 and i == 0:
            fn += 1
    return tp/(tp+fn+1)


def eval_f1(predict, gt):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i, j in zip(predict, gt):
        if i == 1 and j == 1:
            tp += 1
        elif i == 1 and j == 0:
            fp += 1
        elif i == 0 and j == 1:
            fn += 1
        else:
            tn += 1
    precision = tp/(fp+tp+1)
    recall = tp/(tp+fn+1)
    if precision == 0 or recall == 0:
        return 0
    return 2*precision*recall/(precision+recall)


def main():
    scoes_dir = "/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/scores_base"
    cls_scores = []
    centerness_scores = []
    gt_label = []
    ious = []
    for seq in os.listdir(scoes_dir):
        config_file = "/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/valc5/{}/seqinfo.ini".\
            format(seq)
        fin = open(config_file, 'r')
        width = None
        height = None
        for line in fin.readlines():
            if "imWidth" in line:
                width = int(line.split('=')[-1])
            if "imHeight" in line:
                height = int(line.split('=')[-1])
        for file in os.listdir(os.path.join(scoes_dir, seq)):
            fin_scores = open(os.path.join(scoes_dir, seq, file), 'rb')
            data = pickle.load(fin_scores)
            if len(data) == 0:
                continue
            atlbrs = [i[2] for i in data]
            gt_path = "/home/sdb/wangshentao/myspace/thesis/data/" \
                      "visdrone_2019_mot/labels_with_ids/valc5/{}/{}".format(seq, file.split('.')[0] + ".txt")
            fin_gt = open(gt_path, 'r')
            btlbrs = []
            for line in fin_gt.readlines():
                line = line.strip('\n')
                line = line.split()
                btlbrs.append(line[2:6])
            btlbrs = [[float(i[0]), float(i[1]), float(i[2]), float(i[3])] for i in btlbrs]
            btlbrs = [[i[0]*width, i[1]*height, i[2]*width, i[3]*height] for i in btlbrs]
            btlbrs = [[i[0]-i[2]/2, i[1]-i[3]/2, i[0]+i[2]/2, i[1]+i[3]/2] for i in btlbrs]
            dist = iou_distance(atlbrs, btlbrs)
            for i in range(dist.shape[0]):
                if np.min(dist[i]) <= 0.5:
                    gt_label.append(1)
                    ious.append(np.min(dist[i]))
                else:
                    gt_label.append(0)

            for i in range(len(data)):
                cls_scores.append(data[i][0])
                centerness_scores.append(data[i][1])
    cls_scores = np.array(cls_scores)
    centerness_scores = np.array(centerness_scores)
    gt_label = np.array(gt_label)
    combine_scores = cls_scores * centerness_scores
    precisions1 = []
    recalls1 = []
    f1 = []
    print("associate bbox: {}, mean iou: {}".format(gt_label.sum(), 1-np.mean(ious)))
    for threshold in np.linspace(0, 1, 101):
        pred = cls_scores > threshold
        precision = eval_precision(pred, gt_label)
        recall = eval_recall(pred, gt_label)
        f = eval_f1(pred, gt_label)
        precisions1.append(precision)
        recalls1.append(recall)
        f1.append(f)

    precisions2 = []
    recalls2 = []
    f2 = []
    for threshold in np.linspace(0, 1, 101):
        pred = centerness_scores > threshold
        precision = eval_precision(pred, gt_label)
        recall = eval_recall(pred, gt_label)
        f = eval_f1(pred, gt_label)
        precisions2.append(precision)
        recalls2.append(recall)
        f2.append(f)

    precisions3 = []
    recalls3 = []
    f3 = []
    for threshold in np.linspace(0, 1, 101):
        pred = combine_scores > threshold
        precision = eval_precision(pred, gt_label)
        recall = eval_recall(pred, gt_label)
        f = eval_f1(pred, gt_label)
        precisions3.append(precision)
        recalls3.append(recall)
        f3.append(f)

    plt.figure(1)
    plt.plot(np.linspace(0, 1, 101), precisions1, label='class')
    plt.plot(np.linspace(0, 1, 101), precisions2, label='centerness')
    plt.plot(np.linspace(0, 1, 101), precisions3, label='combine')
    plt.legend()

    plt.figure(2)
    plt.plot(np.linspace(0, 1, 101), recalls1, label='class')
    plt.plot(np.linspace(0, 1, 101), recalls2, label='centerness')
    plt.plot(np.linspace(0, 1, 101), recalls3, label='combine')
    plt.legend()

    plt.figure(3)
    plt.plot(np.linspace(0, 1, 101), f1, label='class')
    plt.plot(np.linspace(0, 1, 101), f2, label='centerness')
    plt.plot(np.linspace(0, 1, 101), f3, label='combine')
    plt.legend()
    plt.show()


def plot_track_bbox():
    scoes_dir = "/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/scores"
    image_dir = "/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/valc5"
    cls_scores = []
    centerness_scores = []
    gt_label = []
    for seq in os.listdir(scoes_dir):
        config_file = "/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/valc5/{}/seqinfo.ini".\
            format(seq)
        fin = open(config_file, 'r')
        width = None
        height = None
        for line in fin.readlines():
            if "imWidth" in line:
                width = int(line.split('=')[-1])
            if "imHeight" in line:
                height = int(line.split('=')[-1])
        for file in os.listdir(os.path.join(scoes_dir, seq)):
            fin_scores = open(os.path.join(scoes_dir, seq, file), 'rb')
            data = pickle.load(fin_scores)
            if len(data) == 0:
                continue
            atlbrs = [i[2] for i in data]

            gt_path = "/home/sdb/wangshentao/myspace/thesis/data/" \
                      "visdrone_2019_mot/labels_with_ids/valc5/{}/{}".format(seq, file.split('.')[0] + ".txt")
            fin_gt = open(gt_path, 'r')
            btlbrs = []
            for line in fin_gt.readlines():
                line = line.strip('\n')
                line = line.split()
                btlbrs.append(line[2:6])
            btlbrs = [[float(i[0]), float(i[1]), float(i[2]), float(i[3])] for i in btlbrs]
            btlbrs = [[i[0]*width, i[1]*height, i[2]*width, i[3]*height] for i in btlbrs]
            btlbrs = [[i[0]-i[2]/2, i[1]-i[3]/2, i[0]+i[2]/2, i[1]+i[3]/2] for i in btlbrs]

            image = cv2.imread(os.path.join(image_dir, seq, file.split('.')[0] + '.jpg'))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for bbox in atlbrs:
                bbox = [int(i) for i in bbox]
                image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0))
            for bbox in btlbrs:
                bbox = [int(i) for i in bbox]
                image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0))
            plt.figure(figsize=(16, 9))
            plt.imshow(image)
            plt.show()


if __name__ == "__main__":
    main()
    # plot_track_bbox()
