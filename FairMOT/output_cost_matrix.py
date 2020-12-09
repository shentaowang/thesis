import numpy as np
import os
from cython_bbox import bbox_overlaps as bbox_ious


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


def load_mot(detection_file):
    data = []
    raw = np.genfromtxt(detection_file, delimiter=',', dtype=np.float32)
    if np.isnan(raw).all():
        raw = np.genfromtxt(detection_file, defaultfmt=' ', dtype=np.float32)
    end_frame = int(np.max(raw[:, 0]))
    for i in range(1, end_frame+1):
        idx = raw[:, 0] == i
        bbox = raw[idx, 2:6]
        bbox[:, 2:4] += bbox[:, 0:2]
        bbox -= 1
        scores = raw[idx, 6:]
        id = raw[idx, 1:2]
        data.append(np.concatenate([bbox, id, scores], axis=1))
    return data


def main():
    seqs_str = '''
              uav0000086_00000_v
              uav0000117_02622_v
              uav0000137_00458_v
              uav0000182_00000_v
              uav0000268_05773_v
              uav0000305_00000_v
              uav0000339_00001_v
              '''
    seqs = [seq.strip() for seq in seqs_str.split()]
    gt_path = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-val/annotations-c5/"
    out_path = "/home/sdb/wangshentao/myspace/thesis/data/lu/"
    for seq in seqs:
        if not os.path.exists(os.path.join(out_path, seq)):
            os.makedirs(os.path.join(out_path, seq))
        gt_data = load_mot(os.path.join(gt_path, seq+".txt"))
        for frame in range(1, len(gt_data)):
            bboxs_pre = [i[:4] for i in gt_data[frame-1]]
            bboxs = [i[:4] for i in gt_data[frame]]
            dist = iou_distance(bboxs_pre, bboxs)
            ids_pre = [i[4] for i in gt_data[frame-1]]
            ids = [i[4] for i in gt_data[frame]]
            gt_matrix = np.zeros(dist.shape)
            for idx, id in enumerate(ids_pre):
                if id in ids:
                    gt_matrix[idx, ids.index(id)] = 1
            np.savetxt(os.path.join(out_path, seq, "cost_{:0>7d}.txt".format(frame)), dist, fmt='%f', delimiter=',')
            np.savetxt(os.path.join(out_path, seq, "gt_{:0>7d}.txt".format(frame)), gt_matrix, fmt='%d', delimiter=',')


if __name__ == "__main__":
    main()