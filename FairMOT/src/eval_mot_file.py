from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

from tracking_utils.evaluation import Evaluator
from lib.tracking_utils.io import read_results
import os
import motmetrics as mm
import cv2
import matplotlib.pyplot as plt


def main():
    seqs = """
            uav0000086_00000_v
            uav0000117_02622_v
            uav0000137_00458_v
            uav0000182_00000_v
            uav0000268_05773_v
            uav0000305_00000_v
            uav0000339_00001_v
    """
    data_type = 'mot'
    seqs = [seq.strip() for seq in seqs.split()]
    data_root = "/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/valc5/"
    out_path = "/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/results/kcf_tracker_1204"
    accs = []
    for seq in seqs:
        evaluator = Evaluator(data_root, seq, data_type)
        result_filename = os.path.join(out_path, seq+'.txt')
        accs.append(evaluator.eval_file(result_filename))
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)


def cp_result_txt():
    val_seqs_str = '''
                      uav0000086_00000_v
                      uav0000117_02622_v
                      uav0000137_00458_v
                      uav0000182_00000_v
                      uav0000268_05773_v
                      uav0000305_00000_v
                      uav0000339_00001_v
                    '''
    seq = "uav0000182_00000_v"
    track_file1 = "/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/results/" \
                  "fcos_dlav3_1108_2_nms0.4_conf0.6_centerness"
    track_file1 = os.path.join(track_file1, seq+'.txt')
    track_file2 = "/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/results/" \
                  "fcos_dlav3_extend_refine_v2_1108_nms0.4_conf0.6"
    track_file2 = os.path.join(track_file2, seq + '.txt')
    track_file3 = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-val/annotations-c5/"
    track_file3 = os.path.join(track_file3, seq + '.txt')
    image_file = "/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/val"
    image_file = os.path.join(image_file, seq)
    track_data1 = read_results(track_file1, 'mot', False, False)
    track_data2 = read_results(track_file2, 'mot', False, False)
    track_data3 = read_results(track_file3, 'mot', False, False)
    for i in range(1, 100):
        track_bbox1 = track_data1[i]
        track_bbox2 = track_data2[i]
        track_bbox3 = track_data3[i]
        print(len(track_bbox1), len(track_bbox2))
        image = cv2.imread(os.path.join(image_file, "{:0>7d}.jpg".format(i)))
        for bbox in track_bbox1:
            bbox = [int(i) for i in bbox[0]]
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
        for bbox in track_bbox2:
            bbox = [int(i) for i in bbox[0]]
            cv2.rectangle(image, (bbox[0]+2, bbox[1]+2), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
        # for bbox in track_bbox3:
        #     bbox = [int(i) for i in bbox[0]]
        #     cv2.rectangle(image, (bbox[0]+4, bbox[1]+4), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)
        plt.figure(figsize=(16, 9))
        plt.imshow(image)
        plt.title(i)
        plt.show()


if __name__ == "__main__":
    main()
    # cp_result_txt()
