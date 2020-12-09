import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from collections import defaultdict


def main():
    seqs_str = '''
              uav0000182_00000_v
              uav0000268_05773_v
              uav0000305_00000_v
              uav0000339_00001_v
              '''
    seqs = [seq.strip() for seq in seqs_str.split()]
    output_dir = "/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/outputs/"
    result_dir = "/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/results/"
    root_dir = "/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/valc1"
    output1 = "fcos_dlav3_extend_refine_1108_nms0.4_conf0.6"
    output2 = "fcos_dlav3_1108_2_nms0.4_conf0.6_centerness"
    output3 = "fcos_dlav3_extend_refine_1108_nms0.4_conf0.6"
    output_path1 = os.path.join(output_dir, output1)
    output_path2 = os.path.join(output_dir, output2)
    output_path3 = os.path.join(output_dir, output3)
    for seq in seqs:
        # deal result
        fin1 = open(os.path.join(result_dir, output1, seq+"_det.txt"), 'r')
        result1 = defaultdict(list)
        for line in fin1.readlines():
            line = line.split(',')
            if int(line[-1]) == 0:
                result1[line[0]].append(line[2:6])

        fin3 = open(os.path.join(result_dir, output3, seq+"_det.txt"), 'r')
        result3 = defaultdict(list)
        for line in fin3.readlines():
            line = line.split(',')
            if int(line[-1]) == 0:
                result3[line[0]].append(line[2:6])

        image_files = os.listdir(os.path.join(output_path1, seq))
        image_files = [i for i in image_files if ".jpg" in i]
        image_files = sorted(image_files)
        for idx, image_file in enumerate(image_files):
            if idx < 45:
                continue
            frame_id = int(image_file.split('.')[0])
            tracking_detections1 = result1[str(frame_id+1)]
            tracking_detections3 = result3[str(frame_id+1)]

            image1 = cv2.imread(os.path.join(output_path1, seq, image_file))
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image2 = cv2.imread(os.path.join(output_path2, seq, image_file))
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            image_orig = cv2.imread(os.path.join(root_dir, seq, "{:0>7d}.jpg".format(frame_id+1)))
            image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(16, 9))
            plt.subplot(2, 2, 1)
            plt.imshow(image1)
            plt.subplot(2, 2, 2)
            plt.imshow(image2)
            plt.subplot(2, 2, 3)
            image3 = image_orig.copy()
            for detection in tracking_detections1:
                detection = [int(float(i)) for i in detection]
                x, y, w, h = detection
                cv2.rectangle(image3, (x, y), (x+w, y+h), (0, 255, 0), 2)
            plt.imshow(image3)
            plt.subplot(2, 2, 4)
            image4 = image_orig.copy()
            for detection in tracking_detections3:
                detection = [int(float(i)) for i in detection]
                x, y, w, h = detection
                cv2.rectangle(image4, (x, y), (x+w, y+h), (0, 255, 0), 2)
            plt.imshow(image4)
            plt.show()


if __name__ == "__main__":
    main()
