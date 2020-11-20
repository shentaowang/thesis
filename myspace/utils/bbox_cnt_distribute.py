import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import seaborn as sns

class_name = ['ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle',
              'awning-tricycle', 'bus', 'motor', 'others']

track_name_c5 = ['pedestrian', 'car', 'van', 'truck', 'bus']
class_id = [1, 4, 5, 6, 9]
class_dict = {1: 0, 4: 1, 5: 2, 6: 3, 9: 4}


def visdrone_cnt(root_dir):
    annotations_dir = root_dir + "annotations/"
    cnt_array = []
    for filename in os.listdir(annotations_dir):
        fin = open(annotations_dir + filename, 'r')
        for line in fin.readlines():
            line = line.split(',')
            label = int(line[5])
            if label not in class_id:
                continue
            cnt_array.append(class_dict[label])
    return cnt_array


if __name__ == "__main__":
    root_dir = '/home/sdb/wangshentao/myspace/thesis/data/visdrone2019/VisDrone2019-DET-train/'
    train_cnt = visdrone_cnt(root_dir)
    root_dir = "/home/sdb/wangshentao/myspace/thesis/data/visdrone2019/VisDrone2019-DET-val/"
    val_cnt = visdrone_cnt(root_dir)
    root_dir = "/home/sdb/wangshentao/myspace/thesis/data/visdrone2019/VisDrone2019-DET-dev/"
    dev_cnt = visdrone_cnt(root_dir)
    data = np.array([train_cnt, val_cnt, dev_cnt])
    f, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=np.arange(6)-0.5, histtype='bar', label=["train", "val", "test-dev"])
    ax.legend(prop={'size': 10})
    plt.xticks(range(len(track_name_c5)), track_name_c5)
    plt.xlabel("class")
    plt.ylabel("num")
    f.tight_layout()
    plt.show()


