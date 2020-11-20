import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import seaborn as sns

class_name = ['ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle',
              'awning-tricycle', 'bus', 'motor', 'others']

track_name_c5 = ['pedestrian', 'car', 'van', 'truck', 'bus']
class_id = [1, 4, 5, 6, 9]


def visdrone_area(root_dir):
    annotations_dir = root_dir + "annotations/"
    areas = []
    for filename in os.listdir(annotations_dir):
        fin = open(annotations_dir + filename, 'r')
        for line in fin.readlines():
            line = line.split(',')
            if int(line[5]) not in class_id:
                continue
            l = int(line[0])
            r = (l + int(line[2])-1)
            t = int(line[1])
            b = (t + int(line[3])-1)
            s = (r-l)*(b-t)
            if s > 0:
                areas.append(s)
    return areas


if __name__ == "__main__":
    root_dir = '/home/sdb/wangshentao/myspace/thesis/data/visdrone2019/VisDrone2019-DET-train/'
    train_area = visdrone_area(root_dir)
    train_area = sorted(train_area)
    train_area = train_area[0: int(len(train_area)*0.9)]
    root_dir = "/home/sdb/wangshentao/myspace/thesis/data/visdrone2019/VisDrone2019-DET-val/"
    val_area = visdrone_area(root_dir)
    val_area = sorted(val_area)
    val_area = val_area[0: int(len(val_area) * 0.9)]
    root_dir = "/home/sdb/wangshentao/myspace/thesis/data/visdrone2019/VisDrone2019-DET-dev/"
    dev_area = visdrone_area(root_dir)
    dev_area = sorted(dev_area)
    dev_area = dev_area[0: int(len(dev_area) * 0.9)]
    data = np.array([train_area, val_area, dev_area])
    f, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=20, histtype='bar', label=["train", "val", "test-dev"])
    ax.legend(prop={'size': 10})
    plt.xlabel("area")
    plt.ylabel("num")
    f.tight_layout()
    plt.show()


