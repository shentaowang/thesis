import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import seaborn as sns


def visdrone_data(label):
    root_dir = "/home/sdb/wangshentao/myspace/mechanical/data/visdrone2019/VisDrone2019-DET-val/"
    annotations_dir = root_dir + "annotations/"
    image_dir = root_dir + "JPEGImages/"
    class_name = ['ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle',
                  'awning-tricycle', 'bus', 'motor', 'others']

    img_bbox = []
    img_size = []

    w, h = 416, 416

    for filename in os.listdir(annotations_dir):
        fin = open(annotations_dir + filename, 'r')
        image_name = filename.split('.')[0]
        img = Image.open(image_dir + image_name + ".jpg")
        bboxs = []
        org_width, org_height = img.size[0], img.size[1]
        bbox_size = [w, h]

        for line in fin.readlines():
            line = line.split(',')
            if int(line[5]) != label:
                continue
            l = int(line[0])
            r = (l + int(line[2])-1)
            t = int(line[1])
            d = (t + int(line[3])-1)

            l = l * w // org_width
            r = r * w // org_width
            t = t * h // org_height
            d = d * h // org_height
            s = (r-l)*(d-t)
            assert s >= 0
            bboxs.append([l, r, t, d, s])
        if len(bboxs) > 0:
            img_bbox.append(bboxs)
            img_size.append(bbox_size)

    return img_bbox, img_size


def plot_bbox_2d(img_bboxs):
    bbox_h = [i[1] - i[0] for i in img_bboxs]
    bbox_w = [i[3] - i[2] for i in img_bboxs]
    h_min, h_max = np.min(bbox_h), np.max(bbox_h)
    w_min, w_max = np.min(bbox_w), np.max(bbox_w)
    data = np.zeros((h_max - h_min + 1, w_max - w_min + 1))
    for i in range(len(bbox_h)):
        data[bbox_h[i], bbox_w[i]] += 1
    f, ax = plt.subplots(figsize=(6, 4))
    cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
    sns.heatmap(data, linewidths=0.05, ax=ax, cmap=cmap)
    ax.set_xlabel('width')
    ax.set_ylabel('height')
    plt.show()


def plot_bbox_2d_clip(img_bboxs):
    bbox_h = [i[1] - i[0] for i in img_bboxs]
    bbox_w = [i[3] - i[2] for i in img_bboxs]
    h_min, h_max = 0, 90
    w_min, w_max = 0, 160
    data = np.zeros((h_max - h_min + 1, w_max - w_min + 1))
    for i in range(len(bbox_h)):
        bbox_h[i] = np.clip(bbox_h[i], 0, 90)
        bbox_w[i] = np.clip(bbox_w[i], 0, 160)
        data[bbox_h[i], bbox_w[i]] += 1
    f, ax = plt.subplots(figsize=(6, 4))
    cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
    sns.heatmap(data, linewidths=0.05, ax=ax, cmap=cmap, xticklabels=10, yticklabels=10)
    ax.set_xlabel('width')
    ax.set_ylabel('height')
    plt.show()


if __name__ == "__main__":
    img_bboxs = []
    for i in range(1, 11):
        bbox, _ = visdrone_data(i)
        img_bboxs += bbox
    img_bboxs_flatten = []
    for i in img_bboxs:
        img_bboxs_flatten += i
    plot_bbox_2d_clip(img_bboxs_flatten)
