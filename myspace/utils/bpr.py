import os
from PIL import Image
import numpy as np


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
        bboxs=[]
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

        img_bbox.append(bboxs)
        img_size.append(bbox_size)

    return img_bbox, img_size


def eval_bpr(img_bbox, img_size, ratio):
    recall_ratio = []
    for i in range(len(img_bbox)):
        if len(img_bbox[i]) == 0:
            continue
        width = img_size[i][0]//ratio
        height = img_size[i][1]//ratio
        visited = np.zeros((len(img_bbox[i])))
        hmap = np.zeros((height, width))
        for m in range(len(img_bbox[i])):
            bbox = img_bbox[i][m]
            bbox = [i//ratio for i in bbox]
            if hmap[bbox[2]:bbox[3], bbox[0]:bbox[1]].sum()<((bbox[3]-bbox[2])*(bbox[1]-bbox[0])):
                hmap[bbox[2]:bbox[3], bbox[0]:bbox[1]]=1
                visited[m] = 1
        # for k in range(0, width+1):
        #     for l in range(0, height-1):
        #         for m in range(len(img_bbox[i])):
        #             bbox = img_bbox[i][m]
        #             bbox = [i//ratio for i in bbox]
        #             if visited[m]==0 and bbox[0]<=k and bbox[1]>=k and bbox[2]<=l and bbox[3]>=l:
        #                 visited[m]=1
        #                 break
        recall_ratio.append(np.mean(visited))
    print("ratio:{}, bpr:{}".format(ratio, np.mean(recall_ratio)))


def main():
    for label in range(1, 11):
        img_bbox, img_size = visdrone_data(label)
        sorted_bbox = []
        for bbox in img_bbox:
            bbox = sorted(bbox, key=lambda x: x[4])
            sorted_bbox.append(bbox)
        eval_bpr(img_bbox, img_size, 1)
        # eval_bpr(sorted_bbox, img_size, 4)


if __name__ == "__main__":
    main()
