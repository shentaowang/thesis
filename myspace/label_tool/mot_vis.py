import cv2
import os

def visual_bbox():
    labels_path = '/home/sdb/wangshentao/myspace/mechanical/data/visdrone_2019_mot/labels_with_ids/val/'
    images_path = '/home/sdb/wangshentao/myspace/mechanical/data/visdrone_2019_mot/images/val/'
    h, w = 806 , 1088
    for seq in os.listdir(labels_path):
        image_files = os.listdir(os.path.join(images_path, seq))
        image_files = sorted(image_files)
        for image_path in image_files:
            if '.jpg' not in image_path:
                continue
            image = cv2.imread(os.path.join(images_path, seq, image_path))
            image = cv2.resize(image, (w, h))
            label_path = image_path.replace('.jpg', '.txt')
            with open(os.path.join(labels_path, seq, label_path),'r') as fin:
                for line in fin.readlines():
                    line = line.split()
                    bbox = [line[2], line[3], line[4], line[5]]
                    bbox = [float(i) for i in bbox]
                    bbox[0] = int(bbox[0]*w)
                    bbox[1] = int(bbox[1]*h)
                    bbox[2] = int(bbox[2]*w)//2
                    bbox[3] = int(bbox[3]*h)//2
                    cv2.rectangle(image, (bbox[0]-bbox[2], bbox[1]-bbox[3]), (bbox[0]+bbox[2], bbox[1]+bbox[3]),(0, 255, 0), 2)
            cv2.imshow('dets', image)
            cv2.waitKey(0)


def visual_bbox_abs():
    labels_path = '/home/sdb/wangshentao/myspace/mechanical/data/visdrone_2019_mot/labels_with_ids/valv2/'
    images_path = '/home/sdb/wangshentao/myspace/mechanical/data/visdrone_2019_mot/images/val/'
    for seq in os.listdir(labels_path):
        image_files = os.listdir(os.path.join(images_path, seq))
        image_files = sorted(image_files)
        for image_path in image_files:
            if '.jpg' not in image_path:
                continue
            image = cv2.imread(os.path.join(images_path, seq, image_path))
            label_path = image_path.replace('.jpg', '.txt')
            with open(os.path.join(labels_path, seq, label_path),'r') as fin:
                for line in fin.readlines():
                    line = line.split()
                    bbox = [line[2], line[3], line[4], line[5]]
                    bbox = [int(float(i)) for i in bbox]
                    cv2.rectangle(image, (bbox[0]-bbox[2]//2, bbox[1]-bbox[3]//2), (bbox[0]+bbox[2]//2, bbox[1]+bbox[3]//2),(0, 255, 0), 2)
            cv2.imshow('dets', image)
            cv2.waitKey(0)

if __name__ == '__main__':
    visual_bbox()



