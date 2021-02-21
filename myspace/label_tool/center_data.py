import numpy as np
import os
import shutil


def prepare_image():
    path = '/home/sdb/wangshentao/myspace/thesis/data/center_data/val/'
    for seq in os.listdir(path):
        seq_path = os.path.join(path, seq)
        if not os.path.exists(os.path.join(seq_path, 'img1')):
            os.makedirs(os.path.join(seq_path, 'img1'))
        img_files = os.listdir(seq_path)
        for img_path in img_files:
            if '.jpg' not in img_path:
                continue
            shutil.move(os.path.join(seq_path, img_path), os.path.join(seq_path, 'img1', img_path))


def prepare_gt():
    path = '/home/sdb/wangshentao/myspace/thesis/data/center_data/test/'
    annotation_path = '/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/annotations'
    for seq in os.listdir(path):
        seq_path = os.path.join(path, seq)
        if not os.path.exists(os.path.join(seq_path, 'gt')):
            os.makedirs(os.path.join(seq_path, 'gt'))
        annotation_file = seq + '.txt'
        shutil.copy(os.path.join(annotation_path, annotation_file), os.path.join(seq_path, 'gt', 'gt.txt'))


def prepare_det():
    path = '/home/sdb/wangshentao/myspace/thesis/data/center_data/train/'
    det_path = '/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-train/center-detections-1204'
    for seq in os.listdir(path):
        seq_path = os.path.join(path, seq)
        if not os.path.exists(os.path.join(seq_path, 'det')):
            os.makedirs(os.path.join(seq_path, 'det'))
        det_file = seq + '.txt'
        shutil.copy(os.path.join(det_path, det_file), os.path.join(seq_path, 'det', 'det.txt'))




if __name__ == "__main__":
    # prepare_image()
    # prepare_gt()
    # prepare_det()
