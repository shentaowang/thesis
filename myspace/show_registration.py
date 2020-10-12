import numpy
import matplotlib.pyplot as plt
import os
import pickle
import cv2


def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular img to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def eval_affine_dict():
    """
    eval the affine dict for some sequences
    Returns:

    """
    seqs_str = """
               uav0000249_00001_v
               uav0000249_02688_v
               """
    data_root = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/testc5'
    affine_dict_path = '/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/affine/'
    seqs = [seq.strip() for seq in seqs_str.split()]
    for seq in seqs:
        with open(os.path.join(affine_dict_path, seq+'.pickle'), 'rb') as fin:
            affine_dict = pickle.load(fin)
        img_files = os.listdir(os.path.join(data_root, seq))
        img_files = sorted(img_files)
        img_files = [i for i in img_files if '.jpg' in i]
        for i in range(len(img_files)-1):
            img_file0 = img_files[i]
            img_file1 = img_files[i+1]
            affine_mat = affine_dict[img_file0]
            img0 = cv2.imread(os.path.join(data_root, seq, img_file0))
            img0, ratio0, dw0, dh0 = letterbox(img0)
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            img1 = cv2.imread(os.path.join(data_root, seq, img_file1))
            img1, ratio1, dw1, dh1 = letterbox(img1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            plt.figure(i, figsize=(16, 9))
            plt.subplot(2, 2, 1)
            plt.imshow(img0, cmap='gray')
            plt.title('img0')
            plt.subplot(2, 2, 2)
            plt.imshow(img1, cmap='gray')
            plt.title('img1')
            plt.subplot(2, 2, 3)
            img_registration = cv2.warpPerspective(img0, affine_mat, (img0.shape[1], img0.shape[0]))
            plt.imshow(img_registration, cmap='gray')
            plt.title('img0 to img1')
            plt.show()


if __name__ == '__main__':
    eval_affine_dict()
