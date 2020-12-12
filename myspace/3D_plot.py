import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import os


def main():
    depth_dir = '/home/sdb/wangshentao/myspace/thesis/data/2020-10-20-19-50-01/depth_image/'
    file_name = '1603194728.506196.pickle'
    fin = open(os.path.join(depth_dir, file_name), 'rb')
    data = pickle.load(fin, encoding='latin1')
    depth_image = cv2.applyColorMap(cv2.convertScaleAbs(data, alpha=0.03), cv2.COLORMAP_JET)
    cv2.imwrite('data/depth.png', depth_image)


if __name__ == "__main__":
    main()
