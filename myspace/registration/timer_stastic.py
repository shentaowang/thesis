import numpy as np
import cv2
import os
import time


def fast_surf(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    height, width = image1.shape[0], image1.shape[1]
    ratio = np.min([480/height, 640/width])
    height_resize, width_resize = int(height*ratio), int(width*ratio)
    image1 = cv2.resize(image1, (width_resize, height_resize))
    image2 = cv2.resize(image2, (width_resize, height_resize))
    surf = cv2.xfeatures2d.SURF_create()
    orb = cv2.ORB_create()
    kp1 = surf.detect(image1, None)
    kp1, des1 = orb.compute(image1, kp1)
    kp2 = surf.detect(image1, None)
    kp2, des2 = orb.compute(image2, kp2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches
    M = None
    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, maxIters=200)
    return M


def main():
    data_root = "/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/val/"
    annotation_root = "/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/labels_with_ids/val/"
    val_seqs_str = '''
                      uav0000086_00000_v
                      uav0000117_02622_v
                      uav0000137_00458_v
                      uav0000182_00000_v
                      uav0000268_05773_v
                      uav0000305_00000_v
                      uav0000339_00001_v
                    '''
    seqs = [seq.strip() for seq in val_seqs_str.split()]
    image_cnt_acc, timer_acc = 0, 0
    for seq in seqs:
        image_files = os.listdir(os.path.join(data_root, seq))
        image_files = sorted(image_files)
        print(seq)
        image_files = [i for i in image_files if 'jpg' in i]
        for i in range(100):
            image_cnt_acc += 1
            image1 = cv2.imread(os.path.join(data_root, seq, image_files[i]))
            image2 = cv2.imread(os.path.join(data_root, seq, image_files[i+1]))
            start_time = time.time()
            fast_surf(image1, image2)
            timer_acc += (time.time() - start_time)
    print("avg time: {}".format(timer_acc / image_cnt_acc))


if __name__ == '__main__':
    main()
