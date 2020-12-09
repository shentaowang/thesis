import numpy as np
import cv2
import os
import time


def load_mask(annotation_root, seq, idx, width, height):
    file = "{:0>7d}.txt".format(idx)
    file = os.path.join(annotation_root, seq, file)
    fin = open(file, 'r')
    bboxs = []
    for line in fin.readlines():
        line = line.strip('\n')
        line = line.split()[2:]
        line = [float(i) for i in line]
        line[0], line[2] = int(line[0] * width), int(line[2] * width)
        line[1], line[3] = int(line[1] * height), int(line[3] * height)
        bboxs.append([line[0] - line[2]//2, line[1] - line[3]//2, line[0] + line[2]//2, line[1] + line[3]//2])
    mask = np.ones((height, width)) * 255
    for bbox in bboxs:
        mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0
    return mask


def orb_detect(image, mask):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    ratio = np.min([640 / width, 480 / height])
    width_resize, height_resize = int(ratio * width), int(ratio * height)
    image = cv2.resize(image, (width_resize, height_resize))
    mask = cv2.resize(mask, (width_resize, height_resize))
    orb = cv2.ORB_create()
    kps = orb.detect(image, None)
    foreground, background = 0, 0
    for kp in kps:
        if mask[int(kp.pt[1]), int(kp.pt[0])] == 0:
            foreground += 1
        else:
            background += 1
    return foreground, background


def surf_detect(image, mask):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    ratio = np.min([640/width, 480/height])
    width_resize, height_resize = int(ratio*width), int(ratio*height)
    image = cv2.resize(image, (width_resize, height_resize))
    mask = cv2.resize(mask, (width_resize, height_resize))
    surf = cv2.xfeatures2d.SURF_create()
    kps = surf.detect(image, None)
    foreground, background = 0, 0
    for kp in kps:
        if mask[int(kp.pt[1]), int(kp.pt[0])] == 0:
            foreground += 1
        else:
            background += 1
    return foreground, background


def harris_detect(image, mask):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    ratio = np.min([640 / width, 480 / height])
    width_resize, height_resize = int(ratio * width), int(ratio * height)
    image = cv2.resize(image, (width_resize, height_resize))
    mask = cv2.resize(mask, (width_resize, height_resize))
    image = np.float32(image)
    threshold = cv2.dilate(image, None).max()
    block_size, aperture_szie, k = 2, 3, 0.04
    pts_norm = cv2.cornerHarris(image, block_size, aperture_szie, k)
    foreground, background = 0, 0
    for i in range(pts_norm.shape[0]):
        for j in range(pts_norm.shape[1]):
            if pts_norm[i][j] > threshold:
                if mask[i][j] == 0:
                    foreground += 1
                else:
                    background += 1
    return foreground, background


def harris_detect_v2(image, mask):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    ratio = np.min([640 / width, 480 / height])
    width_resize, height_resize = int(ratio * width), int(ratio * height)
    image = cv2.resize(image, (width_resize, height_resize))
    mask = cv2.resize(mask, (width_resize, height_resize))
    threshold = 120
    block_size, aperture_size, k = 2, 3, 0.04
    dst = cv2.cornerHarris(image, block_size, aperture_size, k)
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    foreground, background = 0, 0
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if dst_norm[i][j] > threshold:
                if mask[i][j] == 0:
                    foreground += 1
                else:
                    background += 1
    return foreground, background


def shi_tomasi_detect(image, mask):
    max_corners, quality_level, min_distance = 2000, 0.01, 10
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    ratio = np.min([640 / width, 480 / height])
    width_resize, height_resize = int(ratio * width), int(ratio * height)
    image = cv2.resize(image, (width_resize, height_resize))
    mask = cv2.resize(mask, (width_resize, height_resize))
    pts = cv2.goodFeaturesToTrack(image, max_corners, quality_level, min_distance)
    corners = np.int0(pts)
    foreground, background = 0, 0
    for i in corners:
        x, y = i.ravel()
        if mask[y, x] == 0:
            foreground += 1
        else:
            background += 1
    return foreground, background


def sift_detect(image, mask):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    ratio = np.min([640/width, 480/height])
    width_resize, height_resize = int(ratio*width), int(ratio*height)
    image = cv2.resize(image, (width_resize, height_resize))
    mask = cv2.resize(mask, (width_resize, height_resize))
    sift = cv2.xfeatures2d.SIFT_create()
    kps = sift.detect(image, None)
    foreground, background = 0, 0
    for kp in kps:
        if mask[int(kp.pt[1]), int(kp.pt[0])] == 0:
            foreground += 1
        else:
            background += 1
    return foreground, background


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
    foreground_acc, background_acc = 0, 0
    image_cnt_acc, timer_acc = 0, 0
    for seq in seqs:
        image_files = os.listdir(os.path.join(data_root, seq))
        image_files = sorted(image_files)
        print(seq)
        image_files = [i for i in image_files if 'jpg' in i]
        for i in range(len(image_files)):
            # print(i)
            image_cnt_acc += 1
            image = cv2.imread(os.path.join(data_root, seq, image_files[i]))
            mask = load_mask(annotation_root, seq, i+1, image.shape[1], image.shape[0])
            start_time = time.time()
            foreground, background = surf_detect(image, mask)
            timer_acc += (time.time() - start_time)
            foreground_acc += foreground
            background_acc += background
    print("foreground: {}, background: {}, ratio: {}".
          format(foreground_acc, background_acc, foreground_acc / (foreground_acc + background_acc)))
    print("avg points: {}".format((foreground_acc + background_acc) / image_cnt_acc))
    print("use time: {}".format(timer_acc / image_cnt_acc))


if __name__ == "__main__":
    main()
