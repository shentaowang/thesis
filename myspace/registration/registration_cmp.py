import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

std_width = 640
std_height = 480


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


def surf_registration(image1, image2, mask1=None, mask2=None):
    # use surf
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(image1, (std_width, std_height))
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image2 = cv2.resize(image2, (std_width, std_height))
    surf = cv2.xfeatures2d.SURF_create()
    kp1, des1 = surf.detectAndCompute(image1, mask=mask1)
    kp2, des2 = surf.detectAndCompute(image2, mask=mask2)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, maxIters=200)
    return M


def orb_registration(image1, image2, mask1=None, mask2=None):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(image1, (std_width, std_height))
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image2 = cv2.resize(image2, (std_width, std_height))
    mask1 = cv2.resize(mask1, (std_width, std_height))
    mask2 = cv2.resize(mask2, (std_width, std_height))
    # use orb
    surf = cv2.xfeatures2d.SURF_create()
    orb = cv2.ORB_create()
    fast = cv2.FastFeatureDetector_create(threshold=40, nonmaxSuppression=True,
                                          type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    # orb.setNLevels(1)
    kp1, des1 = orb.detectAndCompute(image1, None)
    # kp1 = surf.detect(image1, None)
    # kp1, des1 = orb.compute(image1, kp1)
    kp2, des2 = orb.detectAndCompute(image2, None)
    # kp2 = surf.detect(image2, None)
    # kp2, des2 = orb.compute(image2, kp2)
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(des1, des2)
    # matches = sorted(matches, key=lambda x: x.distance)
    # good = matches

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_leval=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    #
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1, des2, k=2)
    # good = []
    # for m, n in matches:
    #     if m.distance < 0.7 * n.distance:
    #         good.append(m)

    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        filter_src_pts = []
        filter_dst_pts = []
        for p1, p2 in zip(src_pts, dst_pts):
            if np.abs(p1[0, 0] - p2[0, 0]) + np.abs(p1[0, 1] - p2[0, 1]) < 100:
                filter_src_pts.append(p1)
                filter_dst_pts.append(p2)
        filter_src_pts = np.array(filter_src_pts)
        filter_dst_pts = np.array(filter_dst_pts)
        M, mask = cv2.findHomography(filter_src_pts, filter_dst_pts, cv2.RANSAC, 5.0, maxIters=200)
    return M, filter_src_pts, filter_dst_pts


def diff_m(M1, M2):
    x = np.linspace(0, 639, 640)
    y = np.linspace(0, 479, 480)
    [x, y] = np.meshgrid(x, y)
    x, y = x.reshape(1, -1), y.reshape(1, -1)
    pts = np.concatenate([x, y, np.ones(x.shape)], axis=0)
    surf_trans_pts = np.dot(M1, pts)
    surf_trans_pts = surf_trans_pts / surf_trans_pts[2, :]
    orb_trans_pts = np.dot(M2, pts)
    orb_trans_pts = orb_trans_pts / orb_trans_pts[2, :]
    diff = surf_trans_pts - orb_trans_pts
    diff = diff * diff
    return np.max(diff)


def show_surf_match(image1, image2):
    image1_resize = cv2.resize(image1, (std_width, std_height))
    image2_resize = cv2.resize(image2, (std_width, std_height))
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(image1, (std_width, std_height))
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image2 = cv2.resize(image2, (std_width, std_height))
    surf = cv2.xfeatures2d.SURF_create()
    kp1, des1 = surf.detectAndCompute(image1, mask=None)
    kp2, des2 = surf.detectAndCompute(image2, mask=None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    plt.figure()
    image_two = np.concatenate([image1_resize, image2_resize], axis=1)
    image_two = cv2.cvtColor(image_two, cv2.COLOR_BGR2RGB)
    dst_pts = dst_pts + np.array([std_width, 0])
    for p1, p2 in zip(src_pts, dst_pts):
        cv2.line(image_two, (int(p1[0, 0]), int(p1[0, 1])), (int(p2[0, 0]), int(p2[0, 1])), (0, 255, 0))
    plt.imshow(image_two)
    plt.title('surf')
    plt.show()


def show_orb_match(image1, image2):
    image1_resize = cv2.resize(image1, (std_width, std_height))
    image2_resize = cv2.resize(image2, (std_width, std_height))
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(image1, (std_width, std_height))
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image2 = cv2.resize(image2, (std_width, std_height))
    orb = cv2.ORB_create()
    # orb.setNLevels(1)
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(des1, des2)
    # matches = sorted(matches, key=lambda x: x.distance)
    # good = matches

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_leval=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    #
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1, des2, k=2)
    # good = []
    # for m, n in matches:
    #     if m.distance < 0.7 * n.distance:
    #         good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    plt.figure()
    image_two = np.concatenate([image1_resize, image2_resize], axis=1)
    image_two = cv2.cvtColor(image_two, cv2.COLOR_BGR2RGB)
    dst_pts = dst_pts + np.array([std_width, 0])
    for p1, p2 in zip(src_pts, dst_pts):
        cv2.line(image_two, (int(p1[0, 0]), int(p1[0, 1])), (int(p2[0, 0]), int(p2[0, 1])), (0, 255, 0))
    plt.imshow(image_two)
    plt.axis('off')
    plt.title('orb')
    plt.show()


def show_fast_surf(image1, image2):
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image1_gray = cv2.resize(image1_gray, (std_width, std_height))
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.resize(image2_gray, (std_width, std_height))
    orb = cv2.ORB_create()
    surf = cv2.xfeatures2d.SURF_create()
    kp1 = surf.detect(image1_gray, None)
    kp1, des1 = orb.compute(image1_gray, kp1)
    kp2 = surf.detect(image2_gray, None)
    kp2, des2 = orb.compute(image2_gray, kp2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches
    M = np.eye(3, 3)
    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, maxIters=200)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.grid()
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.grid()
    plt.subplot(1, 3, 3)
    registration_image = cv2.warpPerspective(image1, M, (image2.shape[1], image2.shape[0]))
    plt.imshow(cv2.cvtColor(registration_image, cv2.COLOR_BGR2RGB))
    plt.grid()
    plt.show()


def main():
    # width, height = 576, 320
    data_root = "/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/val/"
    annotation_root = "/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/labels_with_ids/val/"
    val_seqs_str = '''

                      uav0000137_00458_v
                      uav0000182_00000_v
                      uav0000268_05773_v
                      uav0000305_00000_v
                      uav0000339_00001_v
                    '''
    seqs = [seq.strip() for seq in val_seqs_str.split()]
    for seq in seqs:
        image_files = os.listdir(os.path.join(data_root, seq))
        image_files = sorted(image_files)
        print(seq)
        image_files = [i for i in image_files if 'jpg' in i]
        for i in range(5, len(image_files)):
            print(i)
            image1 = cv2.imread(os.path.join(data_root, seq, image_files[i-5]))
            image2 = cv2.imread(os.path.join(data_root, seq, image_files[i]))
            mask1 = load_mask(annotation_root, seq, i-4, image1.shape[1], image1.shape[0])
            mask2 = load_mask(annotation_root, seq, i+1, image1.shape[1], image1.shape[0])
            # plt.figure()
            # plt.subplot(1, 2, 1)
            # plt.imshow(image1)
            # plt.subplot(1, 2, 2)
            # plt.imshow(mask1)
            # plt.show()

            surf_m = surf_registration(image1, image2)
            orb_m, src_pts, dst_pts = orb_registration(image1, image2, mask1, mask2)

            # show_fast_surf(image1, image2)

            if diff_m(surf_m, orb_m) > 900:
                show_surf_match(image1, image2)
                show_orb_match(image1, image2)

                # show the orb registration result
                plt.figure(0)
                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
                plt.grid()
                plt.subplot(1, 2, 2)
                registration_image = cv2.warpPerspective(image1, orb_m, (image2.shape[1], image2.shape[0]))
                plt.imshow(cv2.cvtColor(registration_image, cv2.COLOR_BGR2RGB))
                plt.grid()

                # show the orb and surf registration result
                plt.figure(1)
                plt.subplot(2, 2, 1)
                plt.imshow(image2, cmap='gray')
                plt.grid()
                plt.subplot(2, 2, 2)
                registration_image = cv2.warpPerspective(image1, surf_m, (image2.shape[1], image2.shape[0]))
                plt.imshow(registration_image, cmap='gray')
                plt.grid()
                plt.title("surf")
                plt.subplot(2, 2, 3)
                plt.imshow(image2, cmap='gray')
                plt.grid()
                plt.subplot(2, 2, 4)
                registration_image = cv2.warpPerspective(image1, orb_m, (image2.shape[1], image2.shape[0]))
                plt.imshow(registration_image, cmap='gray')
                plt.grid()
                plt.title("orb")

                # show the orb line
                plt.figure(2)
                image1_resize = cv2.resize(image1, (std_width, std_height))
                image2_resize = cv2.resize(image2, (std_width, std_height))
                image_two = np.concatenate([image1_resize, image2_resize], axis=1)
                dst_pts = dst_pts + np.array([std_width, 0])
                for p1, p2 in zip(src_pts, dst_pts):
                    cv2.line(image_two, (int(p1[0, 0]), int(p1[0, 1])), (int(p2[0, 0]), int(p2[0, 1])), (0, 255, 0))
                plt.imshow(image_two)

                # show the orb point
                plt.figure(3)
                orb = cv2.ORB_create()
                kp1 = orb.detect(image1_resize, None)
                plt.subplot(1, 2, 1)
                image1_show = image1_resize.copy()
                image1_show = cv2.cvtColor(image1_show, cv2.COLOR_BGR2RGB)
                for pt in kp1:
                    cv2.circle(image1_show, (int(pt.pt[0]), int(pt.pt[1])), 4, (0, 0, 255), 1)
                plt.imshow(image1_show)
                plt.axis('off')
                plt.subplot(1, 2, 2)
                image2_show = image2_resize.copy()
                image2_show = cv2.cvtColor(image2_show, cv2.COLOR_BGR2RGB)
                kp2 = orb.detect(image2_resize, None)
                for pt in kp2:
                    cv2.circle(image2_show, (int(pt.pt[0]), int(pt.pt[1])), 4, (0, 0, 255), 1)
                plt.imshow(image2_show)
                plt.axis('off')

                # show the fast point, more dense
                plt.figure(4)
                fast = cv2.FastFeatureDetector_create(threshold=40, nonmaxSuppression=True,
                                                      type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
                kp1 = fast.detect(image1_resize, None)
                plt.subplot(1, 2, 1)
                image1_show = image1_resize.copy()
                for pt in kp1:
                    cv2.circle(image1_show, (int(pt.pt[0]), int(pt.pt[1])), 4, (0, 0, 255), 1)
                plt.imshow(image1_show)
                plt.subplot(1, 2, 2)
                image2_show = image2_resize.copy()
                kp2 = fast.detect(image2_show, None)
                for pt in kp2:
                    cv2.circle(image2_show, (int(pt.pt[0]), int(pt.pt[1])), 4, (0, 0, 255), 1)
                plt.imshow(image2_show)

                # show the surf point
                plt.figure(5)
                surf = cv2.xfeatures2d.SURF_create()
                kp1 = surf.detect(image1_resize, None)
                plt.subplot(1, 2, 1)
                image1_show = image1_resize.copy()
                for pt in kp1:
                    cv2.circle(image1_show, (int(pt.pt[0]), int(pt.pt[1])), 4, (0, 0, 255), 1)
                plt.imshow(image1_show)
                plt.subplot(1, 2, 2)
                image2_show = image2_resize.copy()
                kp2 = surf.detect(image2_show, None)
                for pt in kp2:
                    cv2.circle(image2_show, (int(pt.pt[0]), int(pt.pt[1])), 4, (0, 0, 255), 1)
                plt.imshow(image2_show)
                plt.show()


if __name__ == "__main__":
    main()
