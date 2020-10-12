import os.path as osp
import os
import numpy as np
import cv2


class_name = ['ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle',
              'awning-tricycle', 'bus', 'motor', 'others']

id2name = {0: 'ignored regions', 1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van', 6: 'truck',
           7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor', 11: 'others'}

track_name_c5 = ['pedestrian', 'car', 'van',  'truck', 'bus']

track_name2id_c5 = {'pedestrian': 0, 'car': 1, 'van': 2, 'truck': 3, 'bus': 4}

track_name_c6 = ['pedestrian', 'people' 'car', 'van',  'truck', 'bus']

track_name2id_c6 = {'pedestrian': 0, 'people': 1, 'car': 2, 'van': 3, 'truck': 4, 'bus': 5}


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def get_seq_info():
    """
    generate the ini file
    Returns:

    """
    seq_root = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/testc5/'
    seqs = [s for s in os.listdir(seq_root)]
    for seq in seqs:
        print(seq)
        height = None
        width = None
        frame_cnt = 0
        for img_path in os.listdir(osp.join(seq_root, seq)):
            if img_path.split('.')[1] == 'ini':
                continue
            img = cv2.imread(osp.join(seq_root, seq, img_path))
            frame_cnt += 1
            if height:
                assert height == img.shape[0]
                assert width == img.shape[1]
            else:
                height = img.shape[0]
                width = img.shape[1]
        with open(osp.join(seq_root, seq, 'seqinfo.ini'), 'w') as f:
            f.write("[Sequence]\n")
            f.write("name={}\n".format(seq))
            f.write("imgDir=img1\n")
            f.write("frameRate=30\n")
            f.write("seqLength={}\n".format(frame_cnt))
            f.write("imWidth={}\n".format(width))
            f.write("imHeight={}\n".format(height))
            f.write("imExt=.jpg")


def train():
    """
    generate the train data label
    Returns:

    """
    seq_root = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/train/'
    label_root = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/labels_with_ids/train'
    annotations = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/annotations/train'
    mkdirs(label_root)
    seqs = [s for s in os.listdir(seq_root)]

    tid_curr = 0
    tid_last = -1
    for seq in seqs:
        seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

        gt_txt = osp.join(annotations, seq+'.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

        seq_label_root = osp.join(label_root, seq)
        if os.path.exists(seq_label_root):
            os.removedirs(seq_label_root)
        mkdirs(seq_label_root)

        for fid, tid, x, y, w, h, mark, label, _, _ in gt:
            if int(label) == 0 or int(label) == 11:
                print('ignore')
                continue
            fid = int(fid)
            tid = int(tid)
            label = int(label)-1
            assert label >= 0
            assert label <= 10
            if not tid == tid_last:
                tid_curr += 1
                tid_last = tid
            x += w / 2
            y += h / 2
            label_fpath = osp.join(seq_label_root, '{:07d}.txt'.format(fid))
            label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                label, tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)


def trainC6():
    """
    generate the train data label, align to mot competition
    Returns:

    """
    seq_root = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/trainc6/'
    label_root = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/labels_with_ids/trainc6'
    annotations = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/annotations/train'
    mkdirs(label_root)
    seqs = [s for s in os.listdir(seq_root)]

    tid_curr = 0
    tid_last = -1
    for seq in seqs:
        seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

        gt_txt = osp.join(annotations, seq+'.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

        seq_label_root = osp.join(label_root, seq)
        if os.path.exists(seq_label_root):
            os.removedirs(seq_label_root)
        mkdirs(seq_label_root)

        for fid, tid, x, y, w, h, mark, label, _, _ in gt:
            if int(label) in [0, 3, 7, 8, 10, 11]:
                # print('ignore')
                continue
            fid = int(fid)
            tid = int(tid)
            label = track_name2id_c6[id2name[int(label)]]
            assert label >= 0
            assert label <= 10
            if not tid == tid_last:
                tid_curr += 1
                tid_last = tid
            x += w / 2
            y += h / 2
            label_fpath = osp.join(seq_label_root, '{:07d}.txt'.format(fid))
            label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                label, tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)


def val():
    """
    generate the val label
    Returns:

    """
    seq_root = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/val/'
    label_root = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/labels_with_ids/val-test'
    annotations = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/annotations/val'
    mkdirs(label_root)
    seqs = [s for s in os.listdir(seq_root)]

    tid_curr = 0
    tid_last = -1
    label_freq = {}
    for i in range(10):
        label_freq[i] = 0
    for seq in seqs:
        seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])
        gt_txt = osp.join(annotations, seq+'.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
        seq_label_root = osp.join(label_root, seq)
        if os.path.exists(seq_label_root):
            os.removedirs(seq_label_root)
        mkdirs(seq_label_root)
        for fid, tid, x, y, w, h, mark, label, _, _ in gt:
            if int(mark) == 0 or int(label) == 0 or int(label) == 11:
                print('ignore')
                continue
            fid = int(fid)
            tid = int(tid)
            label = int(label)-1
            assert label >= 0
            assert label <= 10
            if not tid == tid_last:
                tid_curr += 1
                tid_last = tid
            x += w / 2
            y += h / 2
            label_fpath = osp.join(seq_label_root, '{:07d}.txt'.format(fid))
            label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                label, tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)
            label_freq[label] += 1

    print(label_freq)


def valC6():
    """
    generate the val label, align to mot competition
    Returns:

    """
    seq_root = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/valc6/'
    label_root = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/labels_with_ids/valc6'
    annotations = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/annotations/val'
    mkdirs(label_root)
    seqs = [s for s in os.listdir(seq_root)]

    tid_curr = 0
    tid_last = -1
    label_freq = {}
    for i in range(10):
        label_freq[i] = 0
    for seq in seqs:
        seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])
        gt_txt = osp.join(annotations, seq+'.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
        seq_label_root = osp.join(label_root, seq)
        if os.path.exists(seq_label_root):
            os.removedirs(seq_label_root)
        mkdirs(seq_label_root)
        for fid, tid, x, y, w, h, mark, label, _, _ in gt:
            if int(mark) == 0 or int(label) in [0, 3, 7, 8, 10, 11]:
                # print('ignore')
                continue
            fid = int(fid)
            tid = int(tid)
            print("{}-{}".format(label, id2name[int(label)]))
            label = track_name2id_c6[id2name[int(label)]]
            assert label >= 0
            assert label <= 10
            if not tid == tid_last:
                tid_curr += 1
                tid_last = tid
            x += w / 2
            y += h / 2
            label_fpath = osp.join(seq_label_root, '{:07d}.txt'.format(fid))
            label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                label, tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)
            label_freq[label] += 1

    print(label_freq)


def testC5():
    """
    generate the test label, align to mot competition
    Returns:

    """
    seq_root = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/testc5/'
    label_root = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/labels_with_ids/testc5'
    annotations = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/annotations/test'
    mkdirs(label_root)
    seqs = [s for s in os.listdir(seq_root)]

    tid_curr = 0
    tid_last = -1
    label_freq = {}
    for i in range(10):
        label_freq[i] = 0
    for seq in seqs:
        seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])
        gt_txt = osp.join(annotations, seq+'.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
        seq_label_root = osp.join(label_root, seq)
        if os.path.exists(seq_label_root):
            os.removedirs(seq_label_root)
        mkdirs(seq_label_root)
        for fid, tid, x, y, w, h, mark, label, _, _ in gt:
            if int(mark) == 0 or int(label) in [0, 2, 3, 7, 8, 10, 11]:
                # print('ignore')
                continue
            fid = int(fid)
            tid = int(tid)
            print("{}-{}".format(label, id2name[int(label)]))
            label = track_name2id_c5[id2name[int(label)]]
            assert label >= 0
            assert label <= 10
            if not tid == tid_last:
                tid_curr += 1
                tid_last = tid
            x += w / 2
            y += h / 2
            label_fpath = osp.join(seq_label_root, '{:07d}.txt'.format(fid))
            label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                label, tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)
            label_freq[label] += 1

    print(label_freq)


def detection_train():
    """
    generate the detection train label
    Returns:

    """
    seq_root = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/train_detection/train_2019/'
    label_root = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/labels_with_ids/train_detection/train_2019'
    annotations = '/home/sdb/wangshentao/myspace/thesis/data/visdrone2019/VisDrone2019-DET-train/annotations/'
    label_freq = {}
    max_obj = 0
    for i in range(10):
        label_freq[i] = 0
    for anno_file in os.listdir(annotations):
        gt_txt = os.path.join(annotations, anno_file)
        # print(gt_txt)
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
        image_file = os.path.join(seq_root, anno_file.replace('.txt', '.jpg'))
        image = cv2.imread(image_file)
        height, width = image.shape[:2]
        max_obj = np.max([max_obj, gt.shape[0]])
        if gt.ndim == 1:
            x, y, w, h, mark, label, _, _ = gt
            if int(mark) == 0 or int(label) == 0 or int(label) == 11:
                print('ignore')
                continue
            label = int(label) - 1
            assert label >= 0
            assert label <= 10
            x += w / 2
            y += h / 2
            label_fpath = osp.join(label_root, anno_file)
            label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                label, -1, x / width, y / height, w / width, h / height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)
            label_freq[label] += 1

        else:
            for x, y, w, h, mark, label, _, _ in gt:
                if int(mark) == 0 or int(label) == 0 or int(label) == 11:
                    print('ignore')
                    continue
                label = int(label)-1
                assert label >= 0
                assert label <= 10
                x += w / 2
                y += h / 2
                label_fpath = osp.join(label_root, anno_file)
                label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    label, -1, x / width, y / height, w / width, h / height)
                with open(label_fpath, 'a') as f:
                    f.write(label_str)
                label_freq[label] += 1

    print(label_freq)
    print(max_obj)


def detection_val():
    """
    generate the detection val label
    Returns:

    """
    seq_root = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/val_detection_pure/detection/'
    label_root = '/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/labels_with_ids/val_detection_pure/' \
                 'detection'
    annotations = '/home/sdb/wangshentao/myspace/thesis/data/visdrone2019/VisDrone2019-DET-val/annotations/'
    label_freq = {}
    max_obj = 0
    for i in range(10):
        label_freq[i] = 0
    for anno_file in os.listdir(annotations):
        gt_txt = os.path.join(annotations, anno_file)
        # print(gt_txt)
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
        image_file = os.path.join(seq_root, anno_file.replace('.txt', '.jpg'))
        image = cv2.imread(image_file)
        height, width = image.shape[:2]
        max_obj = np.max([max_obj, gt.shape[0]])
        if gt.ndim == 1:
            x, y, w, h, mark, label, _, _ = gt
            if int(mark) == 0 or int(label) == 0 or int(label) == 11:
                print('ignore')
                continue
            label = int(label) - 1
            assert label >= 0
            assert label <= 10
            x += w / 2
            y += h / 2
            label_fpath = osp.join(label_root, anno_file)
            label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                label, -1, x / width, y / height, w / width, h / height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)
            label_freq[label] += 1

        else:
            for x, y, w, h, mark, label, _, _ in gt:
                if int(mark) == 0 or int(label) == 0 or int(label) == 11:
                    print('ignore')
                    continue
                label = int(label)-1
                assert label >= 0
                assert label <= 10
                x += w / 2
                y += h / 2
                label_fpath = osp.join(label_root, anno_file)
                label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    label, -1, x / width, y / height, w / width, h / height)
                with open(label_fpath, 'a') as f:
                    f.write(label_str)
                label_freq[label] += 1

    print(label_freq)
    print(max_obj)


def gen_image_list():
    """
    generate the image list
    Returns:

    """
    root_path = '/home/sdb/wangshentao/myspace/thesis/data/'
    train_image_path = 'visdrone_2019_mot/images/trainc6/'
    val_image_path = 'visdrone_2019_mot/images/valc6'
    train_label_path = 'visdrone_2019_mot/labels_with_ids/trainc6/'
    val_label_path = 'visdrone_2019_mot/labels_with_ids/valc6/'

    train_config = 'visdrone_2019_mot.trainc6'
    val_config = 'visdrone_2019_mot.valc6'

    seqs = os.listdir(os.path.join(root_path, train_label_path))
    seqs = sorted(seqs)
    with open(train_config, 'w') as fout:
        for seq in seqs:
            frames = os.listdir(os.path.join(root_path, train_label_path, seq))
            frames = sorted(frames)
            for frame in frames:
                frame = frame.replace('txt', 'jpg')
                fout.write(os.path.join(train_image_path, seq, frame))
                fout.write('\n')

    seqs = os.listdir(os.path.join(root_path, val_label_path))
    seqs = sorted(seqs)
    with open(val_config, 'w') as fout:
        for seq in seqs:
            frames = os.listdir(os.path.join(root_path, val_label_path, seq))
            frames = sorted(frames)
            for frame in frames:
                frame = frame.replace('txt', 'jpg')
                fout.write(os.path.join(val_image_path, seq, frame))
                fout.write('\n')


def gen_image_detection_list():
    """
    generate the image list
    Returns:

    """
    root_path = '/home/sdb/wangshentao/myspace/thesis/data/'
    train_image_path = 'visdrone_2019_mot/images/val_detection_pure'
    train_label_path = 'visdrone_2019_mot/labels_with_ids/val_detection_pure'

    train_config = 'visdrone_2019_mot.val_detection_pure'

    seqs = os.listdir(os.path.join(root_path, train_label_path))
    seqs = sorted(seqs)
    with open(train_config, 'w') as fout:
        for seq in seqs:
            frames = os.listdir(os.path.join(root_path, train_label_path, seq))
            frames = sorted(frames)
            for frame in frames:
                frame = frame.replace('txt', 'jpg')
                fout.write(os.path.join(train_image_path, seq, frame))
                fout.write('\n')


if __name__ == "__main__":
    # get_seq_info()
    # train()
    # val()
    # gen_image_detection_list()
    # detection_val()
    # trainC6()
    # valC6()
    # gen_image_list()
    testC5()
