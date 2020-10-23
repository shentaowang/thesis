import os.path as osp
import os
import numpy as np
import shutil
from collections import defaultdict


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


# seq_root = '/home/sdb/wangshentao/myspace/thesis/data/d435i/images/2020-10-20/'
# annotations = '/home/sdb/wangshentao/myspace/thesis/data/d435i/annotations/2020-10-20/'
# label_root = '/home/sdb/wangshentao/myspace/thesis/data/d435i/labels_with_ids/2020-10-20/'
# mkdirs(label_root)
# #seqs = [s for s in os.listdir(seq_root)]
# seqs = ['2020-10-20-19-50-01']
# seq_width = 424
# seq_height = 240
#
# for seq in seqs:
#     gt_txt = osp.join(annotations, seq + '.txt')
#     if not os.path.exists(os.path.join(label_root, seq)):
#         os.makedirs(os.path.join(label_root, seq))
#     with open(gt_txt, 'r') as fin:
#         lines = fin.readlines()
#     for line in lines:
#         line = line.split(',')
#         label_fpath = line[0].replace('.jpg', '.txt')
#         tid = int(line[1])
#         x, y, w, h = int(line[2]), int(line[3]), int(line[4]), int(line[5])
#         x = np.clip(x, 0, seq_width)
#         y = np.clip(y, 0, seq_height)
#         w = np.clip(w, 0, seq_width-x)
#         h = np.clip(h, 0, seq_height-y)
#         x += w/2
#         y += h/2
#         label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(tid, x/seq_width, y/seq_height, w/seq_width, h/seq_height)
#         with open(os.path.join(label_root, seq, label_fpath), 'a') as f:
#             f.write(label_str)

def gen_test_data():
    seq_root = '/home/sdb/wangshentao/myspace/thesis/data/d435i/images/2020-10-20/'
    annotations = '/home/sdb/wangshentao/myspace/thesis/data/d435i/annotations/'
    label_root = '/home/sdb/wangshentao/myspace/thesis/data/d435i/labels_with_ids/2020-10-20/'
    mkdirs(label_root)
    #seqs = [s for s in os.listdir(seq_root)]
    seqs = ['2020-10-20-19-50-01']
    seq_width = 424
    seq_height = 240
    for seq in seqs:
        frame_id = 1
        gt_txt = osp.join(annotations, seq+'.txt')
        seq_std = seq + '-std'
        # rename the image names
        if not os.path.exists(os.path.join(seq_root, seq_std)):
            os.makedirs(os.path.join(seq_root, seq_std))
        if not os.path.exists(os.path.join(label_root, seq_std)):
            os.makedirs(os.path.join(label_root, seq_std))
        image_files = os.listdir(os.path.join(seq_root, seq))
        image_files = [i for i in image_files if '.jpg' in i]
        image_files = sorted(image_files)
        image2id = {}
        for idx, file_name in enumerate(image_files):
            shutil.copy(os.path.join(seq_root, seq, file_name), os.path.join(seq_root, seq_std, "{:0>7d}.jpg".
                                                                             format(frame_id)))
            image2id[file_name] = frame_id
            frame_id += 1
        with open(gt_txt, 'r') as fin:
            lines = fin.readlines()
        fout_data = defaultdict(list)
        for line in lines:
            data = line
            line = line.split(',')
            data = data.replace(line[0], "{}".format(image2id[line[0]]))
            data = data.strip() + ",1,0,0,0\n"
            label_fpath = "{:0>7d}.txt".format(image2id[line[0]])
            tid = int(line[1])
            fout_data[tid].append(data)
            x, y, w, h = int(line[2]), int(line[3]), int(line[4]), int(line[5])
            x = np.clip(x, 0, seq_width)
            y = np.clip(y, 0, seq_height)
            w = np.clip(w, 0, seq_width-x)
            h = np.clip(h, 0, seq_height-y)
            x += w/2
            y += h/2
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(tid, x/seq_width, y/seq_height, w/seq_width, h/seq_height)
            # with open(os.path.join(label_root, seq_std, label_fpath), 'a') as f:
            #     f.write(label_str)
        fout = open(os.path.join(annotations, seq_std + '.txt'), 'w')
        for key in fout_data:
            for data in fout_data[key]:
                fout.write(data)


def rename_files():
    seq_root = "/home/sdb/wangshentao/myspace/thesis/data/d435i/images/2020-10-20/"
    depth_image_dir = "depth_image"
    data_dir = "data"
    seqs = ["2020-10-20-19-50-01"]
    for seq in seqs:
        frame_id = 1
        save_data_dir = os.path.join(seq_root, seq+"-data-std")
        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir)
        save_depth_image_dir = os.path.join(seq_root, seq+"-depth-std")
        if not os.path.exists(save_depth_image_dir):
            os.makedirs(save_depth_image_dir)
        color_image_files = os.listdir(os.path.join(seq_root, seq))
        color_image_files = [i for i in color_image_files if ".jpg" in i]
        color_image_files = sorted(color_image_files)
        for color_image_file in color_image_files:
            file = color_image_file.replace(".jpg", ".pickle")
            shutil.copy(os.path.join(seq_root, data_dir, file),
                        os.path.join(save_data_dir, "{:0>7d}.pickle".format(frame_id)))
            shutil.copy(os.path.join(seq_root, depth_image_dir, file),
                        os.path.join(save_depth_image_dir, "{:0>7d}.pickle".format(frame_id)))
            frame_id += 1


if __name__ == "__main__":
    rename_files()