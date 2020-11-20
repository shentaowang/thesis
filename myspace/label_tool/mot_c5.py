import os


def convert_test_dev():
    keep_class = [1, 4, 5, 6, 9]
    annotation_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/annotations/"
    out_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/annotations-c5"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for file in os.listdir(annotation_dir):
        fin = open(os.path.join(annotation_dir, file), 'r')
        fout = open(os.path.join(out_dir, file), 'w')
        for line in fin.readlines():
            items = line.split(",")
            if int(items[-3]) in keep_class:
                fout.write(line)


def convert_val():
    keep_class = [1, 4, 5, 6, 9]
    annotation_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-val/annotations/"
    out_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-val/annotations-c5"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for file in os.listdir(annotation_dir):
        fin = open(os.path.join(annotation_dir, file), 'r')
        fout = open(os.path.join(out_dir, file), 'w')
        for line in fin.readlines():
            items = line.split(",")
            if int(items[-3]) in keep_class:
                fout.write(line)


if __name__ == "__main__":
    convert_val()
