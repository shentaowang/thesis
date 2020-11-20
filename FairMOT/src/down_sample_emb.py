import os


def main():
    read_file = "/home/sdb/wangshentao/myspace/thesis/code/FairMOT/src/data/visdrone_2019_mot.valc6"
    save_file = "/home/sdb/wangshentao/myspace/thesis/code/FairMOT/src/data/visdrone_2019_mot.valc6_d4"
    down_ratio = 4
    last_seq = ""
    fin = open(read_file, 'r')
    fout = open(save_file, 'w')
    cnt = 0
    for line in fin.readlines():
        seq = line.split('/')[-2]
        if seq != last_seq:
            last_seq = seq
            cnt = 0
        if cnt % down_ratio == 0:
            fout.write(line)
        cnt += 1


if __name__ == "__main__":
    main()
