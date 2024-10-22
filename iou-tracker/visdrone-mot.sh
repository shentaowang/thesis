#!/bin/bash

# set these variables according to your setup
visdrone_dir=/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/
# base directory of the split (VisDrone2018-MOT-val, VisDrone2018-MOT-train etc.)
results_dir=/home/sdb/wangshentao/myspace/thesis/data/visdrone_2019_mot/images/results/kcf_tracker_1204/
# output directory, will be created if not existing
vis_tracker=KCF2                      # [MEDIANFLOW, KCF2, NONE] parameter set as used in the paper


if [ "${vis_tracker}" = "MEDIANFLOW" ]; then
  options="-v MEDIANFLOW  -sl 0.15 -sh 0. -si 0.1 -tm 2 --ttl 2  -fmt visdrone"
elif [ "${vis_tracker}" = "KCF2" ]; then
  options="-v KCF2 -sl 0.15 -sh 0.15 -si 0.3 -tm 2 --ttl 8  -fmt visdrone"
elif [ "${vis_tracker}" = "NONE" ]; then
  options="-sl 0.15 -sh 0.15 -si 0.3 -tm 2  -fmt visdrone"
else
  echo "unknown tracker '${vis_tracker}'"
  exit
fi

mkdir -p ${results_dir}

seq_dir=${visdrone_dir}/sequences

echo "using '${vis_tracker}' option for visual tracking:"
echo "${options}"
for seq in $(ls $seq_dir); do
  echo "processing ${seq} ...."

  python demo.py -f ${seq_dir}/${seq}/{:07d}.jpg -d ${visdrone_dir}/detections-1204/${seq}.txt \
         -o ${results_dir}/${seq}.txt ${options}
done
