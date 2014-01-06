#!/usr/bin/env bash

videoDir=${1:-../res/videos/virtual/}
outDir=../out/
poseDir=../out/pose/

echo "Batch-run record pose task"
mkdir -p $poseDir
echo "Reading input videos from: $videoDir"
for videoFile in `ls $videoDir*.mp4 | xargs -n 1 basename`
do
  videoFilepath="$videoDir$videoFile"
  poseFilepath="$poseDir${videoFile%.mp4}.dat"
  echo
  echo "*** Running: $videoFilepath => $poseFilepath"
  ./Main.py $videoFilepath --sync_video --task=RecordPoseTask
  if [ -f ${outDir}pose.dat ]
  then
    mv ${outDir}pose.dat $poseFilepath
  fi
done
