#!/usr/bin/env bash

videoDir=${1:-../res/videos/virtual/}

echo "Batch-run record pose task"
echo "Reading input videos from: $videoDir"
for videoFile in `ls $videoDir`
do
  videoFilepath="$videoDir$videoFile"
  poseFilepath="../out/pose/${videoFile%.mp4}.dat"
  echo "\n*** Running: $videoFilepath => $poseFilepath"
  python Main.py ../res $videoFilepath
  if [ -f ../out/pose.dat ]
  then
    mv ../out/pose.dat $poseFilepath
  fi
done
