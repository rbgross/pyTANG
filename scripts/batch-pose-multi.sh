#!/usr/bin/env bash

videoDir=${1:-../res/videos/virtual/}
outDir=../out/
poseDir=../out/pose-multi/
fpsRates=(10 20 30 40 50)

echo "Batch-run record pose task at multiple speeds"
echo "FPS rates: ${fpsRates[@]}"

#rm ../out/pose.dat
mkdir -p $poseDir

echo "Reading input videos from: $videoDir"
echo "Saving pose sequences to : $poseDir"
for videoFile in `ls $videoDir*.mp4 | xargs -n 1 basename`
do
  videoFilepath="$videoDir$videoFile"
  for fps in "${fpsRates[@]}"
  do
    poseFilepath="$poseDir${videoFile%.mp4}_${fps}fps.dat"
    echo
    echo "*** Running: $videoFilepath => $poseFilepath"
    ./Main.py $videoFilepath --sync_video --video_fps=$fps --task=RecordPoseTask
    if [ -f ${outDir}pose.dat ]
    then
      mv ${outDir}pose.dat $poseFilepath
    fi
  done
done
