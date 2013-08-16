#!/usr/bin/env python

"""
Records a video from the default camera for a specified duration using OpenCV.
Usage:
  camrecord.py [gui|nogui [<duration> [<video_filename> [<start_after>]]]]
Video file: <video_filename> (default: video_filename = video.mpeg, duration = 10 secs., start_after = 0 secs.)
Video data file (frame count, recorded duration, avg. fps): <video_filename>.dat
"""

import sys
import cv2
import cv2.cv as cv
from time import sleep

defaultDuration = 10.0  # sec
delay = 50  # ms
delayS = delay / 1000.0  # sec

camera_frame_width = 640
camera_frame_height = 480

camera = cv2.VideoCapture(0)
print "Opening camera..."
if not camera.isOpened():
  print "Error opening camera; aborting..."
  sys.exit(1)

result_width = camera.set(cv.CV_CAP_PROP_FRAME_WIDTH, camera_frame_width)
result_height = camera.set(cv.CV_CAP_PROP_FRAME_HEIGHT, camera_frame_height)
print "Camera frame size set to {width}x{height} (result: {result_width}, {result_height})".format(width=camera_frame_width, height=camera_frame_height, result_width=result_width, result_height=result_height)

gui = True
duration = defaultDuration
videoFilename = "video.mpeg"
startAfter = 0.0  # sec
video = None
dat = None

if len(sys.argv) > 1:
  gui = (sys.argv[1] == "gui")

if len(sys.argv) > 2:
  duration = float(sys.argv[2])

if len(sys.argv) > 3:
  videoFilename = sys.argv[3]

if len(sys.argv) > 4:
  startAfter = float(sys.argv[4])

if videoFilename is not None:
  print "Opening video file \"" + videoFilename + "\"..."
  video = cv2.VideoWriter(videoFilename, cv2.cv.CV_FOURCC('M', 'P', 'E', 'G'), 30, (640, 480))
  if not video.isOpened():
    print "Error opening video file; aborting..."
    sys.exit(1)
  
  datFilename = videoFilename + ".dat"
  print "Opening video data file \"" + datFilename + "\"..."
  dat = open(datFilename, 'w')

if video is None:
  print "Video file not specified or cannot be opened (check usage); aborting..."
  sys.exit(1)

print "Main loop (GUI: {0}, duration: {1} secs. (start after {2} secs.), video file: {3})...".format(gui, duration, startAfter, videoFilename)
isWaiting = (startAfter > 0.0)
frameCount = 0
fps = 0.0
timeStart = cv2.getTickCount() / cv2.getTickFrequency()
timeLast = timeNow = 0.0

while True:
  timeNow = (cv2.getTickCount() / cv2.getTickFrequency()) - timeStart
  
  timeDiff = (timeNow - timeLast)
  fps = (1.0 / timeDiff) if (timeDiff > 0.0) else 0.0
  #print "Frame %d: %5.2f fps" % (frameCount, fps)
  
  _, frame = camera.read()
  if frame is None:
    break
  
  if isWaiting:
    if timeNow >= startAfter:
      timeStart += timeNow
      timeLast = timeNow = 0.0
      frameCount = 0
      fps = 0.0
      isWaiting = False
    else:
      if gui:
        cv2.putText(frame, "{:.2f}".format(startAfter - timeNow), (frame.shape[1] - 80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
  else:
    if video is not None:
      video.write(frame)
  
  if gui:
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(delay)
    if key != -1:
      break
  else:
    sleep(delayS)
  
  if timeNow > duration:
    break
  
  frameCount = frameCount + 1
  timeLast = timeNow

avgFPS = frameCount / timeNow
print "Done; %d frames, %.2f secs, %.2f fps" % (frameCount, timeNow, avgFPS)

if video is not None:
  print "Releasing video file..."
  video.release()
  
  if dat is not None:
    dat.write("{0} {1} {2}".format(frameCount, timeNow, avgFPS))
    print "Closing video data file..."
    dat.close()

print "Releasing camera..."
camera.release()
