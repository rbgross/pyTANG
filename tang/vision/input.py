"""Components to help manage video and camera inputs."""

# Python imports
import sys
import logging

# OpenCV imports
import cv2
import cv2.cv as cv

# Custom imports
from util import KeyCode, isImageFile
from base import FrameProcessor

# Globals
cameraWidth = 640
cameraHeight = 480

class VideoInput:
  """Abstracts away the handling of recorded video files and live camera as input."""
  # TODO Incorporate static images, and option of syncing video playback to realtime
  
  def __init__(self, camera, options):
    # * Obtain video source (camera) and optional parameters
    self.camera = camera
    self.isVideo = options.get('isVideo', False)
    self.isImage = options.get('isImage', False)
    self.loopVideo = options.get('loopVideo', True)
    self.cameraWidth = options.get('cameraWidth', cameraWidth)
    self.cameraHeight = options.get('cameraHeight', cameraHeight)
    
    # * Acquire logger and initialize other members
    self.logger = logging.getLogger(self.__class__.__name__)
    self.frameCount = 0
    
    # * Set camera frame size (if this is a live camera)
    if not self.isVideo and not self.isImage:
      #_, self.imageIn = self.camera.read()  # pre-grab
      # NOTE: If camera frame size is not one supported by the hardware, grabbed images are scaled to desired size, discarding aspect-ratio
      self.camera.set(cv.CV_CAP_PROP_FRAME_WIDTH, self.cameraWidth)
      self.camera.set(cv.CV_CAP_PROP_FRAME_HEIGHT, self.cameraHeight)
    
    # * Check if this is a static image or camera/video
    if self.isImage:
      # ** Supplied camera object should be an image, copy it
      self.image = self.camera
    else:
      # ** Grab test image and read some properties
      _, self.image = self.camera.read()  # post-grab (to apply any camera prop changes made)
      self.frameCount += 1
      self.logger.info("Camera size: {}x{}".format(int(self.camera.get(cv.CV_CAP_PROP_FRAME_WIDTH)), int(self.camera.get(cv.CV_CAP_PROP_FRAME_HEIGHT))))
    self.imageSize = (self.image.shape[1], self.image.shape[0])
    self.logger.info("Image size: {}x{}".format(self.imageSize[0], self.imageSize[1]))
    if self.isVideo:
      self.numVideoFrames = int(self.camera.get(cv.CV_CAP_PROP_FRAME_COUNT))  # read num frames (if video)
      self.logger.info("Video file with {} frames".format(self.numVideoFrames))
    
    self.isOkay = True  # all good, so far
  
  def read(self):
    if self.isVideo and self.loopVideo and self.frameCount >= self.numVideoFrames:
      self.camera.set(cv.CV_CAP_PROP_POS_FRAMES, 0)
      self.frameCount = 0
      self.logger.debug("Video reset...")
      # TODO Figure out what's causing the off-by-ten bug (after a reset, the last 10-11 frames cannot be read anymore!)
    
    if not self.isImage:
      self.isOkay, self.image = self.camera.read()
      self.frameCount += 1
    
    return self.isOkay


def run(processor=FrameProcessor(options={ 'gui': True, 'debug': True }), gui=True, debug=True):  # default options
  """Run a FrameProcessor object on a static image (repeatedly) or on frames from a camera/video."""
  # TODO Use VideoInput instance instead of duplicating input logic
  # * Initialize parameters and flags
  delay = 10  # ms
  delayS = delay / 1000.0  # sec; only used in non-GUI mode, so this can be set to 0
  #gui = options.get('gui', True)
  #debug = options.get('debug', True)
  showInput = gui
  showOutput = gui
  showFPS = False
  showKeys = False
  
  isImage = False
  isVideo = False
  isOkay = False
  isFrozen = False
  
  # * Setup logging
  logging.basicConfig(format="%(levelname)s | %(module)s | %(funcName)s() | %(message)s", level=logging.DEBUG if debug else logging.INFO)
  
  # * Read input image or video, if specified
  if len(sys.argv) > 1:
    filename = sys.argv[1]
    if isImageFile(filename):
      print "run(): Reading image: \"" + filename + "\""
      frame = cv2.imread(filename)
      if frame is not None:
        if showInput:
          cv2.imshow("Input", frame)
        isImage = True
        isOkay = True
      else:
        print "run(): Error reading image; fallback to camera."
    else:
      print "run(): Reading video: \"" + filename + "\""
      camera = cv2.VideoCapture(filename)
      if camera.isOpened():
        isVideo = True
        isOkay = True
      else:
        print "run(): Error reading video; fallback to camera."
  
  # * Open camera if image/video is not provided/available
  if not isOkay:
    print "run(): Opening camera..."
    camera = cv2.VideoCapture(0)
    # ** Final check before processing loop
    if camera.isOpened():
      result_width = camera.set(cv.CV_CAP_PROP_FRAME_WIDTH, cameraWidth)
      result_height = camera.set(cv.CV_CAP_PROP_FRAME_HEIGHT, cameraHeight)
      print "run(): Camera frame size set to {width}x{height} (result: {result_width}, {result_height})".format(width=cameraWidth, height=cameraHeight, result_width=result_width, result_height=result_height)
      isOkay = True
    else:
      print "run(): Error opening camera; giving up now."
      return
  
  # * Initialize supporting variables
  fresh = True
  
  # * Processing loop
  timeStart = cv2.getTickCount() / cv2.getTickFrequency()
  timeLast = timeNow = 0.0
  while(1):
    # ** [timing] Obtain relative timestamp for this loop iteration
    timeNow = (cv2.getTickCount() / cv2.getTickFrequency()) - timeStart
    if showFPS:
      timeDiff = (timeNow - timeLast)
      fps = (1.0 / timeDiff) if (timeDiff > 0.0) else 0.0
      print "run(): {0:5.2f} fps".format(fps)
    
    # ** If not static image, read frame from video/camera
    if not isImage and not isFrozen:
      isValid, frame = camera.read()
      if not isValid:
        break  # camera disconnected or reached end of video
      
      if showInput:
        cv2.imshow("Input", frame)
    
    # ** Initialize FrameProcessor, if required
    if(fresh):
      processor.initialize(frame, timeNow) # timeNow should be zero on initialize
      fresh = False
    
    # ** Process frame
    imageOut = processor.process(frame, timeNow)
    
    # ** Show output image
    if showOutput and imageOut is not None:
      cv2.imshow("Output", imageOut)
    
    # ** Check if GUI is available
    if gui:
      # *** If so, wait for inter-frame delay and process keyboard events using OpenCV
      key = cv2.waitKey(delay)
      if key != -1:
        keyCode = key & 0x00007f  # key code is in the last 8 bits, pick 7 bits for correct ASCII interpretation (8th bit indicates 
        keyChar = chr(keyCode) if not (key & KeyCode.SPECIAL) else None # if keyCode is normal, convert to char (str)
        
        if showKeys:
          print "run(): Key: " + KeyCode.describeKey(key)
          #print "run(): key = {key:#06x}, keyCode = {keyCode}, keyChar = {keyChar}".format(key=key, keyCode=keyCode, keyChar=keyChar)
        
        if keyCode == 0x1b or keyChar == 'q':
          break
        elif keyChar == ' ':
          print "run(): [PAUSED] Press any key to continue..."
          ticksPaused = cv2.getTickCount()  # [timing] save time when paused
          cv2.waitKey()  # wait indefinitely for a key press
          timeStart += (cv2.getTickCount() - ticksPaused) / cv2.getTickFrequency()  # [timing] compensate for duration paused
        elif keyCode == 0x0d:
          isFrozen = not isFrozen  # freeze frame, but keep processors running
        elif keyChar == 'f':
          showFPS = not showFPS
        elif keyChar == 'k':
          showKeys = not showKeys
        elif keyChar == 'i':
          showInput = not showInput
          if not showInput:
            cv2.destroyWindow("Input")
        elif keyChar == 'o':
          showOutput = not showOutput
          if not showOutput:
            cv2.destroyWindow("Output")
        elif not processor.onKeyPress(key, keyChar):
          break
    else:
      # *** Else, wait for inter-frame delay using system method
      sleep(delayS)
    
    # ** [timing] Save timestamp for fps calculation
    timeLast = timeNow
  
  # * Clean-up
  print "run(): Cleaning up..."
  if gui:
    cv2.destroyAllWindows()
  if not isImage:
    camera.release()
