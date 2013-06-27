"""Components to help manage video and camera inputs."""

# Python imports
import logging

# OpenCV imports
import cv2
import cv2.cv as cv

class VideoInput:
  """Abstracts away the handling of recorded video files and live camera as input."""
  # TODO Incorporate static images, and option of syncing video playback to realtime
  
  def __init__(self, camera, options):
    # * Obtain video source (camera) and optional parameters
    self.camera = camera
    self.isVideo = options.get('isVideo', False)
    self.loopVideo = options.get('loopVideo', True)
    self.cameraWidth = options.get('cameraWidth', 640)
    self.cameraWidth = options.get('cameraHeight', 480)
    
    # * Acquire logger and initialize other members
    self.logger = logging.getLogger(self.__class__.__name__)
    self.frameCount = 0
    
    # * Set camera frame size (if this is a live camera)
    if not self.isVideo:
      #_, self.imageIn = self.camera.read()  # pre-grab
      # NOTE: If camera frame size is not one supported by the hardware, grabbed images are scaled to desired size, discarding aspect-ratio
      self.camera.set(cv.CV_CAP_PROP_FRAME_WIDTH, self.cameraWidth)
      self.camera.set(cv.CV_CAP_PROP_FRAME_HEIGHT, self.cameraHeight)
    
    # * Grab test image and read some properties
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
    
    self.isOkay, self.image = self.camera.read()
    self.frameCount += 1
