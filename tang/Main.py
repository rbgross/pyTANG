# Python imports
import sys
import os
import time
import numpy as np
import logging
from threading import Thread

# GL imports
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
import hommat as hm

# CV imports, if available
haveCV = False
try:
  import cv2
  import cv2.cv as cv
  haveCV = True
except ImportError:
  print >> sys.stderr, "Main: [Warning] Unable to import OpenCV; all CV functionality will be disabled"

# Custom imports
from Context import Context
from Renderer import Renderer
from Scene import Scene
from Controller import Controller
if haveCV:
  from vision.input import VideoInput
  from vision.gl import FrameProcessorGL
  from vision.colortracking import CubeTracker

# Globals
# NOTE: glBlitFramebuffer() doesn't play well if source and destination sizes are different, so keep these same
windowWidth, windowHeight = (640, 480)
cameraWidth, cameraHeight = (640, 480)
gui = False
debug = False

class Main:
  """Main application class."""
  
  def __init__(self):

    #sys.argv = ['Main.py', '../res', '../res/videos/test-10.mpeg']
    self.experimentalMode = False
    
    # * Initialize global context (command line args are parsed by Context)
    self.context = Context.createInstance(sys.argv)
    # NOTE Most objects require an initialized context, so do this as soon as possible
    
    # * Obtain a logger (and show off!)
    self.logger = logging.getLogger(__name__)
    self.logger.info("Resource path: {}".format(self.context.resPath))
    if not haveCV:
      self.logger.warn("OpenCV library not available")
    
    # * Initialize GL rendering context and associated objects (NOTE order of initialization may be important)
    self.context.renderer = Renderer()
    self.context.scene = Scene()
    self.context.controller = Controller()
    
    # * Find cube (TODO and other tools?) in scene (or add programmatically?)
    self.cubeActor = self.context.scene.findActorByComponent('Cube')
    self.cubeComponent = self.cubeActor.components['Cube'] if self.cubeActor is not None else None
  
  def run(self):
    # * Open camera/video file and start vision loop, if OpenCV is available
    if haveCV:
      self.logger.info("Camera device / video input file: {}".format(self.context.cameraDevice))
      camera = cv2.VideoCapture(self.context.cameraDevice)
      options={ 'gui': self.context.gui, 'debug': self.context.debug,
                'isVideo': self.context.isVideo, 'loopVideo': True,
                'cameraWidth': cameraWidth, 'cameraHeight': cameraHeight,
                'windowWidth': windowWidth, 'windowHeight': windowHeight }
      videoInput = VideoInput(camera, options)
      cubeTracker = CubeTracker(options)  # specialized cube tracker
      
      # TODO let input image stabilize by eating up some frames, then configure camera
      #   e.g. on Mac OS, use uvc-ctrl to turn off auto-exposure:
      #   $ ./uvc-ctrl -s 1 3 10
      
      # Setup tracking
      if self.cubeComponent is not None:
        cubeTracker.addMarkersFromTrackable(self.cubeComponent)
      
      imageBlitter = FrameProcessorGL(options)  # CV-GL renderer that blits (copies) CV image to OpenGL window
      # TODO Evaluate 2 options: Have separate tracker and blitter/renderer or one combined tracker that IS-A FrameProcessorGL? (or make a pipeline?)
      cubeTracker.initialize(videoInput.image, time.time())
      imageBlitter.initialize(cubeTracker.imageOut if cubeTracker.imageOut is not None else videoInput.image, 0.0)
      
      def visionLoop():
        self.logger.info("[Vision loop] Starting...")
        while self.context.renderer.windowOpen():
          if videoInput.read():
            cubeTracker.process(videoInput.image, time.time())  # NOTE this can be computationally expensive
            imageBlitter.process(cubeTracker.imageOut if cubeTracker.imageOut is not None else videoInput.image, time.time())
            
            # Rotate and translate model transform to match tracked cube (NOTE Y and Z axis directions are inverted between CV and GL)
            if not self.context.controller.manualControl:
              self.context.scene.transform[0:3, 0:3], _ = cv2.Rodrigues(cubeTracker.rvec)  # convert rotation vector to rotation matrix (3x3) and populate model transformation matrix
              self.context.scene.transform[0:3, 3] = cubeTracker.tvec[0:3, 0]  # copy in translation vector into 4th column of model transformation matrix
        self.logger.info("[Vision loop] Done.")
      
      visionThread = Thread(target=visionLoop)
      visionThread.start()
    
    # * Start GL render loop
    while self.context.renderer.windowOpen():
      self.context.controller.pollInput()
      self.context.renderer.startDraw()
      if not self.experimentalMode:
        imageBlitter.render()
      self.context.scene.draw()
      self.context.renderer.endDraw()
    
    # * Clean up
    if haveCV:
      self.logger.info("Waiting on vision thread to finish...")
      visionThread.join()
      self.logger.info("Cleaning up...")
      camera.release()

def usage():
  print "Usage: {} [<resource_path> [<camera_device> | <video_filename>]]".format(sys.argv[0])
  print "Arguments:"
  print "  resource_path   Path to \'res\' dir. containing models, shaders, etc."
  print "  camera_device   Integer specifying camera to read from (default: 0)"
  print "  video_filename  Path to video file to use instead of live camera"
  print "Note: Only one of camera_device or video_filename should be specified."
  print

if __name__ == '__main__':
  usage()
  Main().run()
