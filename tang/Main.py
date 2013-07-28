# Python imports
import sys
import os
import time
import numpy as np
import logging.config
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
from Renderer import Renderer
from Environment import Environment
from Controller import Controller
if haveCV:
    from vision.input import VideoInput
    from vision.gl import FrameProcessorGL
    from vision.colortracking import ColorTracker

# Globals
# NOTE: glBlitFramebuffer() doesn't play well if source and destination sizes are different, so keep these same
windowWidth, windowHeight = (640, 480)
cameraWidth, cameraHeight = (640, 480)
gui = False
debug = False

def usage():
    print "Usage: {} [<resource_path> [<camera_device> | <video_filename>]]".format(sys.argv[0])
    print "Arguments:"
    print "  resource_path   Path to \'res\' dir. containing models, shaders, etc."
    print "  camera_device   Integer specifying camera to read from (default: 0)"
    print "  video_filename  Path to video file to use instead of live camera"
    print "Note: Only one of camera_device or video_filename should be specified."
    print


def main():
    usage()
    
    # * Obtain resource path and other parameters
    # TODO Start using optparse/argparse instead of positional arguments
    resPath = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "res"))  # NOTE only absolute path seems to work properly
    #print "main(): Resource path:", resPath  # only needed if logging is not working
    gui = '--gui' in sys.argv
    debug = '--debug' in sys.argv
    
    # * Setup logging (before any other object is initialized that obtains a logger)
    # ** Load configuration from file
    logConfigFile = os.path.abspath(os.path.join(resPath, "config", "logging.conf"))  # TODO make log config filename an optional argument
    os.chdir(os.path.dirname(logConfigFile))  # change to log config file's directory (it contains relative paths)
    logging.config.fileConfig(logConfigFile)  # load configuration
    os.chdir(sys.path[0])  # change back to current script's directory
    # ** Tweak root logger configuration based on command-line arguments
    if debug and logging.getLogger().getEffectiveLevel() > logging.DEBUG:
      logging.getLogger().setLevel(logging.DEBUG)
    elif not debug and logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
      logging.getLogger().setLevel(logging.INFO)  # one level above DEBUG
      # NOTE Logging level order: DEBUG < INFO < WARN < ERROR < CRITICAL
    # ** Obtain a logger for this module
    logger = logging.getLogger(__name__)
    #logger.debug("Logging system ready")  # Example: Log a debug message
    #logger.log(INFO, "Resource path: %s", resPath)  # Example: Log a message with a specified level (INFO) and formatted args.
    if not haveCV:
      logger.warn("OpenCV library not available")  # Example: Log a warning message (will be stored in log file)
    
    # * Obtain camera device no. / input video filename
    cameraDevice = 0  # NOTE A default video filename can be specified here, but isVideo must also be set to true then
    isVideo = False
    if len(sys.argv) > 2:
        try:
            cameraDevice = int(sys.argv[2])  # works if sys.argv[2] is an int (device no.)
        except ValueError:
            cameraDevice = os.path.abspath(sys.argv[2])  # fallback: treat sys.argv[2] as string (filename)
            isVideo = True
    
    # * Initialize GL rendering context and associated objects
    renderer = Renderer(resPath)
    environment = Environment(renderer)
    controller = Controller(environment)
    
    # * Open camera/video file and start vision loop, if OpenCV is available
    if haveCV:
        logger.info("Camera device / video input file: {}".format(cameraDevice))
        camera = cv2.VideoCapture(cameraDevice)
        options={ 'gui': gui, 'debug': debug,
                  'isVideo': isVideo, 'loopVideo': True,
                  'cameraWidth': cameraWidth, 'cameraHeight': cameraHeight,
                  'windowWidth': windowWidth, 'windowHeight': windowHeight }
        videoInput = VideoInput(camera, options)
        colorTracker = ColorTracker(options)  # color marker tracker
        imageBlitter = FrameProcessorGL(options)  # CV-GL renderer that blits (copies) CV image to OpenGL window
        # TODO Evaluate 2 options: Have separate tracker and blitter/renderer or one combined tracker that IS-A FrameProcessorGL? (or make a pipeline?)
        colorTracker.initialize(videoInput.image, 0.0)
        imageBlitter.initialize(colorTracker.imageOut if colorTracker.imageOut is not None else videoInput.image, 0.0)
        
        def visionLoop():
            logger.info("[Vision loop] Starting...")
            while renderer.windowOpen():
                if videoInput.read():
                  colorTracker.process(videoInput.image, 0.0)  # NOTE this can be computationally expensive
                  imageBlitter.process(colorTracker.imageOut if colorTracker.imageOut is not None else videoInput.image, 0.0)
                  
                  # Rotate and translate model to match tracked cube (NOTE Y and Z axis directions are inverted between CV and GL)
                  environment.model[0:3, 0:3], _ = cv2.Rodrigues(colorTracker.rvec)  # convert rotation vector to rotation matrix (3x3) and populate model matrix
                  environment.model[0:3, 3] = colorTracker.tvec[0:3, 0]  # copy in translation vector into 4th column of model matrix
            logger.info("[Vision loop] Done.")
        
        visionThread = Thread(target=visionLoop)
        visionThread.start()
    
    # * Start GL render loop
    while renderer.windowOpen():
        controller.pollInput()
        renderer.startDraw()
        imageBlitter.render()
        environment.draw()
        renderer.endDraw()
    
    # * Clean up
    if haveCV:
        logger.info("Waiting on vision thread to finish...")
        visionThread.join()
        logger.info("Cleaning up...")
        camera.release()


if __name__ == '__main__':
    main()
