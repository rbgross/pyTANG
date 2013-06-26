# Python imports
import sys
import os
import time
import numpy as np
import logging.config

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
    from GLCameraViewer import GLCameraViewer

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
    
    # TODO Start using optparse/argparse instead of positional arguments
    resPath = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "res"))  # NOTE only absolute path seems to work properly
    #print "main(): Resource path:", resPath  # only needed if logging is not working
    
    # Setup logging
    logConfigFile = os.path.abspath(os.path.join(resPath, "config", "logging.conf"))  # TODO make log config filename an optional argument
    os.chdir(os.path.dirname(logConfigFile))  # change to log config file's directory (it contains relative paths)
    logging.config.fileConfig(logConfigFile)  # load configuration
    os.chdir(sys.path[0])  # change back to current script's directory
    logger = logging.getLogger(__name__)
    #logger.debug("Logging system ready")  # Example: Log a debug message
    #logger.log(INFO, "Resource path: %s", resPath)  # Example: Log a message with a specified level (INFO) and formatted args.
    if not haveCV:
      logger.warn("OpenCV library not available")  # Example: Log a warning message (will be stored in log file)
    
    cameraDevice = 0  # NOTE A default video filename can be specified here, but isVideo must be set to true then
    isVideo = False
    if len(sys.argv) > 2:
        try:
            cameraDevice = int(sys.argv[2])  # works if sys.argv[2] is an int (device no.)
        except ValueError:
            cameraDevice = os.path.abspath(sys.argv[2])  # fallback: treat sys.argv[2] as string (filename)
            isVideo = True
    
    renderer = Renderer(resPath)
    environment = Environment(renderer)
    controller = Controller(environment)
    
    if haveCV:
        logger.debug("Camera device / video input file: {}".format(cameraDevice))
        camera = cv2.VideoCapture(cameraDevice)
        cameraViewer = GLCameraViewer(camera, isVideo, True)
    
    while renderer.windowOpen():
        controller.pollInput()
        renderer.startDraw()
        if haveCV:
            cameraViewer.capture()
            cameraViewer.process()  # NOTE this can be computationally expensive
            cameraViewer.render()
        environment.draw()
        renderer.endDraw()

    if haveCV:
        logger.debug("Cleaning up...")
        cameraViewer.cleanUp()
        camera.release()

if __name__ == '__main__':
    main()
