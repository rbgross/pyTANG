#!/usr/bin/env python

"""Tangible Data Exploration.

Usage: Main.py [input_source]
For more options: Main.py --help

"""

# Python imports
import sys
import os
import time
import numpy as np
import logging
import argparse
from threading import Thread
from importlib import import_module

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
from task.Task import Task  # base (and dummy) Task class
if haveCV:
  from vision.input import VideoInput
  from vision.gl import FrameProcessorGL
  from vision.colortracking import CubeTracker

# Globals
# NOTE: glBlitFramebuffer() doesn't play well if source and destination sizes are different, so keep these same
windowWidth, windowHeight = (640, 480)
cameraWidth, cameraHeight = (640, 480)

class Main:
  """Main application class."""
  
  def __init__(self):
    #sys.argv = ['Main.py', '../res/videos/test-14.mpeg', '--hide_input']  # [debug: run with set command-line args]
    
    # * Initialize global context, passing in custom command line args (parsed by Context)
    argParser = argparse.ArgumentParser(add_help=False)
    showInputGroup = argParser.add_mutually_exclusive_group()
    showInputGroup.add_argument('--show_input', dest='show_input', action="store_true", default=True, help="show input video (emulate see-through display)?")
    showInputGroup.add_argument('--hide_input', dest='show_input', action="store_false", default=False, help="hide input video (show only virtual objects)?")
    argParser.add_argument('--task', default="Task", help="task to run (maps to class name)")
    argParser.add_argument('--scene', dest="scene_files", metavar='SCENE_FILE', nargs='+', help="scene fragment(s) to load (filenames in <res>/data/)")
    
    self.context = Context.createInstance(description="Tangible Data Exploration", parent_argparsers=[argParser])
    self.context.main = self  # hijack global context to share a reference to self
    # NOTE Most objects require an initialized context, so do this as soon as possible
    
    # * Obtain a logger (NOTE Context must be initialized first since it configures logging)
    self.logger = logging.getLogger(__name__)
    self.logger.info("Resource path: {}".format(self.context.resPath))
    if not haveCV:
      self.logger.warn("OpenCV library not available")
    
    # * Initialize GL rendering context and associated objects (NOTE order of initialization may be important)
    self.context.renderer = Renderer()
    self.context.controller = Controller()
    
    # * Initialize scene and load base scene fragments, including tools
    self.context.scene = Scene()
    self.context.scene.readXML(self.context.getResourcePath('data', 'CubeScene.xml'))  # just the cube
    #self.context.scene.readXML(self.context.getResourcePath('data', 'DragonScene.xml'))  # Stanford Dragon
    #self.context.scene.readXML(self.context.getResourcePath('data', 'BP3D-FMA7088-heart.xml'))  # BodyParts3D heart model hierarchy
    #self.context.scene.readXML(self.context.getResourcePath('data', 'RadialTreeScene.xml'))
    #self.context.scene.readXML(self.context.getResourcePath('data', 'PerspectiveScene.xml'))
    
    # ** Load scene fragments specified on commandline (only need to specify filename in data/ directory)
    self.logger.info("Scene fragment(s): %s", self.context.options.scene_files)
    if self.context.options.scene_files is not None:
        for scene_file in self.context.options.scene_files:
            self.context.scene.readXML(self.context.getResourcePath('data', scene_file))
    
    # * Initialize task (may load further scene fragments, including task-specific tools)
    try:
      taskModule = import_module('task.' + self.context.options.task)  # fetch module by name from task package
      taskType = getattr(taskModule, self.context.options.task)  # fetch class by name from corresponding module (same name, by convention)
      self.context.task = taskType()  # create an instance of specified task class
    except Exception as e:
      self.logger.error("Task initialization error: {}".format(e))
      self.context.task = Task()  # fallback to dummy task
    
    # * Finalize scene (resolves scene fragments into one hierarchy, builds ID-actor mapping)
    self.context.scene.finalize()  # NOTE should be called after all read*() methods have been called on scene
    
    # * Find cube in scene
    self.cubeActor = self.context.scene.findActorById('cube')
    self.cubeComponent = self.cubeActor.components['Cube'] if self.cubeActor is not None else None
    
    # * Open camera/input file
    self.logger.info("Input device/file: {}".format(self.context.options.input_source))
    self.camera = cv2.VideoCapture(self.context.options.input_source) if not self.context.isImage else cv2.imread(self.context.options.input_source)
    # TODO move some more options (e.g. *video*) to context; introduce config.yaml-like solution with command-line overrides
    self.options={ 'gui': self.context.options.gui, 'debug': self.context.options.debug,
                   'isVideo': self.context.isVideo, 'loopVideo': self.context.options.loop_video, 'syncVideo': self.context.options.sync_video, 'videoFPS': self.context.options.video_fps,
                   'isImage': self.context.isImage,
                   'cameraWidth': cameraWidth, 'cameraHeight': cameraHeight,
                   'windowWidth': windowWidth, 'windowHeight': windowHeight }
    self.context.videoInput = VideoInput(self.camera, self.options)
    # TODO If live camera, let input image stabilize by eating up some frames, then configure camera
    #   e.g. on Mac OS, use uvc-ctrl to turn off auto-exposure:
    #   $ ./uvc-ctrl -s 1 3 10
    
    # * Create image blitter, if input is to be shown
    if self.context.options.show_input:
      self.context.imageBlitter = FrameProcessorGL(self.options)  # CV-GL renderer that blits (copies) CV image to OpenGL window
      # TODO Evaluate 2 options: Have separate tracker and blitter/renderer or one combined tracker that IS-A FrameProcessorGL? (or make a pipeline?)
      self.logger.info("Video see-through mode enabled; input video underlay will be shown")
    else:
      self.logger.info("Video see-through mode disabled; only virtual objects will be shown")
    
    # * Setup tracking
    self.context.cubeTracker = CubeTracker(self.options)  # specialized cube tracker, available in context to allow access to cubeTracker's input and output images etc.
    if self.cubeComponent is not None:
      self.logger.info("Tracking setup: Cube has {} markers".format(len(self.cubeComponent.markers)))
      self.context.cubeTracker.addMarkersFromTrackable(self.cubeComponent)
  
  def run(self):
    # * Activate task (may try to find task-specific tools in scene)
    self.context.task.activate()
    
    # * Reset system-wide timer
    self.context.resetTime()
    
    # * Start CV loop on separate thread
    # ** Initialize CV processors (NOTE videoInput should already have read an image)
    self.context.cubeTracker.initialize(self.context.videoInput.image, self.context.timeNow)
    if self.context.options.show_input:
      self.context.imageBlitter.initialize(
        self.context.cubeTracker.imageOut if self.context.cubeTracker.imageOut is not None
        else self.context.videoInput.image,
        self.context.timeNow)
    # ** Create new thread and start it
    cvThread = Thread(target=self.cvLoop)
    cvThread.start()
    time.sleep(0.001)  # yield to the new thread
    
    # * Start GL loop on main thread (NOTE start GL after CV so that imageBlitter has a chance to get an image)
    self.glLoop()
    
    # * Clean up
    self.logger.info("Waiting on CV thread to finish...")
    cvThread.join()
    self.logger.info("Cleaning up...")
    if self.context.task.active:
      self.context.task.deactivate()  # NOTE required for some tasks (to close files, etc.)
    if not self.context.isImage:
      self.camera.release()
    self.logger.info("Done.")
  
  def cvLoop(self):
    # * CV process loop
    if self.context.videoInput.isVideo:
      self.context.videoInput.resetVideo()
    self.logger.info("[CV loop] Starting...")
    while self.context.renderer.windowOpen() and self.context.videoInput.isOkay:
      # ** Read a frame
      if self.context.videoInput.read():
        # *** If valid frame is available, perform tracking
        self.context.cubeTracker.process(self.context.videoInput.image, self.context.timeNow)  # NOTE this can be computationally expensive
        
        # *** Cache CV output image (normally the raw camera input) for blitting in GL loop
        if self.context.options.show_input:
          self.context.imageBlitter.process(
            self.context.cubeTracker.imageOut if self.context.cubeTracker.imageOut is not None
            else self.context.videoInput.image,
            self.context.timeNow)
        
        # *** Rotate and translate model transform to match tracked cube (NOTE Y and Z axis directions are inverted between CV and GL)
        if not self.context.controller.manualControl:
          self.context.scene.transform[0:3, 0:3], _ = cv2.Rodrigues(self.context.cubeTracker.rvec)  # convert rotation vector to rotation matrix (3x3) and populate model transformation matrix
          self.context.scene.transform[0:3, 3] = self.context.cubeTracker.tvec[0:3, 0]  # copy in translation vector into 4th column of model transformation matrix
        
        # *** Handle OpenCV window events
        if self.context.options.gui:
          cv2.waitKey(1)
    self.logger.info("[CV loop] Done.")
  
  def glLoop(self):
    # * GL render loop
    self.logger.info("[GL loop] Starting...")
    while self.context.renderer.windowOpen() and self.context.videoInput.isOkay:
      # ** Context updates (mainly time)
      self.context.update()
      
      # ** Controller updates
      self.context.controller.pollInput()
      if self.context.controller.doQuit:
        self.context.renderer.quit()
        break
      
      # ** Render background image and scene
      self.context.renderer.startDraw()
      if self.context.options.show_input:
        self.context.imageBlitter.render()
      self.context.scene.draw()
      self.context.renderer.endDraw()
      
      # ** Task-specific updates (here, since transforms are computed during draw calls)
      if self.context.task.active:
        self.context.task.update()
    self.logger.info("[GL loop] Done.")


if __name__ == '__main__':
  Main().run()
