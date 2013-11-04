import logging

import numpy as np
import cv2
import cv2.cv as cv

from task.Task import Task

class RecordPoseTask(Task):
  """Continually record movable cursor object's pose."""
  
  def __init__(self):
    # NOTE Scene must be initialized at this point
    Task.__init__(self)
    self.cursor = None
    self.lastFrameCount = -1
    self.showPoseWindow = True  # to be treated as a constant flag
    if self.showPoseWindow:
      self.imageOut = np.zeros((50, 300, 3), dtype=np.uint8)
      self.windowName = "Cursor pose"
  
  def activate(self):
    self.cursor = self.context.scene.findActorById('cube')  # NOTE cube itself is the cursor object
    if self.cursor is None:
      self.logger.warn("[RecordPoseTask] Cursor object not found; task could not be initialized")
      return
    self.cursor.visible = True
    if self.showPoseWindow:
      self.imageOut.fill(255)
      cv2.imshow(self.windowName, self.imageOut)
      cv2.waitKey(1)
    Task.activate(self)
  
  def deactivate(self):
    Task.deactivate(self)
    self.cursor.visible = False
    if self.showPoseWindow:
      cv2.destroyWindow(self.windowName)
  
  def update(self):
    if self.context.videoInput.frameCount != self.lastFrameCount:
      # Get translation vector
      tvec = self.cursor.transform[0:3, 3]
      #self.logger.debug("Translation vector:-\n{}".format(tvec))
      
      # Get rotation vector
      rmat = self.cursor.transform[0:3, 0:3]
      #self.logger.debug("Rotation matrix:-\n{}".format(rmat))
      rvec = np.float32([0.0, 0.0, 0.0])  # TODO use cv2.Rodrigues to get individual rotation vector (i.e. x, y, z components)
      
      # TODO Record pose (translation & rotation) to file/stdout
      poseRecord = "{timeNow}\t{frameCount}\t{tvec[0]}\t{tvec[1]}\t{tvec[2]}\t{rvec[0]}\t{rvec[1]}\t{rvec[2]}".format(
        timeNow=self.context.timeNow,
        frameCount=self.context.videoInput.frameCount,
        tvec=tvec,
        rvec=rvec)
      #print poseRecord  # [debug]
      
      if self.showPoseWindow:
        self.imageOut.fill(255)
        cv2.putText(self.imageOut, "t: {vec[0]:+7.2f}, {vec[1]:+7.2f}, {vec[2]:+7.2f}".format(vec=tvec), (5, 21), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 100, 100), 1)
        cv2.putText(self.imageOut, "r: {vec[0]:+7.2f}, {vec[1]:+7.2f}, {vec[2]:+7.2f}".format(vec=rvec), (5, 41), cv2.FONT_HERSHEY_DUPLEX, 0.6, (100, 100, 200), 1)
        cv2.imshow(self.windowName, self.imageOut)
        cv2.waitKey(1)  # NOTE might slow things down
      
      self.lastFrameCount = self.context.videoInput.frameCount
