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
    self.outFile = None
    self.poseRecordHeader = "frame\ttime\ttrans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\n"
    self.poseRecordFormat = "{frameCount}\t{timeNow}\t{tvec[0]}\t{tvec[1]}\t{tvec[2]}\t{rvec[0]}\t{rvec[1]}\t{rvec[2]}\n"
    self.lastFrameCount = -1
    self.showPoseWindow = False  # to be treated as a constant flag; may slow down recording
    if self.showPoseWindow:
      self.imageOut = np.zeros((50, 300, 3), dtype=np.uint8)
      self.windowName = "Cursor pose"
  
  def activate(self):
    self.cursor = self.context.scene.findActorById('cube')  # NOTE cube itself is the cursor object
    if self.cursor is None:
      self.logger.warn("[RecordPoseTask] Cursor object not found; task could not be initialized")
      return
    self.cursor.visible = True
    self.outFile = open(self.context.getResourcePath("../out", "pose.dat"), "w")  # TODO change to getOutputPath()?
    self.outFile.write(self.poseRecordHeader)
    self.lastFrameCount = -1
    if self.showPoseWindow:
      self.imageOut.fill(255)
      cv2.imshow(self.windowName, self.imageOut)
      cv2.waitKey(1)
    Task.activate(self)
  
  def deactivate(self):
    if self.active:
      Task.deactivate(self)
      self.cursor.visible = False
      if self.outFile is not None and not self.outFile.closed:
        self.outFile.close()
      if self.showPoseWindow:
        cv2.destroyWindow(self.windowName)
  
  def update(self):
    if self.context.videoInput.frameCount != self.lastFrameCount:
      # Get translation vector
      tvec = self.cursor.transform[0:3, 3]
      #self.logger.info("Translation vector: {}".format(tvec))  # [debug]
      
      # Get rotation vector
      rmat = self.cursor.transform[0:3, 0:3]
      #self.logger.debug("Rotation matrix:-\n{}".format(rmat))
      rvec, _ = cv2.Rodrigues(rmat)  # get rotation vector (i.e. individual x, y, z components)
      rvec.shape = (3,)  # reshape to 1x3 vector
      #self.logger.info("Rotation vector: {}".format(rvec))  # [debug]
      
      # Record pose (translation & rotation) to file
      poseRecord = self.poseRecordFormat.format(
        frameCount=self.context.videoInput.frameCount,
        timeNow=self.context.timeNow,
        tvec=tvec,
        rvec=rvec)
      #print poseRecord  # [debug]
      self.outFile.write(poseRecord)
      
      # Show pose values in GUI
      if self.showPoseWindow:
        self.imageOut.fill(255)
        cv2.putText(self.imageOut, "t: {vec[0]:+7.2f}, {vec[1]:+7.2f}, {vec[2]:+7.2f}".format(vec=tvec), (5, 21), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 100, 100), 1)
        cv2.putText(self.imageOut, "r: {vec[0]:+7.2f}, {vec[1]:+7.2f}, {vec[2]:+7.2f}".format(vec=rvec), (5, 41), cv2.FONT_HERSHEY_DUPLEX, 0.6, (100, 100, 200), 1)
        cv2.imshow(self.windowName, self.imageOut)
        cv2.waitKey(1)  # NOTE might slow things down
      
      self.lastFrameCount = self.context.videoInput.frameCount
