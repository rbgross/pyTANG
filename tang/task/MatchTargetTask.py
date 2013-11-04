import logging

import numpy as np
import cv2
import cv2.cv as cv

from task.Task import Task

class MatchTargetTask(Task):
  """Match movable cursor object with static target object."""
  
  def __init__(self):
    # NOTE Scene must be initialized at this point
    Task.__init__(self)
    
    # * Initialize variables
    self.cursor = None
    self.target = None
    self.showMatchWindow = True  # to be treated as a constant flag
    if self.showMatchWindow:
      self.imageOut = np.zeros((64, 128, 3), dtype=np.uint8)
      self.windowName = "Match error"
    
    # * Load scene fragments
    self.context.scene.readXML(self.context.getResourcePath('data', 'DragonScene.xml'))  # cursor
    self.context.scene.readXML(self.context.getResourcePath('data', 'StaticDragonScene.xml'))  # target
  
  def activate(self):
    self.cursor = self.context.scene.findActorById('cursor')
    self.target = self.context.scene.findActorById('target')
    if self.cursor is None or self.target is None:
      self.logger.warn("[MatchTargetTask] Cursor or target object not found; task could not be initialized")
      return
    self.cursor.visible = True
    self.target.visible = True
    if self.showMatchWindow:
      self.imageOut.fill(255)
      cv2.imshow(self.windowName, self.imageOut)
      cv2.waitKey(1)
    Task.activate(self)
  
  def deactivate(self):
    Task.deactivate(self)
    self.cursor.visible = False
    self.target.visible = False
    if self.showMatchWindow:
      cv2.destroyWindow(self.windowName)
  
  def update(self):
    # Check if the transforms on cursor and target object are close enough
    rmat_diff = self.target.transform[0:3, 0:3] - self.cursor.transform[0:3, 0:3]
    #self.logger.info("Rotation difference matrix:-\n{}".format(rmat_diff))  # [debug]
    rmat_sumAbsDiff = np.sum(np.abs(rmat_diff))
    #self.logger.debug("Rotation, sum of absolute differences: {}".format(rmat_sumAbsDiff)) 
    # TODO use cv2.Rodrigues to compute individual x, y, z rotation components and then compare
    tvec_diff = self.target.transform[0:3, 3] - self.cursor.transform[0:3, 3]
    #self.logger.info("Translation difference vector:-\n{}".format(tvec_diff))  # [debug]
    tvec_normDiff = np.linalg.norm(tvec_diff, ord=2)
    #self.logger.debug("Translation, L2-norm of differences  : {}".format(tvec_normDiff))
    
    if self.showMatchWindow:
      self.imageOut.fill(255)
      cv2.putText(self.imageOut, "t: {:5.2f}".format(tvec_normDiff), (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 100, 100), 2)
      cv2.putText(self.imageOut, "r: {:5.2f}".format(rmat_sumAbsDiff), (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 200), 2)
      if rmat_sumAbsDiff <= 0.1 and tvec_normDiff <= 1.0:
        cv2.rectangle(self.imageOut, (2, 2), (self.imageOut.shape[1] - 3, self.imageOut.shape[0] - 3), (100, 200, 100), 3)
      cv2.imshow(self.windowName, self.imageOut)
      cv2.waitKey(1)  # NOTE might slow things down
