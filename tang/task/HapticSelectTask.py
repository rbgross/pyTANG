import numpy as np

from task.Task import Task
from tool.HapticPointer import HapticPointer

class HapticSelectTask(Task):
  """Use a haptic pointing device to select interactive objects."""
  
  def __init__(self):
    Task.__init__(self)
    
    # * Initialize variables
    self.pointer = HapticPointer()  # TODO create tools directly in Main, just like tasks, and find them here?
    
    # * Load scene fragments
    self.context.scene.readXML(self.context.getResourcePath('data', 'PointerScene.xml'))
    # TODO Add proper pointer model, edit PointerScene.xml; move tool model loading to inside Tool classes?
  
  def activate(self):
    # * Find pointer actor in scene, reposition it and make it visible
    self.pointerActor = self.context.scene.findActorById('pointer')
    if self.pointerActor is None:
      self.logger.warn("[HapticSelectTask] Pointer object not found in scene; task could not be initialized")
      return
    self.pointerActor.components['Transform'].translation = np.float32([0, 0, 80])
    self.pointerActor.visible = True
    Task.activate(self)
  
  def deactivate(self):
    Task.deactivate(self)
    self.pointerActor.visible = False
    self.pointer.close()
  
  def update(self):
    # TODO Update pointer actor's position and orientation in scene, transform coordinates to align reference frames
    if self.pointer.valid:
        self.pointerActor.components['Transform'].translation = np.float32(self.pointer.position)  # TODO: use transform directly
