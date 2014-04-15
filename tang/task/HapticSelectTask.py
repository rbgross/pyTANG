import numpy as np

from task.Task import Task
from tool.HapticPointer import HapticPointer

class HapticSelectTask(Task):
  """Use a haptic pointing device to select interactive objects."""
  
  feedback_gain = 0.5  # refer to OpenHaptics HLAPI docs (HL_EFFECT_PROPERTY_GAIN)
  feedback_magnitude = 0.6  # strength of haptic feedback (HL_EFFECT_PROPERTY_MAGNITUDE)
  
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
    if self.pointer.valid:
      # Update pointer actor's position and orientation in scene (TODO: transform coordinates to align reference frames so that pointer is physically and virtually co-located)
      self.pointerActor.transform_matrix = self.pointer.transform  # use transform directly
    
    if True:  # [debug: enables selection checking even when there is no valid haptic device connected - useful for development and testing, harmless when live]
      # Check for collisions with Collider components
      # TODO: Enhancements (mostly Collider improvements):
      #       1. Generalize to other Collider types; currently supports SphereCollider only (and that too with radius correctly set)
      #          This will only be possible after certain improvements in the scene graph, esp. to query for actors with a certain base component (Collider)
      #       2. Allow a single Collider to represent a whole hierarchy (if indicated), utilizing a bounding box for all meshes by default
      # TODO: Optimizations:
      #       1. Walk once when scene is finalized and cache all collidable actors, their colliders
      #       2. Use a spatial index to speed up matching (otherwise this is very slow)
      # * Walk the scene hierarchy looking for actors with Colliders
      for actor in self.context.scene.findActorsByComponent('SphereCollider'):
        try:
          # ** Check if pointer is touching the actor, if so highlight it (also manage highlight removal)
          dist = np.linalg.norm(actor.transform[:3, 3] - self.pointerActor.transform[:3, 3], ord=2)
          isColliding = dist < actor.components['SphereCollider'].radius
          #self.logger.debug("Actor: %s (%s), distance: %7.3f, colliding? %s", actor.id, actor.description, dist, isColliding)  # [verbose]
          if isColliding and not actor.components['SphereCollider'].isHighlighted:
            actor.components['SphereCollider'].set_highlight(True)
            self.logger.info("Touched %s (%s)! Distance: %.3f, time: %.3f", actor.id, actor.description, dist, self.context.timeNow)  # [debug]
            # Generate haptic feedback (TODO: vary magnitude by distance?)
            self.pointer.setFeedback("on", self.feedback_gain, self.feedback_magnitude)
          elif not isColliding and actor.components['SphereCollider'].isHighlighted:
            actor.components['SphereCollider'].set_highlight(False)
            self.pointer.setFeedback("off")  # default off
        except AttributeError as e:
          self.logger.error("Couldn't compute distance to actor: %s", e)
