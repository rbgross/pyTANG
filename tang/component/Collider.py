import numpy as np

from Actor import Actor
from Component import Component

class Collider(Component):
  """Base class for collider objects that can be used to test for collisions."""
  # NOTE: This class is not registered with Component as it is not meant to be instantiated
  pass


class SphereCollider(Collider):
  default_radius = 1.0
  
  @classmethod
  def fromXMLElement(cls, xmlElement, actor=None):
    radius = xmlElement.get('radius')
    highlight = xmlElement.get('highlight')  # color to use when collisions occur, no change if None
    return SphereCollider(
      float(radius) if radius is not None else cls.default_radius,
      np.fromstring(highlight, dtype=np.float32, sep=' ') if highlight is not None else None,
      actor)
  
  def __init__(self, radius=default_radius, highlight=None, actor=None):
    Component.__init__(self, actor)
    self.radius = radius
    self.highlight = highlight
    
    self.isHighlighted = False  # a runtime property indicating whether this actor should be highlighted (due to a collision)
    self.normalColor = None  # will be set on first highlight
  
  def set_highlight(self, on=True):
    self.isHighlighted = on
    if self.actor is not None and 'Material' in self.actor.components:
      if self.normalColor is None:
        self.normalColor = self.actor.components['Material'].color
      self.actor.components['Material'].color = self.highlight \
            if (self.isHighlighted and self.highlight is not None) \
            else self.normalColor
  
  def toXMLElement(self):
      xmlElement = Component.toXMLElement(self)
      xmlElement.set('radius', str(self.radius))
      if self.highlight is not None:
        xmlElement.set('highlight', str(self.highlight))
      return xmlElement
  
  def toString(self, indent=""):
      return indent + "SphereCollider: { radius: " + str(self.radius) + \
            ((", highlight: " + str(self.highlight)) if self.highlight is not None else "") + \
            " }"
  
  def __str__(self):
      return self.toString()

Component.registerType(SphereCollider)
