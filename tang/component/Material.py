import numpy as np

from Component import Component

class Material(Component):
  @classmethod
  def fromXMLElement(cls, xmlElement, actor=None):
    color = xmlElement.get('color')
    return Material(
      np.fromstring(color, dtype=np.float32, sep=' ') if color is not None else np.float32([0.0, 0.0, 0.0]),
      actor)
  
  def __init__(self, color=np.float32([0.0, 0.0, 0.0]), actor=None):
    Component.__init__(self, actor)
    self.color = color
  
  def toXMLElement(self):
      xmlElement = Component.toXMLElement(self)
      xmlElement.set('color', str(self.color).strip('[ ]'))
      return xmlElement
  
  def toString(self, indent=""):
      return indent + "Material: { color: " + str(self.color) + " }"
  
  def __str__(self):
      return self.toString()

# Register component type for automatic delegation (e.g. when inflating from XML)
Component.registerType(Material)
