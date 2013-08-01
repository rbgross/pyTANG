import numpy as np

from Component import Component

class Transform(Component):
  @classmethod
  def fromXMLElement(cls, xmlElement, actor=None):
    translation = xmlElement.get('translation')
    rotation = xmlElement.get('rotation')
    scale = xmlElement.get('scale')
    return Transform(
      np.fromstring(translation, dtype=np.float32, sep=' ') if translation is not None else np.float32([0.0, 0.0, 0.0]),
      np.fromstring(rotation, dtype=np.float32, sep=' ') if rotation is not None else np.float32([0.0, 0.0, 0.0]),
      np.fromstring(scale, dtype=np.float32, sep=' ') if scale is not None else np.float32([1.0, 1.0, 1.0]),
      actor)
  
  def __init__(self, translation=np.float32([0.0, 0.0, 0.0]), rotation=np.float32([0.0, 0.0, 0.0]), scale=np.float32([1.0, 1.0, 1.0]), actor=None):
    Component.__init__(self, actor)
    
    self.translation = translation
    self.rotation = rotation
    self.scale = scale
  
  def toXMLElement(self):
      xmlElement = Component.toXMLElement(self)
      xmlElement.set('translation', str(self.translation).strip('[ ]'))
      xmlElement.set('rotation', str(self.rotation).strip('[ ]'))
      xmlElement.set('scale', str(self.scale).strip('[ ]'))
      return xmlElement
  
  def toString(self, indent=""):
      return indent + "Transform: {\n" + \
             indent + "  translation: " + str(self.translation) + ",\n" + \
             indent + "  rotation: " + str(self.rotation) + ",\n" + \
             indent + "  scale: " + str(self.scale) + "\n" + \
             indent + "}"
  
  def __str__(self):
      return self.toString()

# Register component type for automatic delegation (e.g. when inflating from XML)
Component.registerType(Transform)
