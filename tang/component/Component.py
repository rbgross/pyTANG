import xml.etree.ElementTree as ET

class Component:
  """Base class for all Actor components."""
  componentTypes = dict()
  
  @classmethod
  def registerType(cls, type):
    """Register a Component class for automatic delegation (no subclass lookup - explicit is better than implicit!)."""
    # NOTE Subclasses must be registered via this method in order to be initialized from XML, etc.
    cls.componentTypes[type.__name__] = type
    #print "Component.registerType(): Component type \'" + type.__name__ + "\' registered."  # [debug]
  
  @classmethod
  def fromXMLElement(cls, xmlElement, actor=None):
    """Create a Component instance from XML element."""
    # NOTE Subclasses should override this to extract relevant properties from xmlElement
    # Delegate component creation to appropriate subclass based on XML tag, return None when invalid
    try:
      return cls.componentTypes[xmlElement.tag].fromXMLElement(xmlElement, actor)
    except KeyError as e:
      print "Component.fromXMLElement(): Unregistered component type name \'" + xmlElement.tag + "\': " + str(e)
    except AttributeError as e:
      print "Component.fromXMLElement(): Invalid component type \'" + xmlElement.tag + "\' [fromXMLElement() method missing?]: " + str(e)
    return None
  
  def __init__(self, actor=None):
    self.actor = actor  # needed to maintain a link to containing actor
    # NOTE Subclasses should call Component.__init__(self, actor) first
  
  def toXMLElement(self):
    """Convert this instance to an XML element."""
    # NOTE Subclasses should call Component.toXMLElement(self) to obtain
    #   base node and then add further attributes and sub-elements
    return ET.Element(self.__class__.__name__)
  
  def toString(self, indent=""):
    """Return a brief string representation of this component object, with optional indentation."""
    return indent + self.__class__.__name__ + ": { }"
  
  def __str__(self):
    return self.toString()
