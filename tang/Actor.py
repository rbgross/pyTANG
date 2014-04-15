import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import numpy as np
import hommat as hm
import xml.etree.ElementTree as ET

from component.Component import Component

class Actor:
  @classmethod
  def fromXMLElement(cls, xmlElement, actorFactory):
    # Initialize empty actor object
    actor = actorFactory.makeEmpty()
    
    # Extract common Actor attributes such as id, description, parent
    actor.id = xmlElement.get('id', None)
    actor.description = xmlElement.get('description', None)
    actor.parent = xmlElement.get('parent', None)
    
    # Load actor's components
    components = xmlElement.find('components')
    for component in components:
      componentObj = Component.fromXMLElement(component, actor)
      if componentObj is not None:
        actor.components[componentObj.__class__.__name__] = componentObj
    
    # Recursively load child actors
    for child in xmlElement.find('children'):
      if child.tag == 'Actor':
        actor.children.append(cls.fromXMLElement(child, actorFactory))
    
    return actor
  
  def __init__(self, renderer, isTransient=False):
    self.renderer = renderer  # TODO make renderer a component?
    self.isTransient = isTransient  # if transient, will not be exported (in XML, etc.)
    
    # Common attributes
    self.id = None
    self.description = None
    self.parent = None
    
    # Dictionary to store all components by name
    self.components = dict()
    
    # List of child actors
    self.children = []
    
    # TODO Consider a finalize() method that copies out references to key components for fast retrieval,
    #   and possibly pre-computes some results (such as a composite transform matrix)
    
    # Set visibility to True upon initialization
    self.visible = True  # only affects this actor, not its children
    
    # Temp variables that are useful to retain
    self.transform = hm.identity()
  
  def draw(self, baseTransform=hm.identity()):
    # TODO draw and recurse down only when this Actor is enabled
    # TODO move to a more generic approach?
    #   e.g.:- for component in self.components: component.apply()
    #   But how do we ensure order is maintained? (Mesh must be rendered after applying Transform and Material) OrderedDict?
    self.transform = baseTransform
    try:
      if hasattr(self, 'transform_matrix'):  # if there is a full transform, use it
        self.transform = np.dot(self.transform, self.transform_matrix)
      else:
        self.transform = hm.translation(self.transform, self.components['Transform'].translation)  # TODO make transform relative to parent, not absolute
        self.transform = hm.rotation(self.transform, self.components['Transform'].rotation[0], [1, 0, 0])
        self.transform = hm.rotation(self.transform, self.components['Transform'].rotation[1], [0, 1, 0])
        self.transform = hm.rotation(self.transform, self.components['Transform'].rotation[2], [0, 0, 1])
        self.transform = hm.scale(self.transform, self.components['Transform'].scale)
    except KeyError, AttributeError:
      # Transform component not present or incomplete/invalid
      pass  # use base (parent) transform (?) - should get set in next step
    
    # Render this actor, if visible
    if self.visible:
      try:
        self.renderer.setModelMatrix(self.transform)
        self.renderer.setDiffCol(self.components['Material'].color)
        # TODO Check if a mesh component is attached, and only then render it
        self.components['Mesh'].render()
      except KeyError:
        # No Material and/or Mesh, silently ignore (TODO: Make this more efficient since we'll have a lot of Empty actors)
        pass
    
    for child in self.children:
      child.draw(self.transform)  # TODO do not draw if not enabled (e.g. an invisible actor should be enabled, but not visible)
  
  def findActorByComponent(self, componentName):
    if componentName in self.components:
      return self
    else:
      for child in self.children:
        targetActor = child.findActorByComponent(componentName)
        if targetActor is not None:
          return targetActor
    return None
  
  def findActorsByComponent(self, componentName):
    """Use generator pattern to return all actors under this hierarchy (including self) that have a given component."""
    if componentName in self.components:  # TODO: we should store the actual component classes as keys to allow subclass matching
      yield self
    for child in self.children:
      for matchingActor in child.findActorsByComponent(componentName):
        yield matchingActor
  
  def toXMLElement(self):
    xmlElement = ET.Element(self.__class__.__name__)
    # TODO: Add common Actor attributes
    
    componentsElement = ET.SubElement(xmlElement, 'components')
    for component in self.components.itervalues():
      componentsElement.append(component.toXMLElement())
    
    childrenElement = ET.SubElement(xmlElement, 'children')
    for child in self.children:
      if not child.isTransient:  # only export non-transient children
        childrenElement.append(child.toXMLElement())
    
    return xmlElement
  
  def toString(self, indent=""):
    # TODO: Make this more efficient and automatic!
    out = indent + "Actor: {\n"
    if self.id is not None:
      out += indent + "  id: \"{}\"".format(self.id)
      out += ",\n" if self.description is not None or self.parent is not None or self.components or self.children else "\n"
    if self.description is not None:
      out += indent + "  description: \"{}\"".format(self.description)
      out += ",\n" if self.parent is not None or self.components or self.children else "\n"
    if self.parent is not None:
      out += indent + "  parent: \"{}\"".format(self.parent)
      out += ",\n" if self.components or self.children else "\n"
    if self.components:
      out += indent + "  components: {\n"
      out += ",\n".join(component.toString(indent + "    ") for component in self.components.itervalues()) + "\n"
      out += indent + "  }"
      out += ",\n" if self.children else "\n"
    if self.children:
      out += indent + "  children: {\n"
      out += ",\n".join(child.toString(indent + "    ") for child in self.children if not child.isTransient) + "\n"  # only show non-transient children
      out += indent + "  }\n"
    out += indent + "}"
    
    return out
  
  def __str__(self):
    return self.toString()
