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

from component.Mesh import Mesh
from component.Transform import Transform
from component.Material import Material

class Actor:
  @classmethod
  def fromXMLElement(cls, xmlElement, actorFactory):
    # Initialize empty actor object
    actor = actorFactory.makeEmpty()
    
    # TODO Extract common element attributes such as id
    
    # Load actor's components
    components = xmlElement.find('components')
    for component in components:
      if component.tag == 'Mesh':
        # NOTE src is a required attribute of Mesh elements
        actor.components['Mesh'] = Mesh.fromXMLElement(component)  # NOTE Meshes are currently shared, therefore not linked to individual actors
      elif component.tag == 'Transform':
        actor.components['Transform'] = Transform.fromXMLElement(component, actor)
      elif component.tag == 'Material':
        actor.components['Material'] = Material.fromXMLElement(component, actor)
    
    # Recursively load child actors
    for child in xmlElement.find('children'):
      if child.tag == 'Actor':
        actor.children.append(cls.fromXMLElement(child, actorFactory))
    
    return actor
  
  def __init__(self, renderer, environment):
    self.renderer = renderer  # TODO make renderer a component (?) and remove dependency on environment
    self.environment = environment
    
    # Dictionary to store all components by name
    self.components = dict()
    
    # List of child actors
    self.children = []
  
  def draw(self):
    # TODO move to a more generic approach?
    #   e.g.:- for component in self.components: component.apply()
    #   But how do we ensure order is maintained? (Mesh must be rendered after Transform and Material have been applied)
    try:
      model = hm.translation(self.environment.model, self.components['Transform'].translation)  # TODO make transform relative to parent, not absolute
      model = hm.rotation(model, self.components['Transform'].rotation[0], [1, 0, 0])
      model = hm.rotation(model, self.components['Transform'].rotation[1], [0, 1, 0])
      model = hm.rotation(model, self.components['Transform'].rotation[2], [0, 0, 1])
      # TODO scale
      self.renderer.setModel(model)
    except KeyError:
      # No Transform component present, use global transform?
      self.renderer.setModel(environment.model)
    
    try:
      self.renderer.setDiffCol(self.components['Material'].color)
      self.components['Mesh'].render()
    except KeyError:
      # No Material and/or Mesh, silently ignore (TODO: Make this more efficient since we'll have a lot of Empty actors)
      pass
    
    for child in self.children:
        child.draw() # TODO implement ability to show/hide actors and/or children
  
  def toXMLElement(self):
    xmlElement = ET.Element(self.__class__.__name__)
    
    componentsElement = ET.SubElement(xmlElement, 'components')
    for component in self.components.itervalues():
      componentsElement.append(component.toXMLElement())
    
    childrenElement = ET.SubElement(xmlElement, 'children')
    for child in self.children:
      childrenElement.append(child.toXMLElement())
    
    return xmlElement
  
  def toString(self, indent=""):
    out = indent + "Actor: {\n"
    out += indent + "  components: {\n"
    if 'Mesh' in self.components:
      out += indent + "    " + str(self.components['Mesh']) + ",\n"
    if 'Transform' in self.components:
      out += indent + "    Transform: {\n"
      out += indent + "      translation: " + str(self.components['Transform'].translation) + ",\n"
      out += indent + "      rotation: " + str(self.components['Transform'].rotation) + ",\n"
      out += indent + "      scale: " + str(self.components['Transform'].scale) + "\n"
      out += indent + "    },\n"
    if 'Material' in self.components:
      out += indent + "    Material: {\n"
      out += indent + "      color: " + str(self.components['Material'].color) + "\n"
      out += indent + "    }\n"
    out += indent + "  },\n"
    out += indent + "  children: {\n"
    for child in self.children:
        out += child.toString(indent + "    ")
    out += indent + "  }\n"
    out += indent + "}"
    
    return out
  
  def __str__(self):
    return self.toString()
