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

from Light import Light
from ActorFactory import ActorFactory
from component.Mesh import Mesh
from component.Transform import Transform
from component.Material import Material

class Environment:
    def __init__(self, renderer):
        self.hideCube = False
        self.renderer = renderer
        self.model = hm.identity()
        self.light = Light(self.renderer)
        
        self.actorFactory = ActorFactory(self.renderer, self)
        self.actors = []
        self.readData(os.path.abspath(os.path.join(self.renderer.resPath, 'data', 'PerspectiveScene.txt')))
        self.readXML(os.path.abspath(os.path.join(self.renderer.resPath, 'data', 'default-scene.xml')))
    
    def readData(self, fileName):
        f = open(fileName, 'r')
        for line in f: 
            s = line.split()
            if len(s) > 0:
                if s[0] == 'Cube':
                    actor = self.actorFactory.makeCube()
                if s[0] == 'Edge':
                    actor = self.actorFactory.makeEdge()
                if s[0] == 'DataPoint':
                    actor = self.actorFactory.makeDataPoint()
                elif s[0] == 'Dragon':
                    actor = self.actorFactory.makeDragon()
                actor.components['Transform'].position = np.array([s[2], s[3], s[4]], dtype = np.float32)
                actor.components['Transform'].rotation = np.array([s[6], s[7], s[8]], dtype = np.float32)
                actor.components['Material'].color = np.array([s[10], s[11], s[12]], dtype = np.float32)
                self.actors.append(actor)
    
    def readXML(self, filename):
        xmlTree = ET.parse(filename)
        rootNode = xmlTree.getroot()
        print "Environment.readXML(): XML Tree:-"
        ET.dump(rootNode)
        #self.printXMLNode(rootNode)  # recursively prints this node and all children
        
        if rootNode.tag == 'Actor':
            self.rootActor = self.loadActorFromXMLNode(rootNode)
            print "\nEnvironment.readXML(): Loaded actor hierarchy:-"
            print self.rootActor
            self.actors.append(self.rootActor)  # include in scene hierarchy
    
    def loadActorFromXMLNode(self, xmlNode):
        # Initialize empty actor object
        actor = self.actorFactory.makeEmpty()
        
        # Load actor's components
        components = xmlNode.find('components')
        for component in components:
            if component.tag == 'Mesh':
                # NOTE src is a required attribute of Mesh elements
                actor.components['Mesh'] = Mesh.fromXMLElement(component)  # NOTE Meshes are currently shared, therefore not linked to individual actors
            elif component.tag == 'Transform':
                actor.components['Transform'] = Transform.fromXMLElement(component, actor)
            elif component.tag == 'Material':
                actor.components['Material'] = Material.fromXMLElement(component, actor)
        
        # Recursively load child actors
        for child in xmlNode.find('children'):
            if child.tag == 'Actor':
                actor.children.append(self.loadActorFromXMLNode(child))
        
        return actor
    
    def printXMLNode(self, xmlNode, indent=""):
        print indent, xmlNode.tag, xmlNode.attrib
        for childNode in xmlNode:
            self.printXMLNode(childNode, indent + "  ")
    
    def draw(self):
        if not self.hideCube:
            for i in xrange(0, 8):
                self.actors[i].draw()
        
        for i in xrange(8, len(self.actors)):
            self.actors[i].draw()
