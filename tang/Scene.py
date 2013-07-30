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

from Context import Context
from Light import Light
from Actor import Actor
from ActorFactory import ActorFactory

class Scene:
    def __init__(self):
        self.context = Context.getInstance()  # NOTE must contain renderer
        
        self.hideCube = False
        self.transform = hm.identity()
        self.light = Light(self.context.renderer)
        
        self.actorFactory = ActorFactory(self.context.renderer)
        self.actors = []
        #self.readData(self.context.getResourcePath('data', 'PerspectiveScene.txt'))  # deprecated; load from XML instead
        #self.writeXML(filename + '.xml')  # enable this to convert a .txt scene file to XML (NOTE will write to file in <resources>/data/!)
        self.readXML(self.context.getResourcePath('data', 'PerspectiveScene.xml'))
        self.readXML(self.context.getResourcePath('data', 'default-scene.xml'))
        print "Scene.__init__(): Loaded {} top-level actor(s)".format(len(self.actors))
        #self.dump()
    
    def readData(self, filename):
        print "Scene.readData(): Parsing file:", filename
        f = open(filename, 'r')
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
                actor.components['Transform'].translation = np.array([s[2], s[3], s[4]], dtype = np.float32)
                actor.components['Transform'].rotation = np.array([s[6], s[7], s[8]], dtype = np.float32)
                actor.components['Material'].color = np.array([s[10], s[11], s[12]], dtype = np.float32)
                self.actors.append(actor)
    
    def readXML(self, filename):
        print "Scene.readXML(): Parsing file:", filename
        xmlTree = ET.parse(filename)
        rootElement = xmlTree.getroot()
        #print "Scene.readXML(): Source XML tree:-"
        #ET.dump(rootElement)
        
        # Root element must be a <scene>, containing one or more <Actor> elements
        # TODO Define shared resources (shaders, meshes, etc.) in a <defs> section (and then actors in <actors> or <group> sections?)
        if rootElement.tag == 'scene':
            for subElement in rootElement:
                if subElement.tag == 'Actor':
                    actor = Actor.fromXMLElement(subElement, self.actorFactory)
                    self.actors.append(actor)  # include in scene hierarchy
        else:
            print "Scene.readXML(): Bad XML: Root must be a <scene> element"
    
    def writeXML(self, filename):
        xmlTree = ET.ElementTree(ET.Element('scene'))
        rootElement = xmlTree.getroot()
        for actor in self.actors:
            rootElement.append(actor.toXMLElement())
        xmlTree.write(open(self.context.getResourcePath('data', filename), 'w'))
        print "Scene.writeXML(): Written to file:", filename
    
    def draw(self):
        for actor in self.actors:
            actor.draw(self.transform)
        
        '''if not self.hideCube:
            for i in xrange(0, 8):
                self.actors[i].draw(self.transform)
        
        for i in xrange(8, len(self.actors)):
            self.actors[i].draw(self.transform)'''
    
    def dump(self):
        print "Scene.dump(): Actors:-"
        for actor in self.actors:
            print actor