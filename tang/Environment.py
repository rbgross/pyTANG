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

class Environment:
    def __init__(self):
        self.context = Context.getInstance()  # NOTE must contain renderer
        
        self.hideCube = False
        self.model = hm.identity()
        self.light = Light(self.context.renderer)
        
        self.actorFactory = ActorFactory(self.context.renderer, self)
        self.actors = []
        self.readData(self.context.getResourcePath('data', 'PerspectiveScene.txt'))
        self.readXML(self.context.getResourcePath('data', 'default-scene.xml'))
    
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
        rootElement = xmlTree.getroot()
        print "Environment.readXML(): XML Tree:-"
        ET.dump(rootElement)
        #self.printXMLElement(rootElement)  # recursively prints this element and all children
        
        if rootElement.tag == 'Actor':
            self.rootActor = Actor.fromXMLElement(rootElement, self.actorFactory)
            print "\nEnvironment.readXML(): Loaded actor hierarchy:-"
            print self.rootActor
            self.actors.append(self.rootActor)  # include in scene hierarchy
    
    def printXMLElement(self, xmlElement, indent=""):
        print indent, xmlElement.tag, xmlElement.attrib
        for subElement in xmlElement:
            self.printXMLElement(subElement, indent + "  ")
    
    def draw(self):
        if not self.hideCube:
            for i in xrange(0, 8):
                self.actors[i].draw()
        
        for i in xrange(8, len(self.actors)):
            self.actors[i].draw()
