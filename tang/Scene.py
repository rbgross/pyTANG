import sys
import os
import time
import numpy as np
import logging
from ctypes import *
import xml.etree.ElementTree as ET

from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
import hommat as hm

from Context import Context
from Renderer import Renderer
from Light import Light
from Actor import Actor
from ActorFactory import ActorFactory

class Scene:
    def __init__(self):
        self.context = Context.getInstance()  # NOTE must contain renderer
        assert hasattr(self.context, 'renderer') and isinstance(self.context.renderer, Renderer), "Context does not contain renderer of correct type"
        
        self.logger = logging.getLogger(__name__)
        self.hideCube = False
        self.transform = hm.identity()
        self.light = Light(self.context.renderer)
        
        self.actorFactory = ActorFactory(self.context.renderer)
        self.actors = []
        self.actorsById = dict()
        
        #self.readXML(self.context.getResourcePath('data', 'default-scene.xml'))  # the cube and a dragon
        #self.readData(self.context.getResourcePath('data', 'PerspectiveScene.txt'))  # deprecated; load from XML instead
        #self.writeXML(filename + '.xml')  # enable this to convert a .txt scene file to XML (NOTE will write to file in <resources>/data/!)
    
    def readData(self, filename):
        self.logger.info("Parsing file: {}".format(filename))
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
        self.logger.info("Parsing file: {}".format(filename))
        xmlTree = ET.parse(filename)
        rootElement = xmlTree.getroot()
        #print "Scene.readXML(): Source XML tree:-"  # [debug]
        #ET.dump(rootElement)  # [debug]
        
        # Root element must be a <scene>, containing one or more <Actor> elements
        # TODO Define shared resources (shaders, meshes, etc.) in a <defs> section (and then actors in <actors> or <group> sections?)
        if rootElement.tag == 'scene':
            for subElement in rootElement:
                if subElement.tag == 'Actor':
                    actor = Actor.fromXMLElement(subElement, self.actorFactory)
                    self.actors.append(actor)  # include in scene hierarchy
                    if actor.id is not None:
                        self.actorsById[actor.id] = actor
                    for child in actor.children:
                        if child.id is not None:
                            self.actorsById[child.id] = child  # grab child actors as well
        else:
            self.logger.warn("Bad XML: Root must be a <scene> element!")
    
    def finalize(self):
        """Finalize scene hierarchy by resolving parent-child relations (for current top-level actors only)."""
        
        if 'cube' not in self.actorsById:
            self.logger.warn("No cube in scene!")
        
        topLevel = []
        for actor in self.actors:
            if actor.parent is not None:
                try:
                    self.actorsById[actor.parent].children.append(actor)
                    continue  # actor should no longer be kept in top-level
                except KeyError:
                    self.logger.warn("Parent id \"{}\" not found!".format(actor.parent))
            topLevel.append(actor)
        self.actors = topLevel
        
        self.logger.info("Scene has {} top-level actor(s)".format(len(self.actors)))
        #self.dump()  # [debug]
    
    def writeXML(self, filename):
        xmlTree = ET.ElementTree(ET.Element('scene'))
        rootElement = xmlTree.getroot()
        for actor in self.actors:
            rootElement.append(actor.toXMLElement())
        xmlTree.write(open(self.context.getResourcePath('data', filename), 'w'))
        self.logger.info("Written to file: {}".format(filename))
    
    def draw(self):
        for actor in self.actors:
            if actor.id == 'cube':
                if self.hideCube: continue  # skip rendering the cube
                actor.draw(self.transform)  # TODO render different trackable objects with their respective tracked transforms
            else:
                actor.draw()  # render static objects with implicit identity transform
    
    def findActorById(self, id):
        return self.actorsById.get(id, None)
    
    def findActorByComponent(self, componentName):
        for actor in self.actors:
            targetActor = actor.findActorByComponent(componentName)
            if targetActor is not None:
                return targetActor
        return None
    
    def findActorsByComponent(self, componentName):
        for actor in self.actors:
            for matchingActor in actor.findActorsByComponent(componentName):
                yield matchingActor
    
    def dump(self):
        print "Scene.dump(): Actors:-"
        for actor in self.actors:
            print actor
