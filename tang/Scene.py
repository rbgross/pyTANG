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
        self.actorsById = dict()
        #self.readData(self.context.getResourcePath('data', 'PerspectiveScene.txt'))  # deprecated; load from XML instead
        #self.writeXML(filename + '.xml')  # enable this to convert a .txt scene file to XML (NOTE will write to file in <resources>/data/!)
        
        #self.readXML(self.context.getResourcePath('data', 'default-scene.xml'))  # the cube and a dragon
        self.readXML(self.context.getResourcePath('data', 'CubeScene.xml'))  # just the cube
        #self.readXML(self.context.getResourcePath('data', 'BP3D-FMA7088-heart.xml'))  # BodyParts3D heart model hierarchy
        #self.readXML(self.context.getResourcePath('data', 'RadialTreeScene.xml'))
        self.readXML(self.context.getResourcePath('data', 'DragonScene.xml'))  # cursor
        self.readXML(self.context.getResourcePath('data', 'StaticDragonScene.xml'))  # target
        #self.readXML(self.context.getResourcePath('data', 'PerspectiveScene.xml'))
        
        self.finalize()  # should be called after all read*() methods have been called
        print "Scene.__init__(): Scene has {} top-level actor(s)".format(len(self.actors))
        #self.dump()  # [debug]
    
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
                    if actor.id is not None:
                        self.actorsById[actor.id] = actor
        else:
            print "Scene.readXML(): Bad XML: Root must be a <scene> element"
    
    def finalize(self):
        """Finalize scene hierarchy by resolving parent-child relations (for current top-level actors only)."""
        
        if 'cube' not in self.actorsById:
            print "Scene.finalize(): WARNING: No cube in scene!"
        
        topLevel = []
        for actor in self.actors:
            if actor.parent is not None:
                try:
                    self.actorsById[actor.parent].children.append(actor)
                    continue  # actor should no longer be kept in top-level
                except KeyError:
                    print "Scene.finalize(): WARNING: Parent id \"{}\" not found!".format(actor.parent)
            topLevel.append(actor)
        self.actors = topLevel
    
    def writeXML(self, filename):
        xmlTree = ET.ElementTree(ET.Element('scene'))
        rootElement = xmlTree.getroot()
        for actor in self.actors:
            rootElement.append(actor.toXMLElement())
        xmlTree.write(open(self.context.getResourcePath('data', filename), 'w'))
        print "Scene.writeXML(): Written to file:", filename
    
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
    
    def dump(self):
        print "Scene.dump(): Actors:-"
        for actor in self.actors:
            print actor
