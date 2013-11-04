import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import math
import time
import numpy as np
import cv2
import hommat as hm

from Context import Context

class Controller:
    input_snapshot_file = "../out/snapshot-input.png"  # file path to store snapshot of input camera image
    
    def __init__(self):
        self.context = Context.getInstance()  # NOTE must contain scene
        
        self.timer = time.clock()
        
        #self.wheelPosition = glfw.GetMouseWheel()
        self.oldMouseX, self.oldMouseY = glfw.GetMousePos()
        self.curMouseX = self.oldMouseX
        self.curMouseY = self.oldMouseY
        self.leftPressed = False
        self.rightPressed = False
        
        self.manualControl = False
        
        self.doQuit = False  # flag to signal stop condition
        self.quitting = False  # to prevent multiple key-presses

    def pollInput(self):
        currentTime = time.clock()
        elapsedTime = currentTime - self.timer
        self.timer = currentTime
        
        #tempWheelPosition = glfw.GetMouseWheel()
        #if tempWheelPosition != self.wheelPosition:
            #self.wheelPosition = tempWheelPosition
            #self.setView(hm.lookat(hm.identity(), np.array([0.0, 0.0, 55.0 - self.wheelPosition, 1.0], dtype = np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype = np.float32)))

        if glfw.GetKey('M'):
            print "Initializing manual control"
            self.manualControl = True
            self.context.scene.transform = hm.translation(hm.identity(), [0, 0, 60])
            mouseX, mouseY = glfw.GetMousePos()
            self.calcArcBallVector(mouseX, mouseY)
            time.sleep(0.5)  # TODO prevent multiple key-presses properly

        if glfw.GetKey('P'):
            print "Stopping manual control"
            self.manualControl = False
            time.sleep(0.5)  # TODO prevent multiple key-presses properly
        
        if glfw.GetKey('A') and self.manualControl:
            self.context.scene.transform = np.dot(hm.translation(hm.identity(), [0, 0, -60]), self.context.scene.transform)
            self.context.scene.transform = np.dot(hm.rotation(hm.identity(), 60 * elapsedTime, [0, 1, 0]), self.context.scene.transform)
            self.context.scene.transform = np.dot(hm.translation(hm.identity(), [0, 0, 60]), self.context.scene.transform)

        if glfw.GetKey('D') and self.manualControl:
            self.context.scene.transform = np.dot(hm.translation(hm.identity(), [0, 0, -60]), self.context.scene.transform)
            self.context.scene.transform = np.dot(hm.rotation(hm.identity(), -60 * elapsedTime, [0, 1, 0]), self.context.scene.transform)
            self.context.scene.transform = np.dot(hm.translation(hm.identity(), [0, 0, 60]), self.context.scene.transform)

        if glfw.GetKey('W') and self.manualControl:
            self.context.scene.transform = np.dot(hm.translation(hm.identity(), [0, 0, -60]), self.context.scene.transform)
            self.context.scene.transform = np.dot(hm.rotation(hm.identity(), -60 * elapsedTime, [1, 0, 0]), self.context.scene.transform)
            self.context.scene.transform = np.dot(hm.translation(hm.identity(), [0, 0, 60]), self.context.scene.transform)

        if glfw.GetKey('S') and self.manualControl:
            self.context.scene.transform = np.dot(hm.translation(hm.identity(), [0, 0, -60]), self.context.scene.transform)
            self.context.scene.transform = np.dot(hm.rotation(hm.identity(), 60 * elapsedTime, [1, 0, 0]), self.context.scene.transform)
            self.context.scene.transform = np.dot(hm.translation(hm.identity(), [0, 0, 60]), self.context.scene.transform)

        if glfw.GetKey('Q') and self.manualControl:
            self.context.scene.transform = np.dot(hm.translation(hm.identity(), [0, 0, -60]), self.context.scene.transform)
            self.context.scene.transform = np.dot(hm.rotation(hm.identity(), -60 * elapsedTime, [0, 0, 1]), self.context.scene.transform)
            self.context.scene.transform = np.dot(hm.translation(hm.identity(), [0, 0, 60]), self.context.scene.transform)

        if glfw.GetKey('E') and self.manualControl:
            self.context.scene.transform = np.dot(hm.translation(hm.identity(), [0, 0, -60]), self.context.scene.transform)
            self.context.scene.transform = np.dot(hm.rotation(hm.identity(), 60 * elapsedTime, [0, 0, 1]), self.context.scene.transform)
            self.context.scene.transform = np.dot(hm.translation(hm.identity(), [0, 0, 60]), self.context.scene.transform)

        if glfw.GetKey('1'):
            print "1 pressed"
        
        if glfw.GetKey('2'):
            print "2 pressed"
        
        if glfw.GetKey('3'):
            print "3 pressed"
        
        if glfw.GetKey('X'):
            self.context.scene.dump()
            time.sleep(0.5)  # TODO prevent multiple key-presses properly
        
        if glfw.GetKey('T'):
            self.context.task.toggle()
            time.sleep(0.5)  # TODO prevent multiple key-presses properly
        
        if glfw.GetKey('I'):
            inputSnapshot = self.context.cubeTracker.imageIn  # grab current input image as snapshot
            cv2.imshow("Input snapshot", inputSnapshot)  # show snapshot in a window
            #cv2.imwrite(self.input_snapshot_file, inputSnapshot)  # write snapshot to file (NOTE doesn't work!)
            #print "Input snapshot saved to {}".format(self.input_snapshot_file)
            time.sleep(0.5)  # TODO prevent multiple key-presses properly
        
        if glfw.GetKey(glfw.KEY_ESC):
            if not self.quitting:
              self.doQuit = True
              self.quitting = True

        if not self.leftPressed and glfw.GetMouseButton(glfw.MOUSE_BUTTON_LEFT):
            self.leftPressed = True
            self.context.scene.hideCube = not self.context.scene.hideCube

        if not glfw.GetMouseButton(glfw.MOUSE_BUTTON_LEFT):
            self.leftPressed = False

        if not self.rightPressed and glfw.GetMouseButton(glfw.MOUSE_BUTTON_RIGHT):
            self.rightPressed = True
            self.oldMouseX, self.oldMouseY = glfw.GetMousePos()
            self.curMouseX = self.oldMouseX
            self.curMouseY = self.oldMouseY

        if not glfw.GetMouseButton(glfw.MOUSE_BUTTON_RIGHT):
            self.rightPressed = False

        if self.rightPressed: #OK
            self.curMouseX, self.curMouseY = glfw.GetMousePos() #OK
            if self.curMouseX != self.oldMouseX or self.curMouseY != self.oldMouseY: #OK
                oldVec = self.calcArcBallVector(self.oldMouseX, self.oldMouseY) #OK
                curVec = self.calcArcBallVector(self.curMouseX, self.curMouseY) #OK
                angle = math.acos(min(1.0, np.dot(oldVec, curVec))) #OK
                cameraAxis = np.cross(oldVec, curVec) #OK
                cameraAxis /= np.linalg.norm(cameraAxis, ord=2) # normalize cameraAxis to be a unit vector
                cameraToObjectCoords = np.linalg.inv(np.dot(self.context.renderer.view[:-1,:-1], self.context.scene.transform[:-1,:-1])) #???
                cameraAxisObjectCoords = np.dot(cameraToObjectCoords, cameraAxis) #OK
                self.context.scene.transform = hm.rotation(self.context.scene.transform, math.degrees(angle), cameraAxisObjectCoords) #OK
                self.oldMouseX = self.curMouseX #OK
                self.oldMouseY = self.curMouseY #OK

    def calcArcBallVector(self, mouseX, mouseY):
        vec = np.array([float(mouseX) / 640. * 2. - 1.0, float(mouseY) / 480. * 2. - 1.0, 0.], dtype = np.float32) #OK
        vec[1] = -vec[1] #OK
        distSquared = vec[0] * vec[0] + vec[1] * vec[1] #OK
        if distSquared <= 1.0: #OK
            vec[2] = math.sqrt(1.0 - distSquared) #OK
        else: #OK   
            vec[0] = vec[0] / math.sqrt(distSquared) #OK
            vec[1] = vec[1] / math.sqrt(distSquared) #OK
        return vec #OK
