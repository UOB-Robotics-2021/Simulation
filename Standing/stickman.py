# Import modules
import pymunk
import pygame
import json

from pymunk.pygame_util import *
from pymunk.vec2d import Vec2d

from pygame.locals import *

import numpy as np

import pandas as pd
import math
import matplotlib.pyplot as plt

# Custom functions
def loadConfig(configFile):

    with open(f'{configFile}', 'r') as cfg:
        config = cfg.read()

    return json.loads(config)

try:
    config = loadConfig('config_standsquat.json')
except:
    config = loadConfig('Standing//config_standsquat.json')

# Set-up environment
space = pymunk.Space()
space.gravity = config['environmentConfig']["gravity"]
b0 = space.static_body

size = w, h = 600, 500
fps = 30
steps = 10

BLACK = (0, 0, 0)
GRAY = (220, 220, 220)
WHITE = (255, 255, 255)

# Classes from Pymunk
class PinJoint:
    def __init__(self, b, b2, a=(0, 0), a2=(0, 0)):
        joint = pymunk.PinJoint(b, b2, a, a2)
        joint.collide_bodies = False
        space.add(joint)

class PivotJoint:
    def __init__(self, b, b2, a=(0, 0), a2=(0, 0), collide=False):
        joint = pymunk.PinJoint(b, b2, a, a2)
        joint.collide_bodies = collide
        space.add(joint)

class Segment:
    def __init__(self, p0, v, m=10, radius=2, layer=None):
        self.body = pymunk.Body()
        self.body.position = p0
        shape = pymunk.Segment(self.body, (0, 0), v, radius)
        shape.mass = m
        shape.density = 0.1
        shape.elasticity = 0.5
        if layer is not None:
            shape.filter = pymunk.ShapeFilter(categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ layer)
        shape.color = (0, 255, 0, 0)
        space.add(self.body, shape)

class Circle:
    def __init__(self, pos, radius=20):
        self.body = pymunk.Body()
        self.body.position = pos
        shape = pymunk.Circle(self.body, radius)
        shape.density = 0.01
        shape.friction = 0.5
        shape.elasticity = 1
        shape.filter = pymunk.ShapeFilter(categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 1)
        space.add(self.body, shape)

# Class to build environment
class App:
    def __init__(self, man):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(size)
        self.draw_options = DrawOptions(self.screen)
        self.running = True
        self.gif = 0
        self.images = []
        self.stickFigure = man
        self.swing = self.stickFigure.swing
        
        
    def run(self):
        while self.running:
            
            #Handle user interaction
            for event in pygame.event.get():
                self.do_event(event)
            self.stickFigure.applyConstraints()
            self.draw()
            self.clock.tick(fps)

            for i in range(steps):
                space.step(1/fps/steps)

        pygame.quit()
        
    def do_event(self, event):
        if event.type == QUIT:
            self.running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            print(self.stickFigure.kneeAngle())
            self.stickFigure.upKey = 1
            self.stickFigure.downKey = 0
            self.stickFigure.extendKnee()
        elif keys[pygame.K_DOWN]:
            self.stickFigure.flexKnee()
            self.stickFigure.downKey = 1
            self.stickFigure.upKey = 0
        elif keys[pygame.K_RIGHT]:
            self.stickFigure.rightKey = 1
            self.stickFigure.leftKey = 0
            self.stickFigure.flexPelvis()
        elif keys[pygame.K_LEFT]:
            self.stickFigure.rightKey = 0
            self.stickFigure.leftKey = 1
            self.stickFigure.extendPelvis()
        elif keys[pygame.K_w]:
            print("flex elbow")
            self.stickFigure.extendElbow()
        elif keys[pygame.K_s]:
            print("flex elbow")
            self.stickFigure.flexElbow()
        elif keys[pygame.K_SPACE]:
            self.stickFigure.stayStill()
        
    def draw(self):
        self.screen.fill(GRAY)
        space.debug_draw(self.draw_options)
        pygame.display.update()

        text = f'fpg: {self.clock.get_fps():.1f}'
        pygame.display.set_caption(text)
        self.make_gif()

    def make_gif(self):
        if self.gif > 0:
            strFormat = 'RGBA'
            raw_str = pygame.image.tostring(self.screen, strFormat, False)
            image = Image.frombytes(
                strFormat, self.screen.get_size(), raw_str)
            self.images.append(image)
            self.gif -= 1
            if self.gif == 0:
                self.images[0].save('joint.gif',
                                    save_all=True, append_images=self.images[1:],
                                    optimize=True, duration=1000//fps, loop=0)
                self.images = []

# Code to generate figure
class Stickman:
    def __init__(self, space, config, scale=1, lean=0, theta=0):
        """
        Generate the stickman and the swing.

        space: The Pymunk space to add the objects in
        config: The config file containing the stickman and swing info
        scale: Scale of the generated stickman
        lean: The angle at which the stickman leans backwards
        theta: The rotation of the entire system
        """
        self.space = space

        self.swing = self.generateSwing(config['swingConfig'], theta + 90)

        self.config = config['squatStandConfig']
        self.generateStickman(scale, theta - lean)

    def generateSwing(self, config, swingAngle):
        # specifies the top of the swing as defined by topPosition
        top = pymunk.Body(10,1000000, pymunk.Body.STATIC)
        top.position = Vec2d(*config['topPosition'])
        top_shape = pymunk.Poly.create_box(top, (20,20))

        self.space.add(top, top_shape)

        joints = [] # list of [body, shape]

        for i, j in zip(config['jointDistances'], config['jointMasses']):
            '''
            Iterate through the list of coordinates as specified by jointLocations,
            relative to the top of the swing
            '''
            point = pymunk.Body(j, 100)
            point.position = top.position + (i * Vec2d(np.cos(swingAngle * np.pi/180), np.sin(swingAngle * np.pi/180)))
            point_shape = pymunk.Segment(point, (0,0), (0,0), 5)
            # if the first joint, join to the top, otherwise join to the preceding joint
            if len(joints) == 0:
                pivot = pymunk.PinJoint(top, point, (0,0))
            else:
                pivot = pymunk.PinJoint(joints[-1][0], point) # selects the body component of the preceding joint
            pivot.collide_bodies = False
            joints.append([point, point_shape])

            self.space.add(point, point_shape)
            self.space.add(pivot)

        self.swingVector = config['swingLength'] * Vec2d(np.cos(swingAngle * np.pi/180), np.sin(swingAngle * np.pi/180))

        return {'rod' : joints, 'top' : [top, top_shape]}
    
    def getJointByNumber(self, num):
        return self.swing['rod'][num][0]
        
    def generateStickman(self, scale, theta):
        # In the json file, the format for limbs is --> "limb": [angle, length, mass].
        # The head has format --> "head": [radius, mass]
        foot_index = -1
        hand_index = int(len(self.swing['rod'])/2-1)
        self.hand_index, self.foot_index = hand_index, foot_index
        self.maxLegAngles = [0, np.pi/2]
        self.theta = theta
        self.footPosition = self.swing['rod'][foot_index][0].position# + self.swingVector
        
        #Generate lower leg and knee
        self.lowerLegVector = self.dirVec("lowerLeg", scale)
        self.lowerLeg = Segment(self.footPosition, self.lowerLegVector, self.limbMass("lowerLeg"))
        self.kneePosition = self.vectorSum(self.footPosition, self.lowerLegVector)
        self.lowerLegMotor = pymunk.SimpleMotor(b0, self.lowerLeg.body, 0)
        self.space.add(self.lowerLegMotor)
        
        #Generate upper leg
        self.upperLegVector = self.dirVec("upperLeg", scale)
        self.upperLeg = Segment(self.kneePosition, self.upperLegVector, self.limbMass("upperLeg"))
        self.knee = PivotJoint(self.lowerLeg.body, self.upperLeg.body, self.lowerLegVector)
        self.kneeMotor = pymunk.SimpleMotor(b0, self.upperLeg.body, 0)
        self.space.add(self.kneeMotor)

        #Generate pelvis and torso
        self.pelvisPosition = self.vectorSum(self.kneePosition, self.upperLegVector)
        self.torsoVector = self.dirVec("torso", scale)
        self.torso = Segment(self.pelvisPosition, self.torsoVector, self.limbMass("torso"))
        self.pelvis = PivotJoint(self.upperLeg.body, self.torso.body, self.upperLegVector)
        self.pelvisMotor = pymunk.SimpleMotor(b0, self.torso.body, 0)
        self.space.add(self.pelvisMotor)
        
        #Generate shoulder and upper arm
        self.shoulderPosition = self.vectorSum(self.pelvisPosition, self.torsoVector)
        self.upperArmVector = self.dirVec("upperArm", scale)
        self.upperArm = Segment(self.shoulderPosition, self.upperArmVector, self.limbMass("upperArm"))
        self.shoulder = PivotJoint(self.torso.body, self.upperArm.body, self.torsoVector)
        
        #Generate elbow and lower arm
        self.elbowPosition = self.vectorSum(self.shoulderPosition, self.upperArmVector)
        self.lowerArmVector = (self.swing['rod'][hand_index][0].position) - self.elbowPosition
        self.lowerArm = Segment(self.elbowPosition, self.lowerArmVector, self.limbMass("lowerArm"))
        self.elbow = PivotJoint(self.upperArm.body, self.lowerArm.body, self.upperArmVector)
        self.elbowMotor = pymunk.SimpleMotor(b0, self.upperArm.body, 0)
        space.add(self.elbowMotor)
        
        #Generate head
        headRadius = self.config["head"][0]
        headPosition = self.shoulderPosition + (headRadius * Vec2d(np.sin(theta * np.pi/180), -np.cos(theta * np.pi/180)))
        self.head = Circle(headPosition, headRadius)
        self.headJoint = PivotJoint(self.torso.body, self.head.body, self.torsoVector + (headRadius * Vec2d(np.sin(theta * np.pi/180), -np.cos(theta * np.pi/180))))

        self.headPosition = self.shoulderPosition + (headRadius * Vec2d(np.sin(theta * np.pi/180), -np.cos(theta * np.pi/180)))
        self.head = Circle(self.headPosition, headRadius)
        self.headJoint = PivotJoint(self.torso.body, self.head.body, self.torsoVector + (headRadius * Vec2d(np.sin(theta * np.pi/180), -np.cos(theta * np.pi/180))))

        #Attack stick figure to swing
        self.holdHand = PinJoint(self.lowerArm.body, self.swing['rod'][hand_index][0], self.lowerArmVector)
        self.holdFoot = PinJoint(self.lowerLeg.body, self.swing['rod'][foot_index][0], (0, 0))

    def dirVec(self, limb, scale):
        """
        Calculates the vector for the limb.
        """
        angle = self.config[limb][0] + self.theta
        return scale * self.config[limb][1] * Vec2d(np.cos(angle * np.pi/180), np.sin(angle * np.pi/180))
    
    def limbMass(self, limb):
        """
        Returns the mass of the limb.
        """
        return self.config[limb][2]

    def vectorSum(self, v1, v2):
        """
        Returns the sum of two vectors.
        """
        return [(v1[0]+v2[0]), (v1[1]+v2[1])]
    
    def legAngle(self):
        """
        Returns the angle between the upper and lower leg.
        """
        upperLegAngle = self.upperLeg.body.angle
        lowerLegAngle = self.lowerLeg.body.angle
        legAngle = upperLegAngle - lowerLegAngle

        return -legAngle
    
    def extendKnee(self):
        x0 = self.upperLeg.body.position[0]
        x1 = self.torso.body.position[0]
        if x0 > x1:
            self.kneeMotor.rate = -config["jointConstraints"]["jointSpeed"]
            self.kneeAngle()
        else:
            print("max extension reached")
            
    def flexKnee(self):
        if self.kneeAngle() < config["jointConstraints"]["kneeFlexion"]:
            self.kneeMotor.rate = config["jointConstraints"]["jointSpeed"]
            self.kneeAngle()
        else:
            print("max flexion reached", config["jointConstraints"]["kneeFlexion"])
    
    def flexPelvis(self):
        self.pelvisMotor.rate = -1
    
    def extendPelvis(self):
        self.pelvisMotor.rate = 1
    
    def flexElbow(self):
        self.elbowMotor.rate = -1
        
    def extendElbow(self):
        self.elbowMotor.rate = 1
    
    def stayStill(self):
        self.kneeMotor.rate = 0
        self.pelvisMotor.rate = 0
        self.elbowMotor.rate = 0
        
    def applyConstraints(self):
        """
        Stops motion if constraints breached (prevents user from holding down an arrow key)
        """
        #print(self.elbowAngle())
        
       
        
       
        if self.kneeMotor.rate != 0:
            x0 = self.upperLeg.body.position[0]
            x1 = self.torso.body.position[0]
            
            if x1 > x0 and self.upKey==1: 
                self.stayStill()
                print("max extension constaint reached")
            elif self.kneeAngle() > config["jointConstraints"]["kneeFlexion"] and self.downKey == 1:
                self.kneeMotor.rate = 0
                print("max flexion constraint reached")
        elif self.pelvisMotor.rate < 0:
            if self.pelvisAngle() > config["jointConstraints"]["pelvisFlexion"] and self.rightKey == 1:
                self.pelvisMotor.rate = 0
                print("max pelvis flexion")
        elif self.pelvisMotor.rate > 0:
            if self.pelvisAngle() < config["jointConstraints"]["pelvisExtension"] and self.leftKey == 1:
                self.pelvisMotor.rate = 0
                print("max pelvis extension")
        elif self.elbowMotor.rate > 0:
            if self.elbowAngle() < 10:
                self.elbowMotor.rate = 0
                print("max elbow extension")
                print(self.elbowAngle())
        elif self.elbowMotor.rate < 0:
            if self.elbowAngle() > config["jointConstraints"]["elbowFlexion"]:
                print("max elbow flexion")
                self.elbowMotor.rate = 0
        else:
            pass
        
        
    
    def kneeAngle(self):
        
        upperLegVector = self.torso.body.position - self.upperLeg.body.position
        lowerLegVector = self.upperLeg.body.position - self.lowerLeg.body.position
        
        v0  = upperLegVector / np.linalg.norm(upperLegVector)
        v1 = lowerLegVector / np.linalg.norm(lowerLegVector)
        dot_product = np.dot(v0, v1)
        angle = math.degrees(np.arccos(dot_product))
        
     
        return angle
    
    def pelvisAngle(self):
       
        torsoAngle = self.torso.body.angle
        upperLegAngle = self.upperLeg.body.angle
        pelvisAngle = torsoAngle - upperLegAngle
        
        torsoVector = self.upperArm.body.position - self.torso.body.position
        upperLegVector = self.torso.body.position - self.upperLeg.body.position
        
        v0  = torsoVector / np.linalg.norm(torsoVector)
        v1 = upperLegVector / np.linalg.norm(upperLegVector)
        dot_product = np.dot(v0, v1)
        angle = math.degrees(np.arccos(dot_product))
        
        return angle
    
    def elbowAngle(self):
        upperArmVector = self.upperArm.body.position - self.lowerArm.body.position
        lowerArmVector = self.lowerArm.body.position - self.getJointByNumber(self.hand_index).position
        
        v0  = upperArmVector / np.linalg.norm(upperArmVector)
        v1 = lowerArmVector / np.linalg.norm(lowerArmVector)
        dot_product = np.dot(v0, v1)
        angle = math.degrees(np.arccos(dot_product))
        
        return angle
    
    def shoulderAngle(self):
        upperArmVector = self.upperArm.body.position - self.lowerArm.body.position
        torsoVector = self.upperArm.body.position - self.torso.body.position
        
        v0  = upperArmVector / np.linalg.norm(upperArmVector)
        v1 = torsoVector / np.linalg.norm(torsoVector)
        dot_product = np.dot(v0, v1)
        angle = math.degrees(np.arccos(dot_product))
        
        return angle


angle = 45
swingPosition = (300, 50)
swingLength = 200

man = Stickman(space=space, config=config, scale=0.7, lean=20, theta=angle)

data = []

App(man).run()

data = pd.DataFrame(data, columns=['tick', 'vx', 'vy'])
data.to_csv('data.csv')
plt.plot(data)