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
#space.damping = 0.1

size = w, h = 600, 500
fps = 30
steps = 10

BLACK = (0, 0, 0)
GRAY = (220, 220, 220)
WHITE = (255, 255, 255)

# Classes from Pymunk

def applyDamping(body, gravity, damping, dt):
    pymunk.Body.update_velocity(body, gravity, config['environmentConfig']["damping"], dt)

def zeroGravity(body, gravity, damping, dt):
    pymunk.Body.update_velocity(body, [0,0], damping, dt)

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

class GearJoint:
    def __init__(self, b, b2, phase, ratio):
        joint = pymunk.GearJoint(b, b2, phase, ratio)
        space.add(joint)
        

class Segment:
    def __init__(self, p0, v, m=10, radius=2, layer=None, gravity=None):
        self.body = pymunk.Body()
        self.body.position = p0
     
        shape = pymunk.Segment(self.body, (0, 0), v, radius)
        shape.mass = m
        shape.density = 0.1
        shape.elasticity = 0.5
        
        if gravity is not None:
            print("hello")
            self.body.velocity_func = zeroGravity
            
        if layer is not None:
            shape.filter = pymunk.ShapeFilter(categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ layer)
        
        shape.filter = pymunk.ShapeFilter(categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)
        shape.color = (0, 255, 0, 0)
        space.add(self.body, shape)

class Circle:
    def __init__(self, pos, radius=20):
        self.body = pymunk.Body()
        self.body.position = pos
        self.body.velocity_func = zeroGravity
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
            
            
            if self.stickFigure.pelvisAngle() > 3900:
                self.running = False
            #Handle user interaction
            for event in pygame.event.get():
                self.do_event(event)
            self.stickFigure.applyConstraints()
            self.draw()
            self.clock.tick(fps)

            for i in range(steps):
                space.step(1/fps/steps)

        i=0
        while self.running == False:
             if i ==0:
                 #debug code
                 pass
                 
             i=i+1
             #Handle user interaction
             for event in pygame.event.get():
                self.do_event(event)
             
            
             

            
        
    def do_event(self, event):
        if event.type == QUIT:
            self.running = False
            pygame.quit()
        
        
        keys = pygame.key.get_pressed()
        self.stickFigure.keys = keys
        
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
        p = top.position
        vArray = []
        gears = []
        
        c=0
        for i, j in zip(config['jointDistances'], config['jointMasses']):
            '''
            Iterate through the list of coordinates as specified by jointLocations,
            relative to the top of the swing
            '''
            c=c+1
            #point = pymunk.Body(j, 100)
            v =(i * Vec2d(np.cos(swingAngle * np.pi/180), np.sin(swingAngle * np.pi/180)))
            
  
            point_shape = Segment(p, v, 5)
            #point_shape.body.velocity_func = applyDamping
            #point_shape.body.velocity_func = zeroGravity
            p = p+v
            #point_shape = pymunk.Segment(point, (0,0), (0,0), 5)
            
            point_shape.filter = pymunk.ShapeFilter(categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)
           
            # if the first joint, join to the top, otherwise join to the preceding joint
            if len(joints) == 0:
                pivot = PivotJoint(top, point_shape.body, (0,0))
                pivot.collide_bodies = False
            else:
                
                pivot = PivotJoint(joints[-1][0], point_shape.body, vArray[-1]) # selects the body component of the preceding joint
                if c== 2 or c==3 or c==4:
                    self.gear = pymunk.GearJoint(joints[-1][0], point_shape.body, 0, 1)
    
                    self.space.add(self.gear)
                    print("test", self.gear.ratio)
            joints.append([point_shape.body, point_shape])
            vArray.append(v)
            #self.space.add(point, point_shape)
            """
            for body in self.space.bodies:
                body.velocity_func = zeroGravity
            """
        self.swingVector = config['swingLength'] * Vec2d(np.cos(swingAngle * np.pi/180), np.sin(swingAngle * np.pi/180))

        return {'rod' : joints, 'top' : [top, top_shape], "gears": gears}
    def getJointByNumber(self, num):
        return self.swing['rod'][num][0]
        
    def generateStickman(self, scale, theta):
        # In the json file, the format for limbs is --> "limb": [angle, length, mass].
        # The head has format --> "head": [radius, mass]
        foot_index = -1
        #hand_index = int(len(self.swing['rod'])/2-1)
        hand_index = 1
        self.hand_index, self.foot_index = hand_index, foot_index
        self.maxLegAngles = [0, np.pi/2]
        self.theta = theta
        """
        self.footPosition = self.swing['rod'][foot_index][0].position# + self.swingVector
        self.footVector = self.dirVec("foot", scale)
        self.foot = Segment(self.footPosition, self.footVector, self.limbMass("foot"))
        

        #Generate lower leg and knee
        
        self.anklePosition = self.vectorSum(self.footPosition, self.footVector)
        self.lowerLegVector = self.dirVec("lowerLeg", scale)
        self.lowerLeg = Segment(self.anklePosition, self.lowerLegVector, self.limbMass("lowerLeg"))
        self.ankle = PivotJoint(self.foot.body, self.lowerLeg.body, self.footVector)
        self.ankleMotor = pymunk.SimpleMotor(b0, self.foot.body, 0)
        self.space.add(self.ankleMotor)
        self.kneePosition = self.vectorSum(self.anklePosition, self.lowerLegVector)
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
        self.pelvisMotor = pymunk.SimpleMotor(b0, self.torso.body, rate=0)
        #self.pelvisMotor.max_force = 100
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
        self.lowerArm.body.body_type = 0
        #self.lowerArmMotor = pymunk.SimpleMotor(b0, self.lowerArm.body, 0)
        #space.add(self.lowerArmMotor)
       
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
        self.holdHand = pymunk.PinJoint(self.lowerArm.body, self.swing['rod'][hand_index][0], self.lowerArmVector)
        #self.wristMotor = pymunk.SimpleMotor(b0, self.lowerArm.body, 0)
        #self.space.add(self.wristMotor)
        
        self.ankleMotor.max_force = 100
        
        self.space.add(self.holdHand)
        self.holdFoot = PinJoint(self.foot.body, self.swing['rod'][foot_index][0], (0, 0))

        
        self.lowerLeg.body.velocity_func = zeroGravity
        self.upperLeg.body.velocity_func = zeroGravity
        self.torso.body.velocity_func = zeroGravity
        self.upperArm.body.velocity_func = zeroGravity
        self.lowerArm.body.velocity_func = zeroGravity
        self.head.body.velocity_func = zeroGravity
        
        #self.foot.body.velocity_func = zeroGravity
        """
       
        #Generate foot and ankle
        self.anklePosition = self.swing['rod'][foot_index][0].position
        
        #Generate lower leg and knee
        self.lowerLegVector = self.dirVec("lowerLeg", scale)
        self.lowerLeg = Segment(self.anklePosition, self.lowerLegVector, self.limbMass("lowerLeg"))
        self.lowerLegMotor =pymunk.SimpleMotor(b0, self.lowerLeg.body, 0)
        self.kneePosition = self.vectorSum(self.anklePosition, self.lowerLegVector)
        self.lowerLegMotor = pymunk.SimpleMotor(b0, self.lowerLeg.body, 0)
        space.add(self.lowerLegMotor)
        
        #Generate upper leg
        self.upperLegVector = self.dirVec("upperLeg", scale)
        self.upperLeg = Segment(self.kneePosition, self.upperLegVector, self.limbMass("upperLeg"))
        self.knee = PivotJoint(self.lowerLeg.body, self.upperLeg.body, self.lowerLegVector)
        self.kneeMotor = pymunk.SimpleMotor(b0, self.upperLeg.body, 0)
        space.add(self.kneeMotor)

        #Generate pelvis and torso
        self.pelvisPosition = self.vectorSum(self.kneePosition, self.upperLegVector)
        self.torsoVector = self.dirVec("torso", scale)
        self.torso = Segment(self.pelvisPosition, self.torsoVector, self.limbMass("torso"))
        self.pelvis = PivotJoint(self.upperLeg.body, self.torso.body, self.upperLegVector)
        self.pelvisMotor = pymunk.SimpleMotor(b0, self.torso.body, 0)
        space.add(self.pelvisMotor)
        
        #Generate shoulder and upper arm
        self.shoulderPosition = self.vectorSum(self.pelvisPosition, self.torsoVector)
        self.upperArmVector = self.dirVec("upperArm", scale)
        self.upperArm = Segment(self.shoulderPosition, self.upperArmVector, self.limbMass("upperArm"))
        self.shoulder = PivotJoint(self.torso.body, self.upperArm.body, self.torsoVector)
        #Elbow motor used instead of shoulder motor as elbow range of motion is the limiting factor
        
        #Generate elbow and lower arm
        self.elbowPosition = self.vectorSum(self.shoulderPosition, self.upperArmVector)
        self.lowerArmVector = self.swing['rod'][hand_index][0].position - self.elbowPosition
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
        self.neckMotor = pymunk.SimpleMotor(b0, self.head.body, 0)
        space.add(self.neckMotor)
        
        self.lowerLegMotor.max_force = 100
        
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
        self.lowerLegMotor.max_force = 1000000
        
    def applyConstraints(self):
        """
        Stops motion if constraints breached (prevents user from holding down an arrow key)
        """
        #print(self.elbowAngle())
        
        
        x0 = self.upperLeg.body.position[0]
        x1 = self.torso.body.position[0]
        
        #Don't let knee extend beyond constraint. Allow flexion
        if x1 > x0 and self.keys[pygame.K_DOWN] == 0:
            print("max knee extension reached")
            #self.stayStill()
            self.kneeMotor.rate = 0
        
        #Don't let knee flex beyond constraint. Allow extension
        elif self.kneeAngle() > config["jointConstraints"]["kneeFlexion"] and self.keys[pygame.K_UP] == 0:
            print("max knee flexion reached")
            #self.stayStill()
            self.kneeMotor.rate = 0
        #print(self.pelvisAngle()
        
        """
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
        """
        
    
    def kneeAngle(self):
        
        upperLegVector = self.torso.body.position - self.upperLeg.body.position
        lowerLegVector = self.upperLeg.body.position - self.lowerLeg.body.position
        
        v0  = upperLegVector / np.linalg.norm(upperLegVector)
        v1 = lowerLegVector / np.linalg.norm(lowerLegVector)
        dot_product = np.dot(v0, v1)
        angle = math.degrees(np.arccos(dot_product))
        
     
        return angle
    
    def pelvisAngle(self):
       
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
    
    def ankleAngle(self):
        lowerLegVector = self.upperLeg.body.position - self.lowerLeg.body.position
        footVector = self.foot.body.position - self.lowerLeg.body.position
        dot_product = np.dot(lowerLegVector, footVector)
        
        v0  = footVector / np.linalg.norm(footVector)
        v1 = lowerLegVector / np.linalg.norm(lowerLegVector)
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

