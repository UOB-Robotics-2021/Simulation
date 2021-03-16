# Import modules
import pymunk
import pygame
import json
import os


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

dir_name = os.path.basename(os.getcwd())
if dir_name == "Standing":
    config = loadConfig('config_standsquat.json')
else:
    config = loadConfig('Standing//config_standsquat.json')

# Set-up environment
space = pymunk.Space()
space.gravity = config['environmentConfig']["gravity"]
space.damping = 0.9
b0 = space.static_body


size = w, h = 600, 500
fps = 30
steps = 10
timestep = 1/fps/steps


BLACK = (0, 0, 0)
GRAY = (220, 220, 220)
WHITE = (255, 255, 255)

def applyDamping(body, gravity, damping, dt):
    pymunk.Body.update_velocity(body, gravity, body.damping, dt)

# Classes from Pymunk
class PinJoint:
    def __init__(self, b, b2, a=(0, 0), a2=(0, 0)):
        joint = pymunk.PinJoint(b, b2, a, a2)
        joint.collide_bodies = False
        space.add(joint)

class PivotJoint:
    def __init__(self, b, b2, a=(0, 0), a2=(0, 0), collide=False):
        joint = pymunk.PinJoint(b, b2, a, a2)
        joint.collide_bodies = False
        space.add(joint)

class Segment:
    def __init__(self, p0, v, m=10, radius=3):
        self.body = pymunk.Body()
        self.body.position = p0
        shape = pymunk.Segment(self.body, (0, 0), v, radius)
        shape.mass = m
        shape.density = 0.1
        shape.elasticity = 0.5
        shape.filter = pymunk.ShapeFilter(group = 1, categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0)
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
        shape.filter = pymunk.ShapeFilter(group = 1, categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0)
        space.add(self.body, shape)

class App:
    def __init__(self, stickFigure):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(size)
        self.draw_options = DrawOptions(self.screen)
        self.running = True
        self.gif = 0
        self.images = []
        self.stickFigure = stickFigure
        self.swing = self.stickFigure.swing
        
        
    def run(self):
        while self.running:
            #Handle user interaction
            for event in pygame.event.get():
                self.do_event(event)
            
            #Apply constraints every timestep
            self.stickFigure.applyConstraints()
            
            #Update animation
            self.draw()
            self.clock.tick(fps)

            for i in range(steps):
                space.step(1/fps/steps)
        
        #Exit simulation
        pygame.quit()
        
    def do_event(self, event):
        
        keys = pygame.key.get_pressed()
        self.stickFigure.keys = keys
        
        if keys[pygame.K_UP]:
            self.stickFigure.driveKnee("extension")
        elif keys[pygame.K_DOWN]:
            self.stickFigure.driveKnee("flexion", motorSpeed=100)
        elif keys[pygame.K_RIGHT]:
            self.stickFigure.flexPelvis()
        elif keys[pygame.K_LEFT]:
            self.stickFigure.extendPelvis()
        elif keys[pygame.K_w]:
            print("extend elbow")
            self.stickFigure.extendElbow()
        elif keys[pygame.K_s]:
            print("flex elbow")
            self.stickFigure.flexElbow()


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

# Code to generate figure and swing
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
        self.config = config
        self.scale = scale
        self.lean = lean
        self.theta = theta
        self.stickFigureAngle = self.theta - self.lean
        self.swingAngle = self.theta + 90

        self.swing = self.generateSwing()
        self.generateStickman()

    def generateSwing(self):
        # specifies the top of the swing as defined by topPosition
        config = self.config["swingConfig"]
        
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
            point.position = top.position + (i * Vec2d(np.cos(self.swingAngle * np.pi/180), np.sin(self.swingAngle * np.pi/180)))
            point_shape = pymunk.Segment(point, (0,0), (0,0), 5)
            point_shape.filter = pymunk.ShapeFilter(group = 1, categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0)
            # if the first joint, join to the top, otherwise join to the preceding joint
            if len(joints) == 0:
                pivot = pymunk.PinJoint(top, point, (0,0))
            else:
                pivot = pymunk.PinJoint(joints[-1][0], point) # selects the body component of the preceding joint
            pivot.collide_bodies = False
            joints.append([point, point_shape])

            self.space.add(point, point_shape)
            self.space.add(pivot)

        self.swingVector = config['swingLength'] * Vec2d(np.cos(self.swingAngle * np.pi/180), np.sin(self.swingAngle * np.pi/180))

        return {'rod' : joints, 'top' : [top, top_shape]}
    
    def getJointByNumber(self, num):
        return self.swing['rod'][num][0]
        
    def generateStickman(self):
        # In the json file, the format for limbs is --> "limb": [angle, length, mass].
        # The head has format --> "head": [radius, mass]
        self.theta = self.theta - self.lean
        self.maxLegAngles = [0, np.pi/2]
        
        foot_index = -1
        hand_index = 1
        self.hand_index, self.foot_index = hand_index, foot_index
        self.maxLegAngles = [0, np.pi/2]
        self.footPosition = self.swing['rod'][foot_index][0].position
        
        #Generate lower leg and knee
        self.lowerLegVector = self.dirVec("lowerLeg")
        self.lowerLeg = Segment(self.footPosition, self.lowerLegVector, self.limbMass("lowerLeg"))
        self.kneePosition = self.vectorSum(self.footPosition, self.lowerLegVector)
        self.footMotor = pymunk.SimpleMotor(b0, self.lowerLeg.body, 0)
        self.footMotor.collide_bodies = False
        self.space.add(self.footMotor)
        
        #Generate upper leg
        self.upperLegVector = self.dirVec("upperLeg")
        self.upperLeg = Segment(self.kneePosition, self.upperLegVector, self.limbMass("upperLeg"))
        self.knee = PivotJoint(self.lowerLeg.body, self.upperLeg.body, self.lowerLegVector)
        self.kneeMotor = pymunk.SimpleMotor(b0, self.upperLeg.body, 0)
        self.kneeMotor.collide_bodies = False
        self.space.add(self.kneeMotor)

        #Generate pelvis and torso
        self.pelvisPosition = self.vectorSum(self.kneePosition, self.upperLegVector)
        self.torsoVector = self.dirVec("torso")
        self.torso = Segment(self.pelvisPosition, self.torsoVector, self.limbMass("torso"))
        self.pelvis = PivotJoint(self.upperLeg.body, self.torso.body, self.upperLegVector)
        self.pelvisMotor = pymunk.SimpleMotor(b0, self.torso.body, 0)
        self.pelvisMotor.collide_bodies = False
        self.space.add(self.pelvisMotor)
        
        #Generate shoulder and upper arm
        self.shoulderPosition = self.vectorSum(self.pelvisPosition, self.torsoVector)
        self.upperArmVector = self.dirVec("upperArm")
        self.upperArm = Segment(self.shoulderPosition, self.upperArmVector, self.limbMass("upperArm"))
        self.shoulder = PivotJoint(self.torso.body, self.upperArm.body, self.torsoVector)
        self.shoulderMotor = pymunk.SimpleMotor(b0, self.upperArm.body, 0)
        self.shoulderMotor.collide_bodies = False
        space.add(self.shoulderMotor)
        
        #Generate elbow and lower arm
        self.elbowPosition = self.vectorSum(self.shoulderPosition, self.upperArmVector)
        self.lowerArmVector = self.getJointByNumber(hand_index).position - self.elbowPosition
        self.lowerArm = Segment(self.elbowPosition, self.lowerArmVector, self.limbMass("lowerArm"))
        self.elbow = PivotJoint(self.upperArm.body, self.lowerArm.body, self.upperArmVector)
        self.elbowMotor = pymunk.SimpleMotor(b0, self.lowerArm.body, 0)
        self.elbowMotor.collide_bodies = False
        space.add(self.elbowMotor)
        
        #Generate head
        headRadius = self.config['squatStandConfig']["head"][0]
        headPosition = self.shoulderPosition + (headRadius * Vec2d(np.sin(self.stickFigureAngle * np.pi/180), -np.cos(self.stickFigureAngle * np.pi/180)))
        self.head = Circle(headPosition, headRadius)
        self.headJoint = PivotJoint(self.torso.body, self.head.body, self.torsoVector + (headRadius * Vec2d(np.sin(self.stickFigureAngle * np.pi/180), -np.cos(self.stickFigureAngle * np.pi/180))))

        #Attack stick figure to swing
        self.holdHand = PinJoint(self.lowerArm.body, self.getJointByNumber(0), self.lowerArmVector)
        self.holdFoot = PinJoint(self.lowerLeg.body, self.getJointByNumber(foot_index), (0, 0))
        print(hand_index, foot_index)
        self.previousKneeAngle = None

        
        # Limbs
        self.limbs = {'lowerLeg': self.lowerLeg, 'upperLeg': self.upperLeg, 'torso': self.torso, 'upperArm': self.upperArm,
                      'lowerArm': self.lowerArm}

        # Joint Relationships
        self.joints = {'foot': 'lowerLeg', 'knee': 'upperLeg', 'pelvis': 'torso', 'shoulder': 'upperArm', 'elbow': 'lowerArm'}

        # Motors
        self.motors = {'footMotor': self.footMotor, 'kneeMotor': self.kneeMotor, 'pelvisMotor': self.pelvisMotor,
                       'shoulderMotor': self.shoulderMotor, 'elbowMotor': self.elbowMotor}

    # Methods used to create stickman
    def dirVec(self, limb):
        """
        Calculates the vector for the limb.
        """
        angle = self.config['squatStandConfig'][limb][0] + self.stickFigureAngle
        return self.scale * self.config['squatStandConfig'][limb][1] * Vec2d(np.cos(angle * np.pi/180), np.sin(angle * np.pi/180))
    
    def limbMass(self, limb):
        """
        Returns the mass of the limb.
        """
        return self.config['squatStandConfig'][limb][2]

    def vectorSum(self, v1, v2):
        return [(v1[0]+v2[0]), (v1[1]+v2[1])]

    # Methods to move the stickman
    def moveLimb(self, joint, direction):
        self.motors[joint + "Motor"].rate = direction
    
    
    def driveKnee(self, motionType, motorSpeed=None, angle=None):
        """
        Flexes or extends the knee by rotating the upper leg anti-clockwise or clockwise using the knee motor
        
        Parameters:
            motorSpeed (positive float): rate at which to turn the motor
            \n angle (positive float): desired final angle between upper leg and lower leg
            \n motionType (string): can be flexion or extension for anti-clockwise and clockwise knee motor motion
        
        """
        
        #Define parameters
        
        #If user not set motorSpeed or is set higher than max value in config file
        if motorSpeed is None or motorSpeed > config["jointConstraints"]["jointSpeed"]:
            motorSpeed = config["jointConstraints"]["jointSpeed"]
        else:
            motorSpeed = abs(motorSpeed)
        
        if motionType == "extension":
            motorSpeed = -motorSpeed
            if angle is None:
                angle = config["jointConstraints"]["kneeExtension"]
            else:
                angle = abs(angle)
                if angle > config["jointConstraints"]["kneeExtension"]:
                    angle = config["jointConstraints"]["kneeExtension"]
        elif motionType == "flexion":
            if angle is None:
                angle = config["jointConstraints"]["kneeFlexion"]
            else:
                angle = abs(angle)
                if angle < config["jointConstraints"]["kneeFlexion"]:
                    angle = config["jointConstraints"]["kneeFlexion"]
        else:
            print("motionType must be flexion or extension")
            return
        
        #Drive motor
        self.kneeMotor.rate = motorSpeed
        self.targetKneeAngle = angle
        self.kneeMotion = motionType
        return
    
    def extendKnee(self):
        x0 = self.upperLeg.body.position[0]
        x1 = self.torso.body.position[0]
        if x1 > x0 and self.kneeAngle() > config["jointConstraints"]["kneeExtension"]: 
             print("max extension reached")
        else:
            self.kneeMotor.rate = -config["jointConstraints"]["jointSpeed"]
            self.kneeAngle()

            
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
        for motor in self.motors.keys():
            self.motors[motor].rate = 0
            
        self.kneeMotion = None
        
    def applyConstraints(self):
        """
        Stops motion if constraints breached (prevents user from holding down an arrow key)
        """
        
        """
        if self.jointAngle('elbow') < config["jointConstraints"]["elbowExtension"]:
            self.moveLimb('elbow', 1) # Extend
        if self.jointAngle('elbow') > config["jointConstraints"]["elbowFlexion"]:
            self.moveLimb('elbow', -1) # Flex
        
        if self.jointAngle('knee') < config["jointConstraints"]["kneeExtension"]:
            self.moveLimb('knee', 1) # Extend
        if self.jointAngle('knee') > config["jointConstraints"]["kneeFlexion"]:
            self.moveLimb('knee', -1) # Flex
        
        if self.jointAngle('pelvis') < config["jointConstraints"]["pelvisExtension"]:
            self.moveLimb('pelvis', -1) # Flex
            print("Flexing pelvis")
        if self.jointAngle('pelvis') > config["jointConstraints"]["pelvisFlexion"]:
            self.moveLimb('pelvis', 1) # Extend
        
        """
        kneeAngle = self.kneeAngle()
        
        if self.kneeMotor.rate != 0:
            if (
                    self.keys[pygame.K_DOWN]==0 
                    and self.kneeMotion == "extension" 
                    and (
                            (kneeAngle > config["jointConstraints"]["kneeExtension"]) 
                            or (self.targetKneeAngle is not None and kneeAngle > self.targetKneeAngle))
                    ): 
                self.stayStill()
                print("Reached extension knee angle of", kneeAngle)
            elif (
                    self.keys[pygame.K_UP]==0 
                    and self.kneeMotion == "flexion" 
                    and self.torso.body.position[0] < self.upperLeg.body.position[0] 
                    and (
                            (kneeAngle < config["jointConstraints"]["kneeFlexion"]) 
                            or (self.targetKneeAngle is not None and kneeAngle < self.targetKneeAngle))
                    ):
                self.stayStill()
                print("Reached flexion knee angle of", kneeAngle)
                
            
            self.previousKneeAngle = kneeAngle
        
    def kneeAngle(self):
        
        #upperLegVector = self.torso.body.position - self.upperLeg.body.position
        upperLegVector = self.upperLeg.body.position - self.torso.body.position
        lowerLegVector = self.upperLeg.body.position - self.lowerLeg.body.position
        
        v0  = upperLegVector / np.linalg.norm(upperLegVector)
        v1 = lowerLegVector / np.linalg.norm(lowerLegVector)
        dot_product = np.dot(v0, v1)
        angle = math.degrees(np.arccos(dot_product))
        
        if self.previousKneeAngle is not None and self.keys[pygame.K_DOWN]==0 and self.kneeMotion == "extension" and (config["jointConstraints"]["kneeExtension"] > 180 or (self.targetKneeAngle is not None and self.targetKneeAngle > 180)) and angle < self.previousKneeAngle and  self.previousKneeAngle > 150:
            angle = 360-angle
        
        return angle
    
    def pelvisAngle(self):
       
        torsoAngle = self.torso.body.angle
        upperLegAngle = self.upperLeg.body.angle
        pelvisAngle = torsoAngle - upperLegAngle
        
    # Methods to measure angles
    def jointAngle(self, joint):
        limb = self.joints[joint]

        if limb == "lowerLeg":
            return self.limbs[limb].body.angle

        temp = list(self.limbs)
        
        if limb == "lowerArm":
            firstVector = self.limbs[limb].body.position - self.getJointByNumber(self.hand_index).position
        else:
            nextKey = temp[temp.index(limb) + 1]
            firstVector = self.limbs[nextKey].body.position - self.limbs[limb].body.position

        prevKey = temp[temp.index(limb) - 1]
        secondVector = self.limbs[limb].body.position - self.limbs[prevKey].body.position

        v0 = firstVector / np.linalg.norm(firstVector)
        v1 = secondVector / np.linalg.norm(secondVector)
        dotProduct = np.dot(v0, v1)
        angle = math.degrees(np.arccos(dotProduct))

        return angle

angle = 0

man = Stickman(space=space, config=config, scale=0.8, lean=0, theta=angle)

data = []

App(man).run()

data = pd.DataFrame(data, columns=['tick', 'vx', 'vy'])
data.to_csv('data.csv')
plt.plot(data)