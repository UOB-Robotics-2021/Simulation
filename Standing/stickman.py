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
    def __init__(self, p0, v, m=10, radius=3, color=(0, 255, 0, 0)):
        self.body = pymunk.Body()
        self.body.position = p0
        shape = pymunk.Segment(self.body, (0, 0), v, radius)
        shape.mass = m
        shape.density = 0.1
        shape.elasticity = 0.5
        shape.filter = pymunk.ShapeFilter(group = 1, categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0)
        shape.color = color
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
        
        #pivot angle
        v_p = self.stickFigure.getJointByNumber(1).position - self.stickFigure.getJointByNumber(0).position
        
        print(0, self.stickFigure.getJointByNumber(0).position)
        print(1, self.stickFigure.getJointByNumber(1).position)
        
        pivot_ang = math.degrees(np.arctan(v_p[0]/v_p[1]))
        
        #joint angle
        v_j = self.stickFigure.lowerLeg.body.position - self.stickFigure.getJointByNumber(1).position        
        
        dot_product = np.dot(v_p, v_j)
        joint_ang = round(dot_product/(np.linalg.norm(v_p)*np.linalg.norm(v_j)), 6)
        ang_j = 180 - math.degrees(np.arccos(joint_ang)) # round to avoid arctan errors for small angles
        
        self.angles = {
                "pivot": pivot_ang, 
                "joint": ang_j,
                "elbow": self.stickFigure.elbowAngle(),
                "shoulder": self.stickFigure.shoulderAngle(),
                "pelvis": self.stickFigure.pelvisAngle(),
                "knee": self.stickFigure.kneeAngle(),
        }
        self.ang_vel = {
                "pivot": 0, 
                "joint": 0,
                "elbow": 0,
                "shoulder": 0,
                "pelvis": 0,
                "knee": 0,
        }
        self.ang_acc = {
                "pivot": 0, #those initial values are incorrect
                "joint": 0,
                "elbow": 0,
                "shoulder": 0,
                "pelvis": 0,
                "knee": 0,  
        }
        
        
    def run(self):
        while self.running:
            #Handle user interaction
            for event in pygame.event.get():
                self.do_event(event)
            
            if self.stickFigure.config["environmentConfig"]["constrainJointAngles"]:
                #Apply constraints every timestep
                self.stickFigure.applyConstraints()
                self.stickFigure.swingResistance()
            
            self.stickFigure.makeDecision(self.angles["pivot"], self.ang_vel["pivot"])

            #Update animation
            self.draw()
            self.clock.tick(fps)

            for i in range(steps):
                space.step(1/fps/steps)
            self.update_state()
        
        #Exit simulation
        pygame.quit()
        
    def do_event(self, event):
        
        if event.type == pygame.QUIT:
            self.running = False
        
        keys = pygame.key.get_pressed()
        self.stickFigure.keys = keys
        
        if keys[pygame.K_UP] and keys[pygame.K_DOWN] == 0:
            self.stickFigure.moveLimb("knee", "extension")
        elif keys[pygame.K_DOWN] and keys[pygame.K_UP] == 0:
            self.stickFigure.moveLimb("knee", "flexion", motorSpeed=100)
        elif keys[pygame.K_RIGHT] and keys[pygame.K_LEFT] == 0:
            self.stickFigure.moveLimb("pelvis", "flexion")
        elif keys[pygame.K_LEFT] and keys[pygame.K_RIGHT] == 0:
            self.stickFigure.moveLimb("pelvis", "extension")
        elif keys[pygame.K_w]:
            self.stickFigure.moveLimb("shoulder", "extension", motorSpeed=1)
        elif keys[pygame.K_s]:
            print("flex elbow")
            self.stickFigure.flexElbow()
            
        
        elif keys[pygame.K_i]:
            print(self.angles)
        elif keys[pygame.K_v]:
            print(self.ang_vel)
        elif keys[pygame.K_y]:
            print(self.ang_acc)
            self.stickFigure.moveLimb("shoulder", "flexion", motorSpeed=1)
        elif keys[pygame.K_a]:
            self.stickFigure.footMotor.rate = 1
        elif keys[pygame.K_d]:
            self.stickFigure.footMotor.rate = -1
        elif keys[pygame.K_SPACE]:
            self.stickFigure.stayStill()
            print(self.angles["pivot"])


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
                
                
                
    def update_state(self):
        # calculate changes in angles, ang_vel and ang_acc
        dt = 1/fps
        
        
        prev_angles = {key: self.angles[key] for key in self.angles}
        prev_ang_vel = self.ang_vel
        
        
        #pivot angle
        v_p = self.stickFigure.getJointByNumber(1).position - self.stickFigure.getJointByNumber(0).position
        
        pivot_ang = math.degrees(np.arctan(v_p[0]/v_p[1]))
        
        #joint angle
        v_j = self.stickFigure.lowerLeg.body.position - self.stickFigure.getJointByNumber(1).position        
        
        dot_product = np.dot(v_p, v_j)
        joint_ang = round(dot_product/(np.linalg.norm(v_p)*np.linalg.norm(v_j)), 6)
        ang_j = 180 - math.degrees(np.arccos(joint_ang)) # round to avoid arctan errors for small angles
        
        self.angles["pivot"] = pivot_ang
        self.angles["joint"] = ang_j
        self.angles["elbow"] = self.stickFigure.elbowAngle()
        self.angles["shoulder"] = self.stickFigure.shoulderAngle()
        self.angles["pelvis"] = self.stickFigure.pelvisAngle()
        self.angles["knee"] = self.stickFigure.kneeAngle()
                
        
        
        dangles = {key: self.angles[key] - prev_angles[key] for key in self.angles} 
        self.ang_vel = {key: dangles[key]/dt for key in dangles}
        
        dv = {key: self.ang_vel[key] - prev_ang_vel[key] for key in self.ang_vel}
        self.ang_acc = {key: dv[key]/dt for key in dv}
        
        
        

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
        self.maxAngles = [self.theta]
        self.squatIndex = 0

        self.swing = self.generateSwing()
        self.generateStickman()

    def generateSwing(self):
        # specifies the top of the swing as defined by topPosition
        config = self.config["swingConfig"]
        
        top = pymunk.Body(10,1000000, pymunk.Body.STATIC)
        top.position = Vec2d(*config['topPosition'])
        top_shape = pymunk.Poly.create_box(top, (20,20))
        top_shape.filter = pymunk.ShapeFilter(group = 1, categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0)

        self.space.add(top, top_shape)

        joints = [] # list of [body, shape]

        self.topVec = config["jointDistances"][0] * Vec2d(np.cos(self.swingAngle * np.pi/180), np.sin(self.swingAngle * np.pi/180))
        topSegment = Segment(top.position, self.topVec, 1, 2, (0, 0, 255, 0))
        PivotJoint(top, topSegment.body, (0, 0))

        self.botVec = config["jointDistances"][1] * Vec2d(np.cos(self.swingAngle * np.pi/180), np.sin(self.swingAngle * np.pi/180))
        botSegment = Segment(top.position + self.topVec, self.botVec, 1, 2, (0, 0, 255, 0))
        PivotJoint(topSegment.body, botSegment.body, self.topVec)

        joints.append(topSegment.body)
        joints.append(botSegment.body)

        self.swingVector = config['swingLength'] * Vec2d(np.cos(self.swingAngle * np.pi/180), np.sin(self.swingAngle * np.pi/180))

        self.rodSwingMotor = pymunk.SimpleMotor(joints[0], joints[1], 0)
        space.add(self.rodSwingMotor)

        return {'rod' : joints, 'top' : [top, top_shape]}
        
    def getJointByNumber(self, num):
        return self.swing['rod'][num]
        
    def swingResistance(self):
        # Fetch speed of top segment
        v = self.getJointByNumber(0).velocity
        r = (self.config["swingConfig"]["jointDistances"][0]/2)
        w = v / r
        # Apply force in opposite direction of velocity, proportional to the magnitude of the velocity
        f_coeff = 200000
        f = -w * f_coeff
        pos = (self.getJointByNumber(1).position - self.getJointByNumber(0).position)/2
        self.getJointByNumber(0).apply_force_at_local_point(f, pos)
        # TODO: Perhaps force is proportional to distance from equilibrium point too? Idk

    def generateStickman(self):
        """
        Returns:
            joints (dictionary) - 2D  dictionary of joints with the following keys:
                motor - motor body
                targetAngle (positive float) - angle to stop motion at
                motionType (string) - flexion or extension
                previousAngle (positive float) - previous angle of the joint
                flexionDirection (boolean) - +1 if flexion is clockwise and -1 if flexion is counter-clockwise
                extensionDirection (boolean) -  +1 if extension is clockwise and -1 if extension is counter-clockwise
                
            limbs (dictionary) - 1D dictionary. Key is limb name and value is limb body
                
                
        """
        # In the json file, the format for limbs is --> "limb": [angle, length, mass].
        # The head has format --> "head": [radius, mass]
        foot_index = -1
        hand_index = 1
        self.hand_index, self.foot_index = hand_index, foot_index
        self.footPosition = self.swing['rod'][foot_index].position + self.botVec
        
        #Generate lower leg and knee
        self.lowerLegVector = self.dirVec("lowerLeg")
        self.lowerLeg = Segment(self.footPosition, self.lowerLegVector, self.limbMass("lowerLeg"))
        self.kneePosition = self.vectorSum(self.footPosition, self.lowerLegVector)
        self.footMotor = pymunk.SimpleMotor(self.getJointByNumber(foot_index), self.lowerLeg.body, 0)
        self.footMotor.collide_bodies = False
        self.space.add(self.footMotor)
        
        #Generate upper leg
        self.upperLegVector = self.dirVec("upperLeg")
        self.upperLeg = Segment(self.kneePosition, self.upperLegVector, self.limbMass("upperLeg"))
        self.knee = PivotJoint(self.lowerLeg.body, self.upperLeg.body, self.lowerLegVector)
        self.kneeMotor = pymunk.SimpleMotor(self.lowerLeg.body, self.upperLeg.body, 0)
        self.kneeMotor.collide_bodies = False
        self.space.add(self.kneeMotor)

        #Generate pelvis and torso
        self.pelvisPosition = self.vectorSum(self.kneePosition, self.upperLegVector)
        self.torsoVector = self.dirVec("torso")
        self.torso = Segment(self.pelvisPosition, self.torsoVector, self.limbMass("torso"))
        self.pelvis = PivotJoint(self.upperLeg.body, self.torso.body, self.upperLegVector)
        self.pelvisMotor = pymunk.SimpleMotor(self.upperLeg.body, self.torso.body, 0)
        self.pelvisMotor.collide_bodies = False
        self.space.add(self.pelvisMotor)

        #Generate shoulder and upper arm
        self.shoulderPosition = self.vectorSum(self.pelvisPosition, self.torsoVector)
        self.upperArmVector = self.dirVec("upperArm")
        self.upperArm = Segment(self.shoulderPosition, self.upperArmVector, self.limbMass("upperArm"))
        self.shoulder = PivotJoint(self.torso.body, self.upperArm.body, self.torsoVector)
        self.shoulderMotor = pymunk.SimpleMotor(self.torso.body, self.upperArm.body, 0)
        self.shoulderMotor.collide_bodies = False
        space.add(self.shoulderMotor)
        
        #Generate elbow and lower arm
        self.elbowPosition = self.vectorSum(self.shoulderPosition, self.upperArmVector)
        self.lowerArmVector = self.getJointByNumber(hand_index).position - self.elbowPosition
        self.lowerArm = Segment(self.elbowPosition, self.lowerArmVector, self.limbMass("lowerArm"))
        self.elbow = PivotJoint(self.upperArm.body, self.lowerArm.body, self.upperArmVector)
        self.elbowMotor = pymunk.SimpleMotor(self.upperArm.body, self.lowerArm.body, 0)
        self.elbowMotor.collide_bodies = False
        #space.add(self.elbowMotor)

        #Generate head
        headRadius = self.config['squatStandConfig']["head"][0]
        headPosition = self.shoulderPosition + (headRadius * Vec2d(np.sin(self.stickFigureAngle * np.pi/180), -np.cos(self.stickFigureAngle * np.pi/180)))
        self.head = Circle(headPosition, headRadius)
        self.headJoint = PivotJoint(self.torso.body, self.head.body, self.torsoVector + (headRadius * Vec2d(np.sin(self.stickFigureAngle * np.pi/180), -np.cos(self.stickFigureAngle * np.pi/180))))

        #Attack stick figure to swing
        self.holdHand = PivotJoint(self.lowerArm.body, self.getJointByNumber(hand_index), self.lowerArmVector)
        self.holdFoot = PivotJoint(self.getJointByNumber(foot_index), self.lowerLeg.body, self.botVec)

        self.previousKneeAngle = None

        
        # Limbs
        self.limbs = {'lowerLeg': self.lowerLeg, 'upperLeg': self.upperLeg, 'torso': self.torso, 'upperArm': self.upperArm,
                      'lowerArm': self.lowerArm}

        self.joints = {
                "knee": {"motor": self.kneeMotor, "targetAngle": None, "motionType": None, "previousAngle": None, "flexionDirection": -1, "extensionDirection":1, "constrainLimb": "upperLeg"},
                "foot": {"motor": self.footMotor, "targetAngle": None, "motionType": None, "previousAngle": None, "extensionDirection":-1, "flexionDirection": 1},
                "pelvis": {"motor": self.pelvisMotor, "targetAngle": None, "motionType": None, "extensionDirection":-1, "flexionDirection": 1, "constrainLimb": "torso"},
                "shoulder": {"motor": self.shoulderMotor, "targetAngle": None, "motionType": None, "previousAngle": None, "extensionDirection":1, "flexionDirection": -1, "constrainLimb": self.upperArm.body},
                "elbow": {"motor": self.elbowMotor, "targetAngle": None, "motionType": None, "previousAngle": None, "extensionDirection":-1, "flexionDirection": 1, "constrainLimb": self.lowerArm.body}
                }
        
     
        #If stickman is a ragdoll let stickman fall
        if self.config["environmentConfig"]["ragdollEnabled"]:
            for joint in self.joints.keys():
                self.joints[joint]["motor"].max_force = 0
        #Otherwise set max force of motors to high limit
        else:
            for joint in self.joints.keys():
                self.joints[joint]["motor"].max_force = self.config["environmentConfig"]["motorMaxForce"]
            
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
    def makeDecision(self, angle, angVel):
        offset = 15
        targetAngle = self.maxAngles[-1] - offset

        # If stickman approaches previous maximum amplitude, squat
        if abs(angle) > abs(targetAngle) and self.squatIndex == 0:
            #self.squat()
            print("Squatting! >.<")
            pos = (self.getJointByNumber(1).position - self.getJointByNumber(0).position)/2 # PLACEHOLDER
            self.getJointByNumber(0).apply_force_at_local_point(self.getJointByNumber(0).velocity * 100000, pos) # PLACEHOLDER
            self.squatIndex = 1

        # If swing reaches ~0 angular velocity, add angle to maxAngles
        if abs(angle) > abs(self.maxAngles[-1]) and abs(angVel) < 3:
            self.maxAngles.append(abs(angle))

        # If swing vertical (more or less), make stickman stand. Edit this to change when he should stand
        if abs(angle) < 3 and self.squatIndex == 1:
            #self.stand()
            print("Standing! :D")
            self.squatIndex = 0


    def moveLimb(self, joint, motionType, angle=None, motorSpeed=None):
       
        
        #If user not set motorSpeed or is set higher than max value in config file
        if motorSpeed is None or motorSpeed > config["jointConstraints"]["jointSpeed"]:
            motorSpeed = config["jointConstraints"]["jointSpeed"]
        else:
            motorSpeed = abs(motorSpeed)
        
        if motionType == "extension":
            #Motor set so that negative rate means clockwise (opposite way round)
            motorSpeed = -self.joints[joint]["extensionDirection"]*motorSpeed
            if angle is None:
                angle = config["jointConstraints"][joint+"Extension"]
            else:
                angle = abs(angle)
                if self.joints[joint]["extensionDirection"] == 1 and angle > config["jointConstraints"][joint+"Extension"]:
                    angle = config["jointConstraints"][joint+"Extension"]
                elif self.joints[joint]["extensionDirection"] == -1  and angle < config["jointConstraints"][joint+"Extension"]:
                    angle = config["jointConstraints"][joint+"Extension"]
        elif motionType == "flexion":
            #Motor set so that negative means clockwise (opposite way round)
            motorSpeed = -self.joints[joint]["flexionDirection"]*motorSpeed
            if angle is None:
                angle = config["jointConstraints"][joint+"Flexion"]
            else:
                angle = abs(angle)
                if self.joints[joint]["flexionDirection"] == 1 and angle > config["jointConstraints"][joint+"Flexion"]:
                    angle = config["jointConstraints"][joint+"Flexion"]
                elif self.joints[joint]["flexionDirection"] == -1  and angle < config["jointConstraints"][joint+"Flexion"]:
                    angle = config["jointConstraints"][joint+"Flexion"]
        else:
            print("motionType must be flexion or extension")
            return
        
        #Drive motor
        self.joints[joint]["motor"].max_force = self.config["environmentConfig"]["motorMaxForce"]
        self.joints[joint]["motor"].rate = motorSpeed
        
        self.joints[joint]["targetAngle"] = angle
        self.joints[joint]["motionType"] = motionType
        
        if self.config["environmentConfig"]["motorUsingTargetAngle"]:
            print(joint + " " + motionType + " to" + " " + str(angle) + " at a rate of " + str(motorSpeed))
        else:
            print(joint + " " + motionType +  " at a rate of " + str(motorSpeed))
            
        return
    
    
    def flexPelvis(self):
        self.pelvisMotor.rate = -1
    
    def extendPelvis(self):
        self.pelvisMotor.rate = 1
    
    def flexElbow(self):
        print("flex elbow")
        self.shoulderMotor.max_force = 100*self.config["environmentConfig"]["motorMaxForce"]
        self.shoulderMotor.rate = -1
        
    def extendElbow(self):
        print("extend elbow")
        self.shoulderMotor.max_force = 100*self.config["environmentConfig"]["motorMaxForce"]
        self.shoulderMotor.rate = 1
    
    def stayStill(self, joint=None):

        #Stop all motors
        if joint == None:
            for joint in self.joints.keys():
                self.joints[joint]["motor"].max_force = self.config["environmentConfig"]["motorMaxForce"]
                self.joints[joint]["motor"].rate = 0
        #Stop specific motor
        else:
            self.joints[joint]["motor"].max_force = self.config["environmentConfig"]["motorMaxForce"]
            self.joints[joint]["motor"].rate = 0
            
        self.kneeMotion = None
        
    def applyConstraints(self):
        """
        Stops motion if constraints breached (prevents user from holding down an arrow key)
        """
        
        kneeAngle = self.limbAngle("knee")
        pelvisAngle = self.limbAngle("pelvis")
        
        if self.kneeMotor.rate != 0 or self.kneeMotor.max_force < 100: #If motor engaged or limited by stickman being a ragdoll
            if (
                    self.joints["knee"]["motionType"] != "flexion"
                    and (
                            (kneeAngle > config["jointConstraints"]["kneeExtension"]) 
                            or (self.joints["knee"]["targetAngle"] is not None and kneeAngle > self.joints["knee"]["targetAngle"] ))
                    ): 
                self.stayStill("knee")
                rotationAngle = self.upperLeg.body.rotation_vector.angle_degrees
                print("test", rotationAngle)
                print("Reached extension knee angle of", kneeAngle)
            elif (
                    self.joints["knee"]["motionType"] != "extension"
                    and self.torso.body.position[0] < self.upperLeg.body.position[0] 
                    and (
                            (kneeAngle < config["jointConstraints"]["kneeFlexion"]) 
                            or (self.joints["knee"]["targetAngle"]  is not None and kneeAngle < self.joints["knee"]["targetAngle"]))
                    ):
                self.stayStill("knee")
                rotationAngle = self.upperLeg.body.rotation_vector.angle_degrees
                print("Reached flexion knee angle of", kneeAngle)
                
            
            self.joints["knee"]["previousAngle"]  = kneeAngle
        elif self.joints["knee"]["motor"].rate != 0 or self.joints["knee"]["motor"].max_force < 100:
            if (
                    self.joints["knee"]["motionType"] != "extension"
                    and(
                            kneeAngle < config["jointConstraints"]["kneeFlexion"]
                            )
                ):
                    self.stayStill()
                    print("Reached knee flexion angle of ", kneeAngle)
        
        elif self.joints["pelvis"]["motor"].rate != 0 or self.joints["pelvis"]["motor"].max_force < 100:

            if (
                    self.joints["pelvis"]["motionType"] != "flexion" #limb not undergoing flexion
                    and (
                            (pelvisAngle < config["jointConstraints"]["pelvisExtension"]) 
                            or (self.joints["pelvis"]["targetAngle"] is not None and pelvisAngle < self.joints["pelvis"]["targetAngle"] ))
                    ):
                        self.stayStill("pelvis")
                        print("Reached extension pelvis angle of",pelvisAngle, config["jointConstraints"]["pelvisExtension"], self.joints["pelvis"]["targetAngle"] )
                        print("upperArm ", self.upperArm.body.position, "torso ", self.torso.body.position)
            elif (
                    self.joints["pelvis"]["motionType"] != "extension" #limb not undergoing flexion
                    and (
                            (pelvisAngle > config["jointConstraints"]["pelvisFlexion"]) 
                            or (self.joints["pelvis"]["targetAngle"]  is not None and pelvisAngle > self.joints["pelvis"]["targetAngle"]))
                    ):
                self.stayStill("pelvis")
                print("Reached flexion pelvis angle of", pelvisAngle, config["jointConstraints"]["pelvisFlexion"],  self.joints["pelvis"]["targetAngle"])
    
    
    def kneeAngle(self):
        
        #upperLegVector = self.torso.body.position - self.upperLeg.body.position
        upperLegVector = self.upperLeg.body.position - self.torso.body.position
        lowerLegVector = self.lowerLeg.body.position - self.upperLeg.body.position
        
        
        v0  = upperLegVector / np.linalg.norm(upperLegVector)
        v1 = lowerLegVector / np.linalg.norm(lowerLegVector)
        dot_product = np.dot(v0, v1)
        angle = math.degrees(np.arccos(dot_product))
        
        #Angles defined so 0 is at 12 and 180 is at 6
        if self.torso.body.position[0] < self.upperLeg.body.position[0]:
            angle = 360 - angle    
      
        
        return angle
    
    def pelvisAngle(self):
        
        torsoVector = self.upperLeg.body.position - self.upperArm.body.position
        upperLegVector = self.upperLeg.body.position - self.torso.body.position
        
        v0  = torsoVector / np.linalg.norm(torsoVector)
        v1 = upperLegVector / np.linalg.norm(upperLegVector)
        dot_product = np.dot(v0, v1)
        angle = math.degrees(np.arccos(dot_product))
        
        #Angles defined so 0 is at 12 and 180 is at 6
        if self.upperArm.body.position[0] < self.torso.body.position[0]:
            angle = 360-angle
        
        
        return angle
    
    def elbowAngle(self):
        upperArmVector = self.upperArm.body.position - self.lowerArm.body.position
        lowerArmVector = self.lowerArm.body.position - self.getJointByNumber(1).position
        
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
    
    # Methods to measure angles
    def jointAngle(self, joint):
        limb = self.joints[joint]

        if limb == "lowerLeg":
            return self.limbs[limb].body.angle

    # Methods to measure angles
    def limbAngle(self, joint):
        
        limbKey = self.joints[joint]["constrainLimb"]
        body = self.limbs.get(limbKey).body
        angle=body.rotation_vector.angle_degrees
        offset = self.config["squatStandConfig"][limbKey][0] + 90
        angle = angle + offset
        
        return angle

angle = 45

man = Stickman(space=space, config=config, scale=0.8, lean=0, theta=angle)

data = []

App(man).run()

data = pd.DataFrame(data, columns=['tick', 'vx', 'vy'])
data.to_csv('data.csv')
plt.plot(data)