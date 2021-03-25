# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 09:59:31 2021

@author: Kieron
"""

import numpy as np
import pymunk
from pymunk.vec2d import Vec2d
import pygame
import math
from neuralnetwork import NeuralNetwork

'''
Skeleton Model
if followed, genetic_environment should, in theory, be able to run a machine learning simulation
on the model provided you define a learning environment and feed an appropriate config file

e.g. learningenvironment = LearningEnvironment(space, Model, loadConfig('config.json')['modelConfig'])

class Model():
    def __init__(self, space, configFile, numActions):
        initialise stuff here

    def generateModel(self):
        build the model

    def destroyModel(self):
        clear the model to allow for reset

    def update(self):
        set up the inputs for the step function; determine an action via the neural network

    def step(self, action):
        change the system in some way in response to the action determined by the neural network

    def calculateFitness(self):
        determine the fitness of the model at the end of the simulation (could be changed?)

'''


class VariablePendulum():

    def __init__(self, space, configFile, numActions=500):
        self.num_actions = numActions
        self.config = configFile
        self.space = space
        self.colour = (np.random.uniform(0, 255),np.random.uniform(0, 255),np.random.uniform(0, 255),255)
        self.objects = self.generateModel()

        self.initial_amplitude = abs(self.angle())
        self.maximum_amplitude = abs(self.angle())
        self.fitness = 0
        self.action_step = 0
        self.num_reversals = 0
        self.last_action = 0
        self.prev_angle = self.angle()
        self.prev_dir = 2
        self.steps_to_done = 0
        self.done = False
        self.reached_max = False
        self.reached_max_step = 0
        self.movable = True

        self.energydata = []
        self.phasedata = []
        self.pumpdata = []

        self.neuralnetwork = NeuralNetwork(4, 2, *self.config['hiddenLayers'])

    def update(self):
        if not self.done:
            self.dtheta = self.angle()-self.prev_angle
            self.ddtheta = self.angle()-self.dtheta
            action = self.neuralnetwork.forward([abs(self.angle()), abs(self.dtheta), abs(self.ddtheta), self.maximum_amplitude])
            self.step(action)
            self.checkMovable()
            self.energydata.append((self.action_step, self.body.kinetic_energy))
            self.phasedata.append((self.angle(), self.dtheta))
            self.pumpdata.append((self.action_step, self.angle(), self.prev_dir))

    def resist(self):
        fx = self.dtheta*self.config['dampingCoefficient']
        fy = self.dtheta*self.config['dampingCoefficient']
        self.body.apply_impulse_at_local_point((fx, fy))

    def generateModel(self):
        # Create objects
        moment_of_inertia = 0.25*self.config["flywheelMass"]*self.config["flywheelRadius"]**2

        self.body = pymunk.Body(mass=self.config["flywheelMass"], moment=moment_of_inertia)
        self.body.position = self.config["flywheelInitialPosition"]
        self.circle = pymunk.Circle(self.body, radius=self.config["flywheelRadius"])
        self.circle.filter = pymunk.ShapeFilter(categories=0b1, mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)
        self.circle.friction = 90000
        # Create joints
        self.joint = pymunk.PinJoint(self.space.static_body, self.body, self.config["pivotPosition"])

        self.top = self.body.position[1] + self.config["flywheelRadius"]

        self.space.add(self.body, self.circle, self.joint)

    def angle(self):
        y = self.config['pivotPosition'][1] - self.body.position.y
        x = self.body.position.x - self.config['pivotPosition'][0]
        angle = np.arctan(2*x/y)
        return angle

    def extendRope(self, direction):
        if direction != self.prev_dir:
            self.num_reversals += 1
        self.prev_dir = direction
        if direction == 0:
            if self.movable:
                self.joint.distance = self.config["minPendulumLength"]
                self.last_action = self.action_step
                self.movable = False
        elif direction == 1:
            if self.movable:
                self.joint.distance = self.config["maxPendulumLength"]
                self.last_action = self.action_step
                self.movable = False

    def checkMovable(self):
        if self.action_step - self.last_action > self.config['actionDelay']:
            self.movable = True

    def step(self, action):
        self.action_step += 1
        self.prev_angle = self.angle()

        amplitude = abs(self.angle())
        if amplitude > self.maximum_amplitude:
            self.maximum_amplitude = amplitude
        if self.body.position.y < self.config['pivotPosition'][1]:
            self.reached_max = True
            self.reached_max_step = self.action_step
            self.done = True
        self.extendRope(np.argmax(action))

        if self.action_step >= self.num_actions:
            self.done = True

    def calculateFitness(self):
        return ((self.maximum_amplitude-self.initial_amplitude)*180/np.pi)**2/(self.num_reversals)**0.5

    def destroyModel(self):
        self.space.remove(self.body, self.circle, self.joint)




class SmoothVariablePendulum():

    def __init__(self, space, configFile, numActions=500):
        self.num_actions = numActions
        self.config = configFile
        self.space = space
        self.colour = (np.random.uniform(0,255),np.random.uniform(0,255),np.random.uniform(0,255),255)
        self.objects = self.generateModel()

        self.initial_amplitude = abs(self.angle())
        self.maximum_amplitude = abs(self.angle())
        self.fitness = 0
        self.action_step = 0
        self.num_reversals = 0
        self.last_action = 0
        self.prev_angle = 0
        self.prev_dir = 2
        self.steps_to_done = 0
        self.done = False
        self.complete = False
        self.movable = True

        self.neuralnetwork = NeuralNetwork(6, 5, *self.config['hiddenLayers'])

    def update(self):
        if not self.done:
            dtheta = self.angle()-self.prev_angle
            ddtheta = self.angle()-dtheta
            action = self.neuralnetwork.forward([abs(self.angle()), abs(dtheta), abs(ddtheta), self.maximum_amplitude, self.body.angular_velocity, self.joint.distance])
            self.step(action)
            self.checkMovable()

    def generateModel(self):
        #Create objects
        moment_of_inertia = 0.25*self.config["flywheelMass"]*self.config["flywheelRadius"]**2

        self.body = pymunk.Body(mass=self.config["flywheelMass"], moment=moment_of_inertia)
        self.body.position = self.config["flywheelInitialPosition"]
        self.circle = pymunk.Circle(self.body, radius=self.config["flywheelRadius"])
        self.circle.filter = pymunk.ShapeFilter(categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)
        self.circle.friction = 90000
        self.circle.color = self.colour
        #Create joints
        self.joint = pymunk.PinJoint(self.space.static_body, self.body, self.config["pivotPosition"])

        self.top = self.body.position[1] + self.config["flywheelRadius"]

        self.space.add(self.body, self.circle, self.joint)

    def angle(self):
        y = self.config['pivotPosition'][1] - self.body.position.y
        x = self.body.position.x - self.config['pivotPosition'][0]
        angle = np.arctan(x/y)
        return angle

    def extendRope(self, direction):
        if direction != self.prev_dir and direction != 2: self.num_reversals += 1
        self.prev_dir = direction
        if direction == 0:
            if self.movable:
                self.joint.distance -= self.config['squattingSpeed']
                if self.joint.distance < self.config['minPendulumLength']:
                    self.joint.distance = self.config["minPendulumLength"]
        elif direction == 1:
            if self.movable:
                self.joint.distance += self.config['squattingSpeed']
                if self.joint.distance > self.config['maxPendulumLength']:
                    self.joint.distance = self.config["maxPendulumLength"]
        elif direction == 2:
            self.body.angular_velocity += 1
            if self.body.angular_velocity > 5:
                self.body.angular_velocity = 5
        elif direction == 3:
            self.body.angular_velocity -= 1
            if self.body.angular_velocity < -5:
                self.body.angular_velocity = -5
        else:
            pass

    def checkMovable(self):
        if self.action_step - self.last_action > self.config['actionDelay']:
            self.movable = True

    def step(self, action):
        self.action_step += 1
        self.prev_angle = self.angle()

        amplitude = abs(self.angle())
        if amplitude > self.maximum_amplitude:
            self.maximum_amplitude = amplitude
        self.extendRope(np.argmax(action))

        if self.action_step >= self.num_actions:
            self.done = True
        if self.body.position.y < self.config['pivotPosition'][1]:
            self.done = True
            self.complete = True

    def calculateFitness(self):
        if self.complete:
            return (self.maximum_amplitude-self.initial_amplitude)+200/self.num_reversals
        else:
            return (self.maximum_amplitude-self.initial_amplitude)

    def destroyModel(self):
        self.space.remove(self.body, self.circle, self.joint)


'''
ALL BELOW IS WIP; NONE ARE VERIFIED TO RUN AS EXPECTED
TODO : Add functions for machine learning implementation
    - fitness functions
    - step functions
'''

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
        top_shape.filter = pymunk.ShapeFilter(group = 1, categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0)

        self.space.add(top, top_shape)

        joints = [] # list of [body, shape]

        self.topVec = config["jointDistances"][0] * Vec2d(np.cos(self.swingAngle * np.pi/180), np.sin(self.swingAngle * np.pi/180))
        topSegment = Segment(top.position, self.topVec, 1, 1, (0, 0, 255, 0))
        PivotJoint(b0, topSegment.body, top.position)

        self.botVec = config["jointDistances"][1] * Vec2d(np.cos(self.swingAngle * np.pi/180), np.sin(self.swingAngle * np.pi/180))
        botSegment = Segment(top.position + self.topVec, self.botVec, 1, 1, (0, 0, 255, 0))
        PivotJoint(topSegment.body, botSegment.body, self.topVec)

        joints.append(topSegment.body)
        joints.append(botSegment.body)

        self.swingVector = config['swingLength'] * Vec2d(np.cos(self.swingAngle * np.pi/180), np.sin(self.swingAngle * np.pi/180))

        return {'rod' : joints, 'top' : [top, top_shape]}

    def getJointByNumber(self, num):
        return self.swing['rod'][num]

    def swingResistance(self):
        # TODO: Fetch speed of top segment
        # TODO: Apply force in opposite direction of velocity, proportional to the magnitude of the velocity
        # TODO: Perhaps force is proportional to distance from equilibrium point too? Idk
        pass

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
        self.theta = self.theta - self.lean
        self.maxLegAngles = [0, np.pi/2]

        foot_index = -1
        hand_index = 1
        self.hand_index, self.foot_index = hand_index, foot_index
        self.maxLegAngles = [0, np.pi/2]
        self.footPosition = self.swing['rod'][foot_index].position + self.botVec

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
        #space.add(self.elbowMotor)

        #Generate head
        headRadius = self.config['squatStandConfig']["head"][0]
        headPosition = self.shoulderPosition + (headRadius * Vec2d(np.sin(self.stickFigureAngle * np.pi/180), -np.cos(self.stickFigureAngle * np.pi/180)))
        self.head = Circle(headPosition, headRadius)
        self.headJoint = PivotJoint(self.torso.body, self.head.body, self.torsoVector + (headRadius * Vec2d(np.sin(self.stickFigureAngle * np.pi/180), -np.cos(self.stickFigureAngle * np.pi/180))))

        #Attack stick figure to swing
        self.holdHand = PinJoint(self.lowerArm.body, self.getJointByNumber(hand_index), self.lowerArmVector)
        self.holdFoot = PinJoint(self.getJointByNumber(foot_index), self.lowerLeg.body, self.botVec)

        self.previousKneeAngle = None


        # Limbs
        self.limbs = {'lowerLeg': self.lowerLeg, 'upperLeg': self.upperLeg, 'torso': self.torso, 'upperArm': self.upperArm,
                      'lowerArm': self.lowerArm}

        self.joints = {
                "knee": {"motor": self.kneeMotor, "targetAngle": None, "motionType": None, "previousAngle": None, "flexionDirection": -1, "extensionDirection":1},
                "foot": {"motor": self.footMotor, "targetAngle": None, "motionType": None, "previousAngle": None, "extensionDirection":-1, "flexionDirection": 1},
                "pelvis": {"motor": self.pelvisMotor, "targetAngle": None, "motionType": None, "previousAngle": None, "extensionDirection":-1, "flexionDirection": 1},
                "shoulder": {"motor": self.shoulderMotor, "targetAngle": None, "motionType": None, "previousAngle": None, "extensionDirection":1, "flexionDirection": -1},
                "elbow": {"motor": self.elbowMotor, "targetAngle": None, "motionType": None, "previousAngle": None, "extensionDirection":-1, "flexionDirection": 1}
                }



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
    def moveLimb(self, joint, motionType, angle=None, motorSpeed=None):
        print(len(self.space.bodies))
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
        self.joints[joint]["motor"].rate = motorSpeed

        self.joints[joint]["targetAngle"] = angle
        self.joints[joint]["motionType"] = motionType
        print(joint + " " + motionType +  " " + "to" + " " + str(angle))
        return


    def flexPelvis(self):
        self.pelvisMotor.rate = -1

    def extendPelvis(self):
        self.pelvisMotor.rate = 1

    def flexElbow(self):
        self.elbowMotor.rate = -1

    def extendElbow(self):
        self.elbowMotor.rate = 1

    def stayStill(self):
        for joint in self.joints.keys():
            self.joints[joint]["motor"].rate = 0

        self.kneeMotion = None

    def applyConstraints(self):
        """
        Stops motion if constraints breached (prevents user from holding down an arrow key)
        """

        kneeAngle = self.kneeAngle()

        if self.kneeMotor.rate != 0:
            if (
                    self.keys[pygame.K_DOWN]==0
                    and self.torso.body.position[0] > self.upperLeg.body.position[0]
                    and self.joints["knee"]["motionType"] == "extension"
                    and (
                            (kneeAngle > config["jointConstraints"]["kneeExtension"])
                            or (self.joints["knee"]["targetAngle"] is not None and kneeAngle > self.joints["knee"]["targetAngle"] ))
                    ):
                self.stayStill()
                print("Reached extension knee angle of", kneeAngle)
            elif (
                    self.keys[pygame.K_UP]==0
                    and self.joints["knee"]["motionType"] == "flexion"
                    and self.torso.body.position[0] < self.upperLeg.body.position[0]
                    and (
                            (kneeAngle < config["jointConstraints"]["kneeFlexion"])
                            or (self.joints["knee"]["targetAngle"]  is not None and kneeAngle < self.joints["knee"]["targetAngle"]))
                    ):
                self.stayStill()
                print("Reached flexion knee angle of", kneeAngle)


            self.joints["knee"]["previousAngle"]  = kneeAngle
        elif self.joints["pelvis"]["motor"].rate != 0:
            pelvisAngle = self.pelvisAngle()

            if (
                    self.keys[pygame.K_RIGHT]==0
                    and self.joints["pelvis"]["motionType"] == "extension"
                    and self.upperArm.body.position[0] < self.torso.body.position[0]
                    and (
                            (pelvisAngle < config["jointConstraints"]["pelvisExtension"])
                            or (self.joints["pelvis"]["targetAngle"] is not None and pelvisAngle < self.joints["pelvis"]["targetAngle"] ))
                    ):
                        self.stayStill()
                        print("Reached extension pelvis angle of", pelvisAngle,config["jointConstraints"]["pelvisExtension"] )
            elif (
                    self.keys[pygame.K_LEFT]==0
                    and self.joints["pelvis"]["motionType"] == "flexion"
                    and self.upperArm.body.position[0] > self.torso.body.position[0]
                    and (
                            (pelvisAngle > config["jointConstraints"]["pelvisFlexion"])
                            or (self.joints["pelvis"]["targetAngle"]  is not None and pelvisAngle > self.joints["pelvis"]["targetAngle"]))
                    ):
                self.stayStill()
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

        temp = list(self.limbs)

        if limb == "lowerArm":
            firstVector = self.limbs[limb].body.position - self.getJointByNumber(1).position
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

class Swing():

    def __init__(self, space, swingConfig):
        self.space = space
        self.objects = self.generateSwing(swingConfig)
        self.seat = self.getJointByNumber(-1)
        self.pos = self.seat.position

    def generateSwing(self, config):
        # specifies the top of the swing as defined by topPosition
        top = pymunk.Body(10, 1000000, pymunk.Body.STATIC)
        top.position = Vec2d(*config['topPosition'])
        top_shape = pymunk.Poly.create_box(top, (20,20))

        self.space.add(top, top_shape)

        joints = [] # list of [body, shape]
        pivots = []
        for i, j in zip(config['jointLocations'], config['jointMasses']):
            '''
            Iterate through the list of coordinates as specified by jointLocations,
            relative to the top of the swing
            '''
            point = pymunk.Body(j, 100)
            point.position = top.position + Vec2d(*i)
            point_shape = pymunk.Segment(point, (0,0), (0,0), 5)
            # if the first joint, join to the top, otherwise join to the preceding joint
            if len(joints) == 0:
                pivot = pymunk.PinJoint(top, point, (0,0))
            else:
                pivot = pymunk.PinJoint(joints[-1][0], point) # selects the body component of the preceding joint
            pivot.collide_bodies = False
            joints.append([point, point_shape])
            pivots.append(pivot)

            self.space.add(point, point_shape)
            self.space.add(pivot)

        return {'rod' : joints, 'top' : [top, top_shape], 'pivots' : pivots}

    def getJointByNumber(self, num):
        return self.objects['rod'][num][0]

    def update(self):
        self.eventHandler()

    def eventHandler(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            self.swing.getJointByNumber(-1).apply_impulse_at_local_point(self.swing.getJointByNumber(-1).mass*Vec2d(10,0))
        elif keys[pygame.K_DOWN]:
            self.swing.getJointByNumber(-1).apply_impulse_at_local_point(self.swing.getJointByNumber(-1).mass*Vec2d(-10,0))

class Person():

    def __init__(self, space, pos, mass=5):
        self.space = space
        self.pos = Vec2d(*pos)
        self.mass = mass
        self.objects = self.generatePerson()
        self.legs = self.objects['body'][1]
        self.knee_motor = self.objects['pivots'][1]

    def generatePerson(self):
        body = pymunk.Body(0.75*self.mass, 100000000000000) # assumes the body from the quads up make up 75% of mass
        body.position = self.pos

        legs = pymunk.Body(0.25*self.mass, 100)
        legs.position = self.pos

        torso_shape = pymunk.Segment(body, (0,0), (0, -30), 3)
        bottom_shape = pymunk.Segment(body, (0,0), (20, 0), 3)

        legs_shape = pymunk.Segment(legs, (20, 0), (20, 20), 3)

        knee_joint = pymunk.PinJoint(legs, body, (20, 0), (20,0))
        knee_motor = pymunk.SimpleMotor(legs, body, 0)
        knee_joint.collide_bodies = False

        self.space.add(body, torso_shape, bottom_shape, legs, legs_shape, knee_joint, knee_motor)

        return {'body' : [(body, torso_shape, legs_shape), (legs, legs_shape)], 'pivots' : [knee_joint, knee_motor]}

    def update(self):
        self.limitRotation()

    def limitRotation(self, limits=(np.pi/4, -np.pi/2)):
        # prevents legs from rotating too far, +pi/4 is 45deg clockwise, -pi/2 is 90 deg anticlockwise
        if self.legs[0].angle > limits[0]:
            self.legs[0].angle = limits[0]
        elif self.legs[0].angle < limits[1]:
            self.legs[0].angle = limits[1]

class SittingSwinger(Swing):

    def __init__(self, space, configFile):
        self.person = Person(space, self.swing.pos, mass=configFile['person'])
        self.seat = pymunk.PinJoint(self.swing.seat, self.person.objects['body'][0][0])
        self.seat.collide_bodies = False
        self.space.add(self.seat)

    def eventHandler(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.person.knee_motor.rate = -np.pi
        elif keys[pygame.K_DOWN]:
            self.person.knee_motor.rate = np.pi