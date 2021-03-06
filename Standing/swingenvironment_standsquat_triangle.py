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

config = loadConfig('config_standsquat_triangle.json')

# Set-up environment
space = pymunk.Space()
space.gravity = config['environmentConfig']["gravity"] #0,900
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
        space.add(joint)

class PivotJoint:
    def __init__(self, b, b2, a=(0, 0), a2=(0, 0), collide=True):
        joint = pymunk.PinJoint(b, b2, a, a2)
        joint.collide_bodies = collide
        space.add(joint)

class Segment:
    def __init__(self, p0, v, m=10, radius=2, kinematic=0, body=0):
        if body == 0:
            self.body = pymunk.Body()
            self.body.position = p0
            shape = pymunk.Segment(self.body, (0, 0), v, radius)
            shape.mass = m
            shape.density = 0.1
            shape.elasticity = 0.5
            shape.filter = pymunk.ShapeFilter(categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)
            shape.color = (0, 255, 0, 0)
            space.add(self.body, shape)
        else:
            print(body)
            shape = pymunk.Segment(body, (0, 0), v, radius)
            shape.mass = m
            shape.density = 0.1
            shape.elasticity = 0.5
            shape.filter = pymunk.ShapeFilter(categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)
            shape.color = (0, 255, 0, 0)
            space.add(shape)


class Circle:
    def __init__(self, pos, radius=20):
        self.body = pymunk.Body()
        self.body.position = pos
        shape = pymunk.Circle(self.body, radius)
        shape.density = 0.01
        shape.friction = 0.5
        shape.elasticity = 1
        shape.filter = pymunk.ShapeFilter(categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)
        space.add(self.body, shape)

class App:
    def __init__(self, man, swing):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(size)
        self.draw_options = DrawOptions(self.screen)
        self.running = True
        self.gif = 0
        self.images = []
        self.stickFigure = man
        self.swing = swing

    def run(self):
        while self.running:
            self.stickFigure.applyConstraints()
            for event in pygame.event.get():
                self.do_event(event)
 
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
            print(self.stickFigure.legAngle())
            self.stickFigure.upKey = 1
            self.stickFigure.downKey = 0
            self.stickFigure.rotateClockwise(self.swing)
        elif keys[pygame.K_DOWN]:
            self.stickFigure.rotateCounterClockwise(self.swing)
            self.stickFigure.downKey = 1
            self.stickFigure.upKey = 0
 
       
        if event.type == KEYDOWN:
            if event.key in (K_q, K_ESCAPE):
                self.running = False

            elif event.key == K_p:
                pygame.image.save(self.screen, 'joint.png')

            elif event.key == K_g:
                self.gif = 60

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
    def __init__(self, config, scale=1, swing=None, lean=0):
        # In the json file, the format for limbs is --> "limb": [angle, length, mass].
        # The head has format --> "head": [radius, mass]
        self.theta = swing.theta - 90 - lean
        self.config = config
        self.generateSwingSystem()
        #Generate foot and ankle
        #self.anklePosition = swing.getJointByNumber(-1).position - (self.config["foot"][1] * Vec2d(np.cos(self.theta * np.pi/180), np.sin(self.theta * np.pi/180)))
        #self.anklePosition = self.swingSegmentArray[0].body.position - (self.config["foot"][1] * Vec2d(np.cos(self.theta * np.pi/180), np.sin(self.theta * np.pi/180)))
       
        #self.footVector = swing.getJointByNumber(-1).position - self.anklePosition
        
        #Generate lower leg and knee
       
        #Attach foot to swing by sharing same body
        """
        self.anklePosition = self.swingSegmentArray[2].body.position
        self.footVector = self.swingSegmentArray[2].body.position - self.anklePosition
        self.foot = Segment(self.anklePosition, self.footVector, self.limbMass("foot"), body=self.swingSegmentArray[1].body)
        
        self.lowerLegVector = self.dirVec("lowerLeg", scale)
        self.lowerLeg = Segment(self.anklePosition, self.lowerLegVector, self.limbMass("lowerLeg"))
        self.ankle = PivotJoint(self.swingSegmentArray[1].body, self.lowerLeg.body, (0,0))
        self.kneePosition = self.vectorSum(self.anklePosition, self.lowerLegVector)
        """
        #self.anklePosition = self.swingSegmentArray[2].body.position
        #self.anklePosition = config["anklePosition"]
        #self.footVector = self.swingSegmentArray[2].body.position - self.anklePosition
        self.footVector = self.dirVec("foot", scale)
        #self.foot = Segment(self.swingSegmentArray[2].body.position, self.footVector, self.limbMass("foot"))
        self.foot = Segment(self.swingSegmentArray[2].body.position, (0,0), self.limbMass("foot"))
        self.holdFoot = PivotJoint(self.foot.body, self.swingSegmentArray[2].body)
        self.anklePosition = self.swingSegmentArray[2].body.position
        
        self.lowerLegVector = self.dirVec("lowerLeg", scale)
        self.lowerLeg = Segment(self.anklePosition, self.lowerLegVector, self.limbMass("lowerLeg"))
        self.ankle = PivotJoint(self.foot.body, self.lowerLeg.body, (0,0))
        self.kneePosition = self.vectorSum(self.anklePosition, self.lowerLegVector)
        
        #Generate upper leg
        self.upperLegVector = self.dirVec("upperLeg", scale)
        self.upperLeg = Segment(self.kneePosition, self.upperLegVector, self.limbMass("upperLeg"))
        self.knee = PivotJoint(self.lowerLeg.body, self.upperLeg.body, self.lowerLegVector)

        #Generate pelvis and torso
        self.pelvisPosition = self.vectorSum(self.kneePosition, self.upperLegVector)
        self.torsoVector = self.dirVec("torso", scale)
        self.torso = Segment(self.pelvisPosition, self.torsoVector, self.limbMass("torso"))
        self.pelvis = PivotJoint(self.upperLeg.body, self.torso.body, self.upperLegVector)
        
        #Generate shoulder and upper arm
        self.shoulderPosition = self.vectorSum(self.pelvisPosition, self.torsoVector)
        self.upperArmVector = self.dirVec("upperArm", scale)
        self.upperArm = Segment(self.shoulderPosition, self.upperArmVector, self.limbMass("upperArm"))
        self.shoulder = PivotJoint(self.torso.body, self.upperArm.body, self.torsoVector)
        
        #Generate elbow and lower arm
        self.elbowPosition = self.vectorSum(self.shoulderPosition, self.upperArmVector)
        #self.lowerArmVector = swing.getJointByNumber(1).position - self.elbowPosition
        
        self.lowerArmVector = (self.swingSegmentArray[1].body.position - self.elbowPosition)
        
        #Attach lower arm to swing by making them share the same body
        #self.lowerArm = Segment(self.elbowPosition, self.lowerArmVector, self.limbMass("lowerArm"), body=self.swingSegmentArray[1].body)
        #self.elbow = PivotJoint(self.upperArm.body, self.swingSegmentArray[1].body, self.upperArmVector)


        self.lowerArm = Segment(self.elbowPosition, self.lowerArmVector, self.limbMass("lowerArm"))
        self.elbow = PivotJoint(self.upperArm.body, self.lowerArm.body, self.upperArmVector)
        self.handPosition = self.vectorSum(self.elbowPosition, self.lowerArmVector)
        self.hand = Segment(self.handPosition, (0,0))
        self.wrist = PivotJoint(self.lowerArm.body, self.hand.body, self.lowerArmVector)
        self.holdFoot = PivotJoint(self.hand.body, self.swingSegmentArray[1].body)
        
        #Generate neck
        self.neckPosition = self.vectorSum(self.pelvisPosition, self.torsoVector)
        self.neckVector = self.dirVec("neck", scale)
        self.neck = Segment(self.neckPosition, self.neckVector, 10)
        self.neckJoint = PivotJoint(self.upperArm.body, self.neck.body)
        
        #Generate head
        headRadius = config["head"][0]
        self.headPosition = self.shoulderPosition + (headRadius * Vec2d(np.sin(theta * np.pi/180), -np.cos(theta * np.pi/180)))
        self.head = Circle(self.headPosition, headRadius)
        self.headJoint = PivotJoint(self.torso.body, self.head.body, self.torsoVector + (headRadius * Vec2d(np.sin(theta * np.pi/180), -np.cos(theta * np.pi/180))))

        #Attack stick figure to swing
        
        print(self.swingSegmentArray)
        """
        self.holdHand = PinJoint(self.lowerArm.body, swing.getJointByNumber(1), self.lowerArmVector)
        self.holdFoot = PinJoint(self.foot.body, swing.getJointByNumber(-1), self.footVector)
        """
        #self.holdHand = PinJoint(self.lowerArm.body, self.swingSegmentArray[1].body, self.lowerArmVector)
        #self.holdFoot = PinJoint(self.foot.body, self.swingSegmentArray[3].body, self.footVector)
    def generateSwingSystem(self):
        swingConfig = config["LswingConfig"]
        
        #Generate top pivot
        top = pymunk.Body(10,1000000, pymunk.Body.STATIC)
        top.position = Vec2d(*swingConfig['topPosition'])
        top_shape = pymunk.Poly.create_box(top, (20,20))
        top_shape.filter = pymunk.ShapeFilter(categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)
        space.add(top, top_shape)
        
        #Generate swing
        p = top.position
        b=top
        jointVector = (0, 0)
        segmentArray = []
        for i,j in enumerate(swingConfig["joints"]):
            print(i)
            angle = j[0] + self.theta
            v = j[1] * Vec2d(np.cos(angle * np.pi/180), np.sin(angle * np.pi/180))
            
            shape = Segment(p, v, j[2])
            segmentArray.append(shape)
            joint = PivotJoint(b, shape.body, jointVector)
            motor = pymunk.SimpleMotor(b0, shape.body, rate=0)
            space.add(motor)
           
            jointVector = v
            b = shape.body
            p = p + v   
                
        self.swingSegmentArray = segmentArray
        
        
               
              
            
            
                
            
                
    def dirVec(self, limb, scale):
        angle = self.config[limb][0] + self.theta
        return scale * self.config[limb][1] * Vec2d(np.cos(angle * np.pi/180), np.sin(angle * np.pi/180))
    
    def limbMass(self, limb):
        return self.config[limb][2]

    def vectorSum(self, v1, v2):
        return [(v1[0]+v2[0]), (v1[1]+v2[1])]
    
    def rotateClockwise(self, swing=None):
        x0 = self.upperLeg.body.position[0]
        x1 = self.torso.body.position[0]
        if x0 > x1:
            self.upperLegMotor.rate = -4
            for motor in swing.objects["motors"]:
                motor.rate = 100
            self.legAngle()
        else:
            print("max extension reached")
    def rotateCounterClockwise(self, swing=None):
        if self.legAngle() < config["jointConstraints"]["kneeFlexion"]:
            for motor in swing.objects["motors"]:
                motor.rate = -100
            self.upperLegMotor.rate = 4
            self.legAngle()
        else:
            print("max flexion reached", config["jointConstraints"]["kneeFlexion"])
    
    
    def clockwiseTorque(self, max_force=10000):
        """
        Make stickman stand using counter-clockwise upper leg torque and clockwise upper arm torque.
        
        Parameters:
            max_force(int) - maximum force to be applied
        
        """
        print("Pushing legs RIGHT")
        swing.getJointByNumber(-1).apply_impulse_at_local_point(swing.getJointByNumber(-1).mass*Vec2d(10,0))

        #Apply force to pelvis and opposite force to knee to cause clockwise torque of upper leg
        f1 = [-max_force*np.cos(self.upperLeg.body.angle), -max_force*np.sin(self.upperLeg.body.angle)]
        f2 = [-f1[0], -f1[1]]
        self.upperLeg.body.apply_impulse_at_local_point(f1, self.pelvisPosition)
        self.upperLeg.body.apply_impulse_at_local_point(f2, self.kneePosition)
    
    def anticlockwiseTorque(self, max_force=1000):
        """
        Make stickman stand using counter-clockwise upper leg torque and clockwise upper arm torque.
        
        Parameters:
            max_force(int) - maximum force to be applied
        
        """
        print("Pushing legs LEFT")
        swing.getJointByNumber(-1).apply_impulse_at_local_point(swing.getJointByNumber(-1).mass*Vec2d(-10,0))
        
        #Apply force to pelvis and opposite force to knee to cause clockwise torque of upper leg
        f1 = [-max_force*np.cos(self.upperLeg.body.angle), -max_force*np.sin(self.upperLeg.body.angle)]
        f2 = [-f1[0], -f1[1]]
            
        self.upperLeg.body.apply_impulse_at_local_point(f1, self.pelvisPosition)
        self.upperLeg.body.apply_impulse_at_local_point(f2, self.kneePosition)

    
    def stopMotion(self):
        """
        Stops stickman moving by setting velocity of all its parts to 0
        """
        #print("Stop motion")
        
        self.foot.body.velocity = pymunk.Vec2d(0, 0)
        self.lowerLeg.body.velocity = pymunk.Vec2d(0, 0)
        self.upperLeg.body.velocity = pymunk.Vec2d(0, 0)
        self.torso.body.velocity = pymunk.Vec2d(0, 0)
        self.upperArm.body.velocity = pymunk.Vec2d(0, 0)
        self.lowerArm.body.velocity = pymunk.Vec2d(0, 0)
        self.neck.body.velocity = pymunk.Vec2d(0, 0)
        self.head.body.velocity = pymunk.Vec2d(0, 0)
        
    def applyConstraints(self):
        """
        Stops motion if joints breach angle range
        """
        
    def legAngle(self):
    
        upperLegAngle = self.upperLeg.body.angle
        lowerLegAngle = self.lowerLeg.body.angle
        legAngle = upperLegAngle - lowerLegAngle
        
        upperLegVector = self.torso.body.position - self.upperLeg.body.position
        lowerLegVector = self.upperLeg.body.position - self.lowerLeg.body.position
        
        v0  = upperLegVector / np.linalg.norm(upperLegVector)
        v1 = lowerLegVector / np.linalg.norm(lowerLegVector)
        dot_product = np.dot(v0, v1)
        angle = math.degrees(np.arccos(dot_product))
        
        x0 = self.upperLeg.body.position[0]
        x1 = self.torso.body.position[0]

        
        return angle


# Code for swing
class Swing():
    def __init__(self, space, swingConfig, theta=0):
        self.space = space
        self.theta = theta + 90
        self.objects = self.generateSwing(swingConfig)

    def generateSwing(self, config):
        # specifies the top of the swing as defined by topPosition
        top = pymunk.Body(10,1000000, pymunk.Body.STATIC)
        top.position = Vec2d(*config['topPosition'])
        top_shape = pymunk.Poly.create_box(top, (20,20))

        self.space.add(top, top_shape)

        joints = [] # list of [body, shape]
        pivots = []

        joints.append([top, top_shape])

        for v, j in zip(config['jointDistances'], config['jointMasses']):
            '''
            Iterate through the list of coordinates as specified by jointLocations,
            relative to the top of the swing
            '''
            x,y=v[0], v[1]
           
            point = pymunk.Body(j, 100)
            point.position = top.position + (x * Vec2d(np.cos(self.theta * np.pi/180), y*np.sin(self.theta * np.pi/180)))
            print(point.position)
            point_shape = pymunk.Segment(point, (0,0), (0,0), 5)
            point_shape.filter = pymunk.ShapeFilter(categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)
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

    def render(self, screen):
        pass

    def update(self):
        self.eventListener()

    def eventListener(self):
        pass
    
    

theta = 45 # Rotation of entire system

#swing = Swing(space, config['swingConfig'], theta=theta)
man = Stickman(config=config["squatStandConfig"], scale=0.7, swing=swing, lean=30)

data = []

App(man, swing).run()

data = pd.DataFrame(data, columns=['tick', 'vx', 'vy'])
data.to_csv('data.csv')
plt.plot(data)