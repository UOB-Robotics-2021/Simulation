import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import numpy as np
import pandas as pd
import math
import json

pygame.init()
screen = pygame.display.set_mode((600,600))


space = pymunk.Space()
space.gravity = (0.0, 900) #idle_speed_threshold, iterations (CPU usage)
draw_options = pymunk.pygame_util.DrawOptions(screen)
clock = pygame.time.Clock()

def loadConfig(configFile):

    with open(f'{configFile}', 'r') as cfg:
        config = cfg.read()

    return json.loads(config)


class Person():

    def __init__(self, space, pos, mass=5):
        self.space = space
        self.pos = Vec2d(*pos) # vector position
        self.mass = mass
        self.objects = self.generatePerson()
        self.legs = self.objects['legs']
        self.knee_motor = self.objects['pivots'][2]

    def generatePerson(self,collide=True):
        #inf moment for body, i.e rigid
        body = pymunk.Body(0.90*self.mass, np.inf) # assumes the body from the quads up make up 75% of mass
        body.position = self.pos
        
        legs = pymunk.Body(0.10*self.mass, 30) #mass, moment
        legs.position = self.pos

        torso_shape = pymunk.Segment(body, (0,0), (0, -50), 5) #body, a(first endpoint of segment), b(second enpoint of segment), radius(thickness of segment)
        bottom_shape = pymunk.Segment(body, (0,0), (20, 0), 5)
        self.space.add(body, torso_shape, bottom_shape)
        
        legs_shape = pymunk.Segment(legs, (20, 0), (20, 30), 4)
        self.space.add(legs, legs_shape)
        
        knee_joint = pymunk.constraints.PinJoint(body, legs , (20, 0), (20,30))
        knee_joint2 = pymunk.constraints.PinJoint(body, legs , (20, 0), (20,0))
        #knee_joint3 = pymunk.constraints.DampedRotarySpring(body,legs, 0, 10000000, 10000)
        knee_motor = pymunk.SimpleMotor(body, legs, 0)
        knee_joint.collide_bodies = False            
        knee_joint2.collide_bodies = False
        self.space.add(knee_joint, knee_joint2, knee_motor)
        
        return {'body' : [body, torso_shape, bottom_shape], 'legs' : [legs, legs_shape], 'pivots' : [knee_joint, knee_joint2, knee_motor]}

    def update(self):
        self.limitRotation()
    
    def limitRotation(self, limits=(np.pi/4, -np.pi/2)):
        # prevents legs from rotating too far, +pi/4 is 45deg clockwise, -pi/2 is 90 deg anticlockwise
        if self.legs[0].angle > limits[0]:
            self.legs[0].angle = limits[0]
        elif self.legs[0].angle < limits[1]:
            self.legs[0].angle = limits[1]
            

class Swing():

    def __init__(self, space, swingConfig):
        self.space = space
        self.objects = self.generateSwing(swingConfig)
        self.seat = self.getJointByNumber()
        self.pos = self.seat.position

    def generateSwing(self, config):
        # specifies the top of the swing as defined by topPosition
        top = pymunk.Body(10, 1000000, pymunk.Body.STATIC)
        top.position = Vec2d(*config['topPosition'])
        top_shape = pymunk.Poly.create_box(top, (20,20))

        self.space.add(top, top_shape)

        for i, j in zip(config['jointLocations'], config['jointMasses']):
            '''
            Iterate through the list of coordinates as specified by jointLocations,
            relative to the top of the swing
            '''
            point = pymunk.Body(j, 1000) #stiffness at point of connection to swing
            point.position = top.position + Vec2d(*i)
            point_shape = pymunk.Segment(point, (0,0), (0,0), 0)
            # if the first joint, join to the top, otherwise join to the preceding joint (removed temporarily)
            pivot = pymunk.PinJoint(top, point, (0,0))
            self.space.add(point, point_shape)
            self.space.add(pivot)

        return {'rod' : [point, point_shape], 'top' : [top, top_shape], 'pivots' : [pivot]}
    

    def getJointByNumber(self):
        return self.objects['rod'][0]

    def update(self):
        swing_pos = self.objects['rod'][0].position
        top_pos = self.objects['top'][0].position
        swing_angle = 90-math.degrees(math.atan2(swing_pos[1]-top_pos[1], swing_pos[0]-top_pos[0]))
        return swing_angle
        


class HillClimbingAgent(): #random and may not get best solution in a long time (sub optimal local maxima) --> random restart?
    def __init__(self,env):
        self.build_model()
        
    def build_model(self):
        self.weights = 1e-4*np.random.rand(*self.state_dim, self.action_size) #random weight matrix from hill climbing algorithm
        self.best_reward = -np.inf #inital reward
        self.best_weights = np.copy(self.weights) #current weights
        self.noise_scale = 1e-2 #scale of random noise
        
        
    def get_action(self, state):
        p = np.dot(state, self.weights)
        action = np.argmax(p) #highest value
        return action

    def update_model(self,reward):
        if reward >= self.best_reward:
            self.best_reward = reward
            self.best_weights = np.copy(self.weights)
            self.noise_scale = max(self.noise_scale/2, 1e-3)#half noise scale if current best
        else:
            self.noise_scale = min(self.noise_scale*2, 2)#mutliply by two otherwise for more exploration of different weights
        
        self.weights = self.best_weights * self.noise_scale * np.random.rand(*self.state_dim, self.action_size)


class LearningArea():

    def __init__(self, space, configFile):
        self.space = space
        self.state_dim = 1 #can be controlled from file?
        self.action_size = 3 #currently trivial
        self.current_best_state = 0
        self.best_state = 0
        self.configFile = configFile
        self.swing = Swing(self.space, self.configFile)
        self.person = None
        self.build_model()
        self.font = pygame.font.SysFont('Arial', 20)

        if self.configFile['person']: #initializes swing by using the attributes within the file
            self.person = Person(space, self.swing.pos, mass=configFile['person'])
            self.seat = pymunk.PinJoint(self.swing.seat, self.person.objects['body'][0])
            self.seat.collide_bodies = False
            self.space.add(self.seat)
            
    def build_model(self):
        self.weights = 1e-4*np.random.rand(self.state_dim, self.action_size) #random weight matrix from hill climbing algorithm
        self.best_reward = -np.inf #inital reward
        self.best_weights = np.copy(self.weights) #current weights
        self.noise_scale = 1e-2 #scale of random noise
        
    def get_action(self):
        p = np.dot(self.swing.update(), self.weights)
        action = np.argmax(p) #highest value
        return action
    
    def update(self): #update model also included?
        #self.eventHandler()
        self.swing.update()
        self.state = self.swing.update()
        if abs(self.state) > abs(self.current_best_state):
            self.current_best_state = abs(self.state)
        
        if self.person:
            self.person.update()
            
        return self.current_best_state 
    
    def update_model(self):
        if abs(self.current_best_state) > abs(self.best_state):
            self.best_weights = np.copy(self.weights)
            self.noise_scale = max(self.noise_scale/2, 1e-3)#half noise scale if current best
            self.best_state = abs(self.current_best_state)
        else:
            self.noise_scale = min(self.noise_scale*2, 2)#mutliply by two otherwise for more exploration of different weights
        
        self.weights = self.best_weights * self.noise_scale * np.random.rand(self.state_dim, self.action_size)
    
    def destruct(self):
        for i in self.swing.objects['rod']:
            self.space.remove(i)
        for i in self.swing.objects['pivots']:
            self.space.remove(i)

        for i in self.swing.objects['top']:
            self.space.remove(i)

        for i in self.person.objects['body']:
            self.space.remove(i)

        for i in self.person.objects['legs']:
            self.space.remove(i)

        for i in self.person.objects['pivots']:
            self.space.remove(i)
            
        self.space.remove(self.seat)


    def eventHandler(self,action):
        if action == 0:
            self.person.knee_motor.rate = np.pi
        elif action == 1:
            self.person.knee_motor.rate = -np.pi
        else:
            self.person.knee_motor.rate = 0
        '''
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.person.knee_motor.rate = np.pi
        elif keys[pygame.K_DOWN]:
            self.person.knee_motor.rate = -np.pi'''
            
    def render(self, screen, elapsed_time, ep, best_reward, old_reward, current_reward, steps):
        text = [f'Time Elasped: {round(elapsed_time, 1)}', f'Episode: {ep}', f'Best Reward: {round(best_reward,2)}', f'Previous Reward: {round(old_reward,2)}', 
                f'Current Reward: {round(current_reward,2)}']
        cnt = 0
        for i in text:
            render = self.font.render(i, True, (0,0,0))
            screen.blit(render, (0, cnt*20))
            cnt+=1
        
        
config = loadConfig('config_swinger.json') #text file located in the folder
data = []

num_episodes = 50
old_reward = 0
best_reward = 0
for ep in range(num_episodes):
    la = LearningArea(space, config['swingWithPerson'])
    elapsed_time = 0
    current_reward = 0
    steps = 0
    start_time = pygame.time.get_ticks()
    while elapsed_time < 60*1000: #15seconds episodes
        screen.fill((255,255,255))
        pygame.display.set_caption("Swing Machine Learning [Hill Climbing], Iteration : {}".format(ep+1))
        la.render(screen, elapsed_time/1000, ep+1, best_reward, old_reward, current_reward, steps)
        space.debug_draw(draw_options)
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        action = la.get_action()
        #print(str(action)+" time:" +str(elapsed_time))
        la.eventHandler(action)
        current_reward = la.update()
        space.step(1/60)
        
        clock.tick(60) #fps

        elapsed_time = pygame.time.get_ticks() - start_time
    
    la.update_model() #updates the weight matrix
    print("Episode: {}, best_reward: {:.2f}".format(ep, la.current_best_state))
    if current_reward > best_reward:
        best_reward = current_reward
    old_reward = current_reward
    la.destruct()

    
    
    
pygame.quit()
data = pd.DataFrame(data, columns=['tick', 'vx', 'vy'])
data.to_csv('data.csv')