import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import numpy as np
import math
import json

'''
The hill climbing algorithm was a useful machine learning algorithm to implement for learning the basics of reinforcement learning using
a Agent-Environment system. It however lacks the capabilities of achieving solutions for a more complex swinger system and so is not 
continually used throughout the rest of the project
'''



def loadConfig(configFile): #opens the config files which contains the swing parameters

    with open(f'{configFile}', 'r') as cfg:
        config = cfg.read()

    return json.loads(config)


class Person():
    '''
    The model of the humanoid swinger
    '''
    
    def __init__(self, space, pos, mass):
        self.space = space
        self.pos = Vec2d(*pos) #using vectors for the positions
        self.mass = mass
        self.objects = self.generatePerson()
        self.legs = self.objects['legs']
        self.knee_motor = self.objects['pivots'][2]

    def generatePerson(self,collide=True):
        body = pymunk.Body(0.90*self.mass, np.inf) #assumes the body from the quads up make up 90% of mass
        body.position = self.pos
        
        legs = pymunk.Body(0.10*self.mass, 30) #parameters: mass, moment (inf moment for body, i.e rigid)
        legs.position = self.pos

        torso_shape = pymunk.Segment(body, (0,0), (0, -50), 5) #parameters: body, a(first endpoint of segment), b(second enpoint of segment), radius(thickness of segment)
        bottom_shape = pymunk.Segment(body, (0,0), (20, 0), 5)
        self.space.add(body, torso_shape, bottom_shape)
        
        legs_shape = pymunk.Segment(legs, (20, 0), (20, 30), 4)
        self.space.add(legs, legs_shape)
        
        knee_joint = pymunk.constraints.PinJoint(body, legs , (20, 0), (20,30)) #PinJoints used to simulate joints in the body
        knee_joint2 = pymunk.constraints.PinJoint(body, legs , (20, 0), (20,0)) #parameters: body 1, body 2, first anchor point, second anchor point
        knee_motor = pymunk.SimpleMotor(body, legs, 0) #motor is used to rotate the limb around the joint
        knee_joint.collide_bodies = False            
        knee_joint2.collide_bodies = False
        self.space.add(knee_joint, knee_joint2, knee_motor)
        
        return {'body' : [body, torso_shape, bottom_shape], 'legs' : [legs, legs_shape], 'pivots' : [knee_joint, knee_joint2, knee_motor]}

    def update(self): #When updating the swinger, limits the maximum angle to which the legs can be rotated
        self.limitRotation()
    
    def limitRotation(self, limits=(np.pi/4, -np.pi/2)):  #prevents legs from rotating too far, +pi/4 is 45deg clockwise, -pi/2 is 90 deg anticlockwise
        if self.legs[0].angle > limits[0]:
            self.legs[0].angle = limits[0]
        elif self.legs[0].angle < limits[1]:
            self.legs[0].angle = limits[1]
            

class Swing():
    '''
    Creation of the swing itself
    '''
    def __init__(self, space, swingConfig):
        self.space = space
        self.objects = self.generateSwing(swingConfig)
        self.seat = self.objects['rod'][0]
        self.pos = self.seat.position

    def generateSwing(self, config):
        ###################### Top of the swing ##############################
        top = pymunk.Body(10, 1000000, pymunk.Body.STATIC) 
        top.position = Vec2d(*config['topPosition']) #specifies the top of the swing as defined by topPosition from the config file
        top_shape = pymunk.Poly.create_box(top, (20,20)) #position of the centre of the top of the swing

        self.space.add(top, top_shape)

        for i, j in zip(config['jointLocations'], config['jointMasses']):
            '''
            Takes the coordinates as specified by jointLocation and mass from jointMasses
            '''
            ###################### Connecting joint between swing and seat ##############################
            point = pymunk.Body(j, 1000) 
            point.position = top.position + Vec2d(*i)
            point_shape = pymunk.Segment(point, (0,0), (0,0), 0)
            
            pivot = pymunk.PinJoint(top, point, (0,0))
            
            self.space.add(point, point_shape)
            self.space.add(pivot)

        return {'rod' : [point, point_shape], 'top' : [top, top_shape], 'pivots' : [pivot]}

    def update(self): #returns the state of the swing, which is the angular displacement
        swing_pos = self.objects['rod'][0].position
        top_pos = self.objects['top'][0].position
        swing_angle = 90-math.degrees(math.atan2(swing_pos[1]-top_pos[1], swing_pos[0]-top_pos[0]))
        return swing_angle
        


class HillClimbingAgent(): 
    
    '''
    Completely random convergence on solution, may take a long time to reach a solution. 
    Has chance of reaching a sub optimal local maxima, from which it is only by chance may the agent 
    find the global maxima.
    '''
        
    def __init__(self):
        self.state_dim = 1 #number of components in the state vector
        self.action_size = 3 #number of actions inside the action space
        self.build_model()
        
    def build_model(self):
        self.weights = 1e-4*np.random.rand(self.state_dim, self.action_size) #random weight matrix used by hill climbing algorithm
        self.best_reward = -np.inf #reward has to increase so that the random weights can converge
        self.best_weights = np.copy(self.weights) #current weighting 
        self.noise_scale = 1e-2 #scale of random noise
        
        
    def get_action(self, state):
        p = np.dot(state, self.weights)
        action = np.argmax(p) #highest value determined through vector multiplication with weights
        return action #returns action with highest p value

    def update_model(self,reward):
        if reward >= self.best_reward: #when the reward has increased
            self.best_reward = reward 
            self.best_weights = np.copy(self.weights) #the best weights are then those that obtained the greatest rewards
            self.noise_scale = max(self.noise_scale/2, 1e-3)#half the noise scale if currently at the highest rewards achieved
        else:
            self.noise_scale = min(self.noise_scale*2, 2)#mutliply the noise scale by two otherwise for more exploration of different weights
        
        self.weights = self.best_weights * self.noise_scale * np.random.rand(self.state_dim, self.action_size) #formula for updating of the states


class SwingFunctions():
    '''
    When intialized, creates the swinger and swing inside of the pymunk space
    Carries all the functions that retrieve the data from the swing itself
    Has a destruct function to remove the swinger and swing from the pymunk space at the end of each episode
    '''

    def __init__(self, space, configFile):
        self.space = space
        self.current_best_state = 0
        self.best_state = 0
        self.configFile = configFile
        self.swing = Swing(self.space, self.configFile)
        self.person = None
        self.font = pygame.font.SysFont('Arial', 20)

        if self.configFile['person']: #initializes swing by using the attributes within the file
            self.person = Person(space, self.swing.pos, mass=configFile['person'])
            self.seat = pymunk.PinJoint(self.swing.seat, self.person.objects['body'][0])
            self.seat.collide_bodies = False
            self.space.add(self.seat)
            
    
    def update(self): #updates the swinger and retrieves current angular displacement of the swing
        self.state = self.swing.update()
        self.person.update()
        if abs(self.state) > abs(self.current_best_state): #updates the best reward achieved throughout the episdoe
            self.current_best_state = abs(self.state)
        return self.state
    
    
    def destruct(self): #removes all the pymunk objects created within the environment, for the start of a new episode
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


    def eventHandler(self,action): #applies the actions on the environment
        if action == 0:
            self.person.knee_motor.rate = np.pi #rotates the legs forwards
        elif action == 1:
            self.person.knee_motor.rate = -np.pi #rotates the legs backwards
        else:
            self.person.knee_motor.rate = 0 #no action

            
    def render(self, screen, elapsed_time, ep, best_reward, old_reward, current_reward, steps):
        text = [f'Time Elasped: {round(elapsed_time, 1)}', f'Episode: {ep}', f'Best Reward: {round(best_reward,2)}', f'Previous Reward: {round(old_reward,2)}', 
                f'Current Reward: {round(current_reward,2)}']
        cnt = 0
        for i in text:
            render = self.font.render(i, True, (0,0,0))
            screen.blit(render, (0, cnt*20))
            cnt+=1
        


##################### Initialization#########################
config = loadConfig('config_swinger.json') #opens the text file 'config_swinger.json' 
pygame.init()
screen = pygame.display.set_mode((600,600)) #size of the pygame window
space = pymunk.Space()
space.gravity = (0.0, 900) #parameters: idle_speed_threshold, iterations (CPU usage)
draw_options = pymunk.pygame_util.DrawOptions(screen)
clock = pygame.time.Clock()

HC = HillClimbingAgent()

##################### Number of episodes #########################
num_episodes = 50
episode_length = 60

best_reward = 0
old_reward = 0


for ep in range(num_episodes):
    
    ##################### Variables reset at the start of each episode #########################
    SF = SwingFunctions(space, config['swingWithPerson']) #creates the swinger and swing
    elapsed_time = 0
    current_reward = 0

    steps = 0 #each step corresponds to the updating of the hill climbing algorithm
    start_time = pygame.time.get_ticks()

    
    while elapsed_time < episode_length*1000: #15seconds episodes
    
        ##################### Graphics #########################
        screen.fill((255,255,255))
        pygame.display.set_caption("Swing Machine Learning [Hill Climbing], Iteration : {}".format(ep+1)) #pygame window header
        SF.render(screen, elapsed_time/1000, ep+1, best_reward, old_reward, current_reward, steps) #refreshes GUI
        space.debug_draw(draw_options)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        ##################### Agent-Environment #########################
        state = SF.update() #retrieves the current angular displacement of the swing, which is the current state of the system
        current_reward = SF.current_best_state
        action = HC.get_action(state) #learning agent decides on an action 
        SF.eventHandler(action) #action is applied to the environment by learning agent

        space.step(1/60)
        clock.tick(60) #fps
        elapsed_time = pygame.time.get_ticks() - start_time
    
    HC.update_model(current_reward) #updates the weight matrix after the episode is over
    
    print("Episode: {}, best_reward: {:.2f}".format(ep, SF.current_best_state))
    if current_reward > best_reward: #updates the best reward obtained throughout the simulations
        best_reward = current_reward
    old_reward = current_reward
    
    SF.destruct() #removes the previous swinger and environment so the new episode can start

    
pygame.quit() #closes the pygame window
