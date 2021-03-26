import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt 
from collections import deque
import math
import json
import random
import logging
import os

'''
  _____                     ____         _                           _             
 |  __ \                   / __ \       | |                         (_)            
 | |  | | ___  ___ _ __   | |  | |______| |     ___  __ _ _ __ _ __  _ _ __   __ _ 
 | |  | |/ _ \/ _ \ '_ \  | |  | |______| |    / _ \/ _` | '__| '_ \| | '_ \ / _` |
 | |__| |  __/  __/ |_) | | |__| |      | |___|  __/ (_| | |  | | | | | | | | (_| |
 |_____/ \___|\___| .__/   \___\_\      |______\___|\__,_|_|  |_| |_|_|_| |_|\__, |
                  | |                                                         __/ |
                  |_|                                                        |___/ Hoe Yin Chai
'''



########################## TENSORFLOW #################################
'''The following commands suppress potential issues with compatibility between dfferent tensorflow versions and a GPU bug'''

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 



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
        self.pos = Vec2d(*pos) #using vectors for the body part positions
        
        ######################## Mass and Body part lengths ############################
        self.mass = mass
        self.leg_length = 35
        self.torso_height = 50
        self.objects = self.generatePerson()
        
        ###################### Variables : Swinger Body Parts ##########################
        self.legs = self.objects['legs']
        self.torso = self.objects['body']
        self.knee_motor = self.objects['pivots'][2]
        self.hip_motor = self.objects['pivots'][4]

    def generatePerson(self):
        #mass distributions across the body parts were chosen based on modelling subgroup findings
        
        ########################### Torso ################################
        torso = pymunk.Body(0.49*self.mass, 100)  #parameters: mass, moment (inf moment for body, i.e rigid)
        torso.position = self.pos
        torso_shape = pymunk.Segment(torso, (-10,0), (-10, -self.torso_height), 3) #body, a(first endpoint of segment), b(second enpoint of segment), radius(thickness of segment)
        self.space.add(torso, torso_shape)
        
        ########################### Thigh ################################
        thigh = pymunk.Body(0.30*self.mass, np.inf) 
        thigh.position = self.pos
        thigh_shape = pymunk.Segment(thigh, (-10,0), (10, 0), 3)
        self.space.add(thigh, thigh_shape)
        
        ########################### Legs ################################
        legs = pymunk.Body(0.21*self.mass, 100)
        legs.position = self.pos
        legs_shape = pymunk.Segment(legs, (10, 0), (10, self.leg_length), 3)
        self.space.add(legs, legs_shape)

        ########################### Knee joint ################################
        knee_joint = pymunk.constraints.PinJoint(legs, thigh , (10, 0), (10,0)) #PinJoints used to simulate joints in the body
        knee_joint2 = pymunk.constraints.PinJoint(legs, thigh , (10, 35), (10,0)) #parameters: body 1, body 2, first anchor point, second anchor point
        knee_motor = pymunk.SimpleMotor(legs, thigh, 0) #motor is used to rotate the limb around the joint
        knee_joint.collide_bodies = False            
        knee_joint2.collide_bodies = False
        self.space.add(knee_joint, knee_joint2, knee_motor)
        
        ########################### Hip joint  ################################
        hip_joint = pymunk.PinJoint(torso, thigh, (-10, 0), (-10, 0))
        hip_motor = pymunk.SimpleMotor(torso, thigh, 0)
        hip_joint.collide_bodies = False
        self.space.add(hip_joint)
        self.space.add(hip_motor)
        
        return {'body' : [torso, torso_shape, thigh, thigh_shape], 'legs' : [legs, legs_shape], 'pivots' : [knee_joint, knee_joint2, knee_motor, hip_joint, hip_motor]}
    
            

class Swing():

    def __init__(self, space, swingConfig, swing_startangle):
        '''
        Creation of the swing itself
        '''
        
        self.space = space
        self.swing_startangle = swing_startangle*np.pi*1/180 #converts the user inputted swing angle into radians for python
        
        ########################### Pendulum length  ################################
        self.pendulum_length = 250
        
        self.objects = self.generateSwing(swingConfig, swing_startangle, self.pendulum_length)
        
        ########################### Swing seat  ################################
        self.seat = self.objects['rod'][0]
        self.seat_pos = self.seat.position


    def generateSwing(self, config, swing_startangle, pendulum_length):
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
            point = pymunk.Body(j, np.inf) #stiffness at point of connection to swing
            point.position = top.position + Vec2d(-pendulum_length*np.sin(self.swing_startangle),pendulum_length*np.cos(self.swing_startangle)) #The initial position of the swing, using the starting swing angle input 
            point_shape = pymunk.Segment(point, (0,0), (0,0), 0)
            

            pivot = pymunk.PinJoint(top, point, (0,0))
            
            self.space.add(point, point_shape)
            self.space.add(pivot)

        return {'rod' : [point, point_shape], 'top' : [top, top_shape], 'pivots' : [pivot]}
    
    


class DeepQNetwork():
    '''
    A deep Q network utilizes a neural network and Q-values to converge on an optimal policy for maximising
    the Q-value based on the actions the agent supplies back to the environment. The neural network and calculation 
    of the loss function, used for reinforcement learning, is implemented here.
    '''
    
    def __init__(self, state_dim, action_size): 
        #placeholders are the actual data points which are to be fed into the network
        '''defining the neurons/nodes within the neural network'''
        self.state_in = tf.placeholder(dtype=tf.float32, shape=[None, state_dim]) #the state input ; None means the placeholder accepts any dimension
        self.action_in = tf.placeholder(dtype=tf.int32, shape=[None]) #the action input 
        self.q_target_in = tf.placeholder(dtype=tf.float32, shape=[None]) #the target Q-value should be of the same type and shape as input placeholder
        self.global_step = tf.Variable(0, trainable=False) #global step tracks how many iterations of reinforcement learning has occured within the network; note that you cannot use a python var as updating is very slow
        
        #################### Machine learning parameter: Learning rate ########################## 
        self.learning_rate = tf.train.exponential_decay(0.01, self.global_step, 3600, 0.75, staircase=True) #parameters: initial value, current steps, steps to decay, fractional decay
        
        action_one_hot = tf.one_hot(self.action_in, depth=action_size) #one hot encoded vector, to convert action into a binary variable representation
        
        '''dense layers form the densely connected (interconnected) neural network, dense layer's hide complexity so we only need to specify outputs'''
        self.hidden1 = tf.layers.dense(self.state_in, 100, activation=tf.nn.relu) #paramters: input, nodes(connections) within layer, activation function 
        self.q_state = tf.layers.dense(self.hidden1, action_size, activation=None) #None type activation function is by default linear
        self.q_state_action = tf.reduce_sum(tf.multiply(self.q_state, action_one_hot), axis=1) #single Q-value for an state-action pair comes from multiplying Q-value with the one hot action vector 
        self.loss = tf.reduce_mean(tf.square(self.q_state_action - self.q_target_in)) #the definition of loss, mean squared error between Q-value and target Q-value, used to update the neural network
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, self.global_step) #optimizer algorithm object from tensorflow, based on the learning will update the neural network by minimizing loss
        
    def update_model(self, session, state, action, q_target):
        ''''updates the neural network, reinforcement occurs here where the loss function is minimized at each update (step)'''
        feed = {self.state_in: state, self.action_in: action, self.q_target_in: q_target}
        session.run(self.optimizer, feed_dict=feed)#sessions are run in tensorflow to obtain the value of a variable by updating it; here the optimizer is run to update the network
        
    def get_q_state(self, session, state): 
        '''retrieves the Q-value of a state, necessary to determine the next action'''
        q_state = session.run(self.q_state, feed_dict={self.state_in: state})
        return q_state




class ReplayBuffer(): 
    '''Makes training rare occurences more likely, by randomly training through batches of old experiences. 
    When it is full, the oldest experiences get pushed out. Enables more stable learning, as it 
    reduces the amount of learning lost as it explores new states'''
    
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen) #deque from python is a list where appending will automcally eject the oldest entries
        
    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size) #if the batch is not great enough, i.e at the start of the simulation, min is the size of the buffer
        samples = random.choices(self.buffer, k=sample_size) 
        return map(list, zip(*samples)) #seperates lists of tuples with lists of values for each tuple. these values are what the environment returned to the agent
    
    
    
    
class SwingFunctions():
    '''
    When intialized, creates the swinger and swing inside of the pymunk space
    Carries all the functions that retrieve the data from the swing itself
    Has a destruct function to remove the swinger and swing from the pymunk space at the end of each episode
    '''
    def __init__(self,space,configFile, swing_startangle):
        self.space = space
        self.configFile = configFile
        self.font = pygame.font.SysFont('Arial', 20)
        
        ####################### Swing and swinger creation ##############################
        self.swing = Swing(self.space, self.configFile, swing_startangle)
        self.person = None
        if self.configFile['person']: #initializes swing by using the attributes within the file
            self.person = Person(space, self.swing.seat_pos, mass=configFile['person'])
            self.seat = pymunk.PinJoint(self.swing.seat, self.person.objects['body'][0])
            self.seat.collide_bodies = False
            self.space.add(self.seat)

        ######### Tracking of the swing variables to calculate swing velocities ###########
        self.last_swing_angle = -swing_startangle
        self.last_leg_angle = 0
        self.last_torso_angle = 0
        
        ####################### Track variable ##############################
        '''Tracks the location of the swing at key locations to decide on when rewards should be issued'''
        self.track = 1 #starting at one means the swing is at maximum amplitude, zero represents the swing at equilibrium
        self.give_reward = 0 #controls when reward is issued, starts at zero for the beginning of the simulation
        
        ####################### Reward and best swing tracking ##############################
        self.reward = 0
        self.best_angle = 0
        
            
    def update(self, state):
        '''Updates the reward and best swing angle achieved'''
        if self.give_reward == 1:
            #################### Reward function ##########################
            self.reward += (abs(state[0])+abs(state[1]))**2 #reward function= (θ+ω)^2
            self.reward -= abs(state[6]) #subtracts effort parameter
            
            if self.reward < 0: #reward cannot be less than zero
                self.reward = 0
                
        if abs(state[0]) > abs(self.best_angle): #If current angular displacement of the swing is greater than best_angle
            self.best_angle = abs(state[0])
        
        self.give_reward = 0
        
        return self.reward
            
            
    def get_state(self, interval):
        ''' Calculation of all the different components which make up the state vector of the swinger '''
        
        #if interval < 1: #these two lines can be introduced when getting a division by zero error, which occurs when interval between steps is so small, that it is rounded to zero by python
            #interval = 1
        
        #################### Swing positions ##########################
        swing_pos = self.swing.objects['rod'][0].position
        top_pos = self.swing.objects['top'][0].position
        swing_angle = 90-math.degrees(math.atan2(swing_pos[1]-top_pos[1], swing_pos[0]-top_pos[0]))
        swing_ang_vel = (swing_angle-self.last_swing_angle)/(interval/100)

        #################### Swing at equilibrium ##########################
        '''Requires swing to be at the natural vertical length of initialized swing, and track = 1 which means it was previously at max amplitude'''
        if swing_pos[1] >= self.swing.pendulum_length+top_pos[1] and self.track == 1:
            self.track = 0
            self.give_reward = 1
            
        #################### Swing at max amplitude ##########################
        '''Requires swing angular velocity to be zero, and track = 0 which means it was previously at equilibrium'''
        if math.floor(swing_ang_vel) == 0 and self.track == 0:
            self.track = 1
            self.give_reward = 1
            
        #################### Leg and Torso positions ##########################    
        leg_angle = self.person.legs[0].angle * 180/np.pi #effort parameter uses angles in degrees
        torso_angle = self.person.torso[0].angle * 180/np.pi
        torso_ang_vel = (torso_angle-self.last_torso_angle)/(interval/100)
        leg_ang_vel = (leg_angle-self.last_leg_angle)/(interval/100)
        leg_length = self.person.leg_length/10 #the lengths of the leg and torso have been scaled for use in the effort parameter
        torso_height = self.person.torso_height/10 
        
        #################### Effort parameter ########################## 
        '''Uses the formula which was created for the sitting swinger by the modelling subgroup'''
        effort_legs = (self.person.legs[0].mass*((leg_length)**2)*((leg_angle-self.last_leg_angle)**2))/(interval)**2
        effort_torso = (self.person.torso[0].mass*((torso_height)**2)*((torso_angle-self.last_torso_angle)**2))/(interval)**2
        delta_leg_angle = (np.cos(abs(swing_angle)+abs(leg_angle))-np.cos(abs(self.last_swing_angle)+abs(self.last_leg_angle)))
        delta_torso_angle = (np.cos(abs(swing_angle)+abs(torso_angle))-np.cos(abs(self.last_swing_angle)+abs(self.last_torso_angle)))
        effort_gravity = 9.81*(self.person.legs[0].mass*leg_length*delta_leg_angle)+9.81*(self.person.torso[0].mass*torso_height*delta_torso_angle)
        effort = effort_legs + effort_torso + effort_gravity
         
        #################### Updating tracking variables ########################## 
        self.last_swing_angle = swing_angle
        self.last_leg_angle = leg_angle
        self.last_torso_angle = torso_angle
        
        return [swing_angle, swing_ang_vel, leg_angle, leg_ang_vel, torso_angle, torso_ang_vel, effort]

            
    def rotate(self,action):
        '''Converts the one hot encoded action variable into an action which can be applied to the swinger'''
        if action == 0:
            self.person.knee_motor.rate = -np.pi*(4/3) #rotates the legs backwards
        elif action == 1: 
            self.person.knee_motor.rate = np.pi*(4/3) #rotates the legs forwards
        elif action == 2:
            self.person.hip_motor.rate = 1 #rotates the torso forwards
        elif action == 3:
            self.person.hip_motor.rate = -1 #rotates the torso backwards
        else:
            pass #do nothing
        
        return(self.model_validity()) #returns whether the swinger is at a maximum position at either the legs or torso
        
    def model_validity(self):
        ''' Prevents the limbs from rotating too far (unphysical) by resetting the angle and ceasing motor rotation
            Returns the numbers corresponding to certain positions of the swinger for the action map'''
            
        if self.person.legs[0].angle > np.pi/4: # +pi/4 is 45deg clockwise
            self.person.legs[0].angle = np.pi/4
            self.person.knee_motor.rate = 0
            return(0) #zero in the action map, represents legs fully tucked in
            
        elif self.person.legs[0].angle < -np.pi/2: # -pi/2 is 90 deg anticlockwise
            self.person.legs[0].angle = -np.pi/2
            self.person.knee_motor.rate = 0
            return(1) #one in the action map, represents legs fully outstreched forwards
            
        elif self.person.torso[0].angle < -np.pi/4:
            self.person.torso[0].angle = -np.pi/4
            self.person.hip_motor.rate = 0
            return(2) #two in the action map, torso of swinger fully leant back
            
        if self.person.torso[0].angle > 0:
            self.person.torso[0].angle = 0
            self.person.hip_motor.rate = 0
            return(3) #three in the action map, represents torso fully upright
        
        return(4) #four in the action map, represents swinger not in any of the above states
            

    def eventHandler(self):
        '''Allows the user to control the motors by using the arrow keys, useful for testing the limits of the swinger'''
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.person.knee_motor.rate = -np.pi*(4/3) #pressing up, rotates the leg forwards
        elif keys[pygame.K_DOWN]:
            self.person.knee_motor.rate = np.pi*(4/3) #pressing up, rotates the leg backwards
        elif keys[pygame.K_LEFT]:
            self.person.hip_motor.rate = -np.pi #pressing left, rotates the torso backwards
        elif keys[pygame.K_RIGHT]:
            self.person.hip_motor.rate = np.pi #pressing right, rotates the torso forwads
        
    def destruct(self): 
        '''Removes all the objects involved with the swing and swinger so that the environment is reset for a new episode'''
        
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
        
        
    def render(self, screen, elapsed_time, ep, best_reward, old_reward, current_reward, swing_angle, steps):
        text = [f'Time Elasped: {round(elapsed_time, 1)}', f'Episode: {ep}', f'Best Reward: {round(best_reward,2)}', f'Previous Reward: {round(old_reward,2)}', 
                f'Current Reward: {round(current_reward,2)}', f'Current Best Swing Angle: {round(swing_angle, 2)}', f'Steps: {steps}']
        cnt = 0
        for i in text:
            render = self.font.render(i, True, (0,0,0))
            screen.blit(render, (0, cnt*20))
            cnt+=1

class DQNAgent(): 
    '''
    The Deep Q-Network agent uses the neural network to decide on actions and controls when the network is 'trained', or updated
    '''
    def __init__(self):
        #################### State and Action vector sizes ########################## 
        self.state_dim = 7 
        self.action_size = 5 
        
        #################### Neural network ########################## 
        self.q_network = DeepQNetwork(self.state_dim, self.action_size)
        
        #################### Replay buffer (Initial size) ########################## 
        self.replay_buffer = ReplayBuffer(maxlen=10000)
        
        #################### Machine learning parameters: Discount factor, Epsilon (Greed) ########################## 
        self.gamma = 0.97 #contributions of future reward on Q-values
        self.eps = 1.0 #probability of selecting an action randomly over the greedy choice
        
        #################### Tensorflow session ########################## 
        self.sess = tf.Session() #tensorflow session to run inputs through
        self.sess.run(tf.global_variables_initializer()) #globalizes all the variables for the network
        
        
    def get_action(self, state):
        ''' Uses the neural network to pick an action based on the Q-value of the given state, then decides on whether to pick this greedy choice or a completely random action'''
        q_state = self.q_network.get_q_state(self.sess, [state]) #retrieves the Q-value of the given state
        action_greedy = np.argmax(q_state) #picks action with the highest q value (Greedy choice)
        action_random = np.random.randint(self.action_size) #randomly chooses an action within the action space
        action = action_random if random.random() < self.eps else action_greedy #if randomly selected number betwwen 0-1 is less than eps factor, chooses random action. chance of this occuring decreases as eps decays.
        return action
    
    
    def train(self, state, action, next_state, reward, done): 
        '''Calculates the target Q-value to 'train' or update the neural network'''
        self.replay_buffer.add((state, action, next_state, reward, done)) #adds each new experience tuple to the buffer
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(50) #get lists of each experience by sampling from the buffer
        q_next_states = self.q_network.get_q_state(self.sess, next_states) #retrives the next Q-values from the neural network currently
        q_next_states[dones] = np.zeros([self.action_size]) #set all the finished states (episode over) to zero
        q_targets = rewards + self.gamma * np.max(q_next_states, axis=1) #formula for q target value using gamma (discount factor)
        self.q_network.update_model(self.sess, states, actions, q_targets) #updates the neural network by using the current state, action and predicted Q-values
        
        if done: self.eps = max(0.1, 0.99*self.eps) #eps decays after each episode as the network converges on an optimal solution; also has a minium value for exploration
    
    def __del__(self): #generally a good habit to close tensorflow sessions
        self.sess.close()
    


##################### Initialization#########################
config = loadConfig('config_swinger.json') #opens the text file 'config_swinger.json' 
pygame.init()
screen = pygame.display.set_mode((600,800)) #size of the pygame window
space = pymunk.Space()
space.gravity = (0.0, 900) #parameters: idle_speed_threshold, iterations (CPU usage)

'''damping factor can be applied to simulation, but can lead to instablities in the swinger motion if the swing is close to equilibrium'''
#space.damping = 0.99 #0.01% of velocity is lost each second

draw_options = pymunk.pygame_util.DrawOptions(screen)
clock = pygame.time.Clock()

DQN = DQNAgent()


##################### Initial swing startingangle #########################
swing_startangle = int(input("Swing starting angle: ")) 


##################### Episode parameters #########################
num_episodes = 250
episode_length = 120
eps = []
for ep in range(num_episodes):
    eps.append(ep+1) 
    
    
##################### Variables tracked for plotting #########################
rewards = []
best_angle = []
old_reward = 0
best_reward = 0
steps = 0

for ep in range(num_episodes):
    
    ##################### Variables reset at the start of each episode #########################
    SF = SwingFunctions(space, config['swingWithPerson'], swing_startangle) #creates the swinger and swing
    state = SF.get_state(math.inf) #initial state should have zero velocity values so time interval is given as infinity
    elapsed_time = 0
    current_reward = 0
    done = False #whether an episode is finished or not
    
    actions = []
    angles = []
    times = []
    
    start_time = pygame.time.get_ticks()

    while not done: 
        
        ##################### Graphics #########################
        screen.fill((255,255,255))
        pygame.display.set_caption("Swing Machine Learning [Q Learning], Iteration : {}".format(ep+1))
        SF.render(screen, elapsed_time/1000, ep+1, best_reward, old_reward, current_reward, SF.best_angle, steps)
        space.debug_draw(draw_options)
        pygame.display.flip()
        
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            
        
        #SF.eventHandler() #unhashtag to allow user control of the swinger while the simulations are being run

        ##################### Agent-Environment #########################
        action = DQN.get_action(state) #learning agent decides on an action based on current state of environment
        ext = SF.rotate(action) #returns any maximum extension of the swinger after applying action
        
        ################ Updating the state and reward #######################
        current_elapsed_time = pygame.time.get_ticks() - start_time

        next_state = SF.get_state(current_elapsed_time-elapsed_time)
        current_reward = SF.update(next_state)
        
        '''Checks whether the episode is over, in which case the done variable is set to true and the current state is supplied back to the network'''
        if elapsed_time >= episode_length*1000: 
            DQN.train(state, action, next_state, current_reward, True)
            done = True #exits the loop, new episode starts
            
        DQN.train(state, action, next_state, current_reward, done)
        
        space.step(1/60)
        clock.tick(60) #fps
        elapsed_time = pygame.time.get_ticks() - start_time
        
        '''Saves the angular displacement/actions taken and time elapsed for the swing at certain episodes which are controlled below'''
        if (ep+1) in [1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250]:
            times.append(elapsed_time/1000)
            angles.append(state[0])
            actions.append(ext)
        
        ################ Updated variables for next machine learning step in the episode #######################
        state = next_state
        steps += 1
        
    ''' Plots the angular displacement/actions taken against time for the swing at certain episodes which are controlled below
        The plots are saved as images to the folder containing the program'''
        
    if (ep+1) in [1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250]:
        angle_fig = plt.figure(num=1, figsize=(40, 10)) 
        angle_subplot = angle_fig.add_subplot(1, 1, 1)
        angle_subplot.plot(times, angles) 
        angle_subplot.set_xlabel("Time (s)", fontsize=20) 
        angle_subplot.set_ylabel("Swing Angle", fontsize=20) 
        
        #Angular displacement plot y axis (displacement) limits
        angle_subplot.set_ylim([-20,20]) 
        
        angle_subplot.set_title("Swing angle against time during episode "+str(ep+1), fontsize=28) 
        angle_fig.savefig("angles_ep{}.png".format(ep+1))
        
        action_fig = plt.figure(num=2, figsize=(50, 15)) 
        action_subplot = action_fig.add_subplot(1, 1, 1)
        action_subplot.scatter(times, actions) 
        action_subplot.set_xlabel("Time (s)", fontsize=26)  
        action_subplot.set_ylabel("Action taken", fontsize=26)
        action_subplot.set_title("Acions taken by ML Agent throughout episode "+str(ep+1), fontsize=36) 
        action_fig.savefig("actions_ep{}.png".format(ep+1))
        
        angle_fig.clear()
        action_fig.clear()
        
        
    best_angle.append(SF.best_angle)
    rewards.append(current_reward)
    print("Episode: {}, Reward: {:.2f}, Best Angle: {:.2f}".format(ep+1, current_reward, SF.best_angle))
    
    if current_reward > best_reward: #updates the best reward obtained throughout the simulations
        best_reward = current_reward
        
    old_reward = current_reward
    SF.destruct()  #removes the previous swinger and environment so the new episode can start
    

'''Saves the rewards, and best angular displacements achieved by the swinger across each episodes to an excel file
   This overwrites the excel file each time it is updated so a new file has to created for each simulation'''
df = DataFrame({'Episode': eps, 'Reward': rewards, 'Angle': best_angle})
df.to_excel('swing_analysis.xlsx', sheet_name='sheet1', index=False)


'''Tensorflow session needs to be closed after each simulation is finished, if not there will be errors when running the program
   If a simulation is stopped prematurely, reset the console before running the program again'''
DQN.__del__() #closes the tensorflow session once the simulation is finished to prevent errors due clashing tensorflow variables

pygame.quit() #closes the pygame window
