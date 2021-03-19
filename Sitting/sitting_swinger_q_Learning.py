import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import numpy as np
from pandas import DataFrame
import math
import json
import random
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


from collections import deque


def loadConfig(configFile):
    with open(f'{configFile}', 'r') as cfg:
        config = cfg.read()

    return json.loads(config)


class Person():

    def __init__(self, space, pos, mass=100):
        self.space = space
        self.pos = Vec2d(*pos) # vector position
        self.mass = mass
        self.objects = self.generatePerson()
        self.legs = self.objects['legs']
        self.knee_motor = self.objects['pivots'][2]

    def generatePerson(self):
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
    
            

class Swing():

    def __init__(self, space, swingConfig, swing_startangle):
        self.space = space
        self.swing_startangle = swing_startangle*np.pi*1/180
        self.pendulum_length = 250
        self.objects = self.generateSwing(swingConfig, swing_startangle, self.pendulum_length)
        self.seat = self.objects['rod'][0]
        self.pos = self.seat.position


    def generateSwing(self, config, swing_startangle, pendulum_length):
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
            point.position = top.position + Vec2d(-pendulum_length*np.sin(self.swing_startangle),pendulum_length*np.cos(self.swing_startangle)) #The initial position of the swing 
            point_shape = pymunk.Segment(point, (0,0), (0,0), 0)
            # if the first joint, join to the top, otherwise join to the preceding joint (removed temporarily)
            pivot = pymunk.PinJoint(top, point, (0,0))
            self.space.add(point, point_shape)
            self.space.add(pivot)

        return {'rod' : [point, point_shape], 'top' : [top, top_shape], 'pivots' : [pivot]}
    
    


class QNetwork():
    def __init__(self, state_dim, action_size): #input, output size of network
        #placeholders are the actual data points needed to be fed into the network
        self.state_in = tf.placeholder(dtype=tf.float32, shape=[None, state_dim]) # None means placeholder accepts any dimension
        self.action_in = tf.placeholder(dtype=tf.int32, shape=[None])
        self.q_target_in = tf.placeholder(dtype=tf.float32, shape=[None]) #target should be of same type and shape as input placeholder
        self.global_step = 0
        self.learning_rate = tf.train.exponential_decay(0.01, self.global_step, 1, 0.01, staircase=True) #initial rate, steps to decay, fractional decay
        action_one_hot = tf.one_hot(self.action_in, depth=action_size) #one hot encoded vector
        
        #dense layers - output q values vector, is handled by dense layer in tensorflow which we only specify outputs
        self.hidden1 = tf.layers.dense(self.state_in, 100, activation=tf.nn.relu) #first arguement is input, second input nodes, third is default for layers (0, positive)
        self.q_state = tf.layers.dense(self.hidden1, action_size, activation=None)
        self.q_state_action = tf.reduce_sum(tf.multiply(self.q_state, action_one_hot), axis=1) #single q value for state action pair comes from multiplying states q values by one hot action vector 
        self.loss = tf.reduce_mean(tf.square(self.q_state_action - self.q_target_in)) #the definition of loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss) #optimizer algorithm object from tensorflow
    
        
    def update_model(self, session, state, action, q_target):
        feed = {self.state_in: state, self.action_in: action, self.q_target_in: q_target}
        session.run(self.optimizer, feed_dict=feed)#pass firstly the operation we want the output of, then the dictionary mapping data place holders
        
    def get_q_state(self, session, state): #get the q state output necessary to determine next action
        q_state = session.run(self.q_state, feed_dict={self.state_in: state})
        return q_state


class ReplayBuffer(): #makes training rare occurences more likely, more training batches of old experiences. When it is full, the oldest experiences get pushed out
    # more stable learning, does not risk losing what is already learnt as often.
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        
    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size) #if batch is not great enough, min is the size of the buffer
        samples = random.choices(self.buffer, k=sample_size) #seperates lists of tuples with lists of values for each tuple
        return map(list, zip(*samples))
    
class SwingFunctions():
    def __init__(self,space,configFile, swing_startangle):
        self.space = space
        self.configFile = configFile
        
        self.swing = Swing(self.space, self.configFile, swing_startangle)
        self.person = None
        self.last_bod_pos = self.swing.seat.position[0],self.swing.seat.position[1]
        self.track = 1 #would need to change should starting from stationary position
        
        self.give_reward = 0
        self.reward = 0
        self.best_angle = 0
        
        self.font = pygame.font.SysFont('Arial', 20)
    
        if self.configFile['person']: #initializes swing by using the attributes within the file
            self.person = Person(space, self.swing.pos, mass=configFile['person'])
            self.seat = pymunk.PinJoint(self.swing.seat, self.person.objects['body'][0])
            self.seat.collide_bodies = False
            self.space.add(self.seat)
            
        self.last_leg_pos =  int(self.person.legs[0].position[0]),int(self.person.legs[0].position[1])
            
    def update(self, state): #update model also included?
        if self.give_reward == 1:
            self.reward += (abs(state[0])+(state[1]**2))
        if abs(state[0]) > abs(self.best_angle):
            self.best_angle = abs(state[0])
            #self.reward = abs(state[0])
        self.give_reward = 0
        return self.reward
            
    def render(self, screen, elapsed_time, ep, best_reward, old_reward, current_reward, swing_angle, steps):
        text = [f'Time Elasped: {round(elapsed_time, 1)}', f'Episode: {ep}', f'Best Reward: {round(best_reward,2)}', f'Previous Reward: {round(old_reward,2)}', 
                f'Current Reward: {round(current_reward,2)}', f'Current Best Swing Angle: {round(swing_angle, 2)}', f'Steps: {steps}']
        cnt = 0
        for i in text:
            render = self.font.render(i, True, (0,0,0))
            screen.blit(render, (0, cnt*20))
            cnt+=1
            
    def get_state(self, interval):
        swing_pos = self.swing.objects['rod'][0].position
        top_pos = self.swing.objects['top'][0].position
        leg_angle = self.person.legs[0].angle
        swing_angle = 90-math.degrees(math.atan2(swing_pos[1]-top_pos[1], swing_pos[0]-top_pos[0]))
        current_bod_position = int(self.swing.seat.position[0]),int(self.swing.seat.position[1])
        current_leg_position = int(self.person.legs[0].position[0]),int(self.person.legs[0].position[1])
        swing_speed = math.hypot(self.last_bod_pos[0]-current_bod_position[0], self.last_bod_pos[1]-current_bod_position[1])/interval
        leg_speed = math.hypot(self.last_leg_pos[0]-current_leg_position[0], self.last_leg_pos[1]-current_leg_position[1])/interval
        #if swing_angle == 0 and self.track == 1: #is not reliable enough, would need to round down which is very inaccurate
            #print("centre")
            #self.track = 0
        if swing_pos[1] >= self.swing.pendulum_length+top_pos[1] and self.track == 1:
            self.track = 0
            self.give_reward = 1
            
        if swing_speed == 0 and self.track == 0:
            self.track = 1
            self.give_reward = 1
            
        self.last_bod_pos = current_bod_position
        self.last_leg_pos = current_leg_position
        
        return [swing_angle, swing_speed, leg_angle, leg_speed]
    
            
    def rotate(self,action):
        if action == 0:
            #pass
            self.person.knee_motor.rate = -np.pi
                
        elif action == 1:
            #pass
            self.person.knee_motor.rate = np.pi
            
        else:
            pass
        
        self.model_validity()
        
    def model_validity(self):
        if self.person.legs[0].angle > np.pi/4:
            self.person.legs[0].angle = np.pi/4
            
        elif self.person.legs[0].angle < -np.pi/2:
            self.person.legs[0].angle = -np.pi/2
        
    def eventHandler(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.person.knee_motor.rate = np.pi
        elif keys[pygame.K_DOWN]:
            self.person.knee_motor.rate = -np.pi
        
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

class DQNAgent(): #random and may not get best solution in a long time (sub optimal local maxima) --> random restart?
    def __init__(self):
        self.state_dim = 4 #can be controlled from file?
        self.action_size = 3 #currently trivial
        self.q_network = QNetwork(self.state_dim, self.action_size)
        self.replay_buffer = ReplayBuffer(maxlen=10000)
        self.gamma = 0.97 #future reward valuability
        self.eps = 1.0 # probability of selecting an action randomly over the greedy choice, starts at one for always randomly exploring at the start (prevents local max)
        
        self.sess = tf.Session() #tensorflow session to run inputs through the graph
        self.sess.run(tf.global_variables_initializer()) #globalizes all the variables for the network
        
    def get_action(self, state): #needs to return action for highest Q for a given state 
        q_state = self.q_network.get_q_state(self.sess, [state])
        action_greedy = np.argmax(q_state) #pick action with highest q value
        action_random = np.random.randint(self.action_size) #pick random action 
        action = action_random if random.random() < self.eps else action_greedy #if randomly selected number betwwen 0-1 is less than eps
        return action
    
    def train(self, state, action, next_state, reward, done): #needs to calculate target q value to train network
        self.replay_buffer.add((state, action, next_state, reward, done))#add each new experience tuple to the buffer
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(50) #get lists of each experience time by sampling from the buffer
        q_next_states = self.q_network.get_q_state(self.sess, next_states) #using next states
        q_next_states[dones] = np.zeros([self.action_size]) #set all finished states to zero
        q_targets = rewards + self.gamma * np.max(q_next_states, axis=1) #formula for q target value using gamma (learning rate)
        self.q_network.update_model(self.sess, states, actions, q_targets)
        
        #if done: self.eps = max(0.1, 0.97*self.eps) #decrease eps after each episode as the network improves, also has a minium value for exploration
        self.eps = max(0.1, 0.99*self.eps) #decrease eps after each episode as the network improves, also has a minium value for exploration
        #print(self.eps)
    
    def __del__(self): #need to close tensorflow session
        self.sess.close()
    
    
config = loadConfig('config_swinger.json') #text file located in the folder

pygame.init()
screen = pygame.display.set_mode((600,800))
draw_options = pymunk.pygame_util.DrawOptions(screen)
pygame.display.set_caption('Swing Machine Learning')
clock = pygame.time.Clock()

space = pymunk.Space()
space.gravity = (0.0, 900) #idle_speed_threshold, iterations (CPU usage)


swing_startangle = int(input("Swing starting angle: ")) 

num_episodes = 200
eps = []
rewards = []
angles = []

for ep in range(num_episodes):
    eps.append(ep+1) 
    
old_reward = 0
best_reward = 0
steps = 0
DQN = DQNAgent()
for ep in range(num_episodes):
    SF = SwingFunctions(space, config['swingWithPerson'], swing_startangle)
    state = SF.get_state(math.inf)
    elapsed_time = 0
    current_reward = 0
    done = False
    start_time = pygame.time.get_ticks()
    while not done: 
        screen.fill((255,255,255))
        pygame.display.set_caption("Swing Machine Learning [Q Learning], Iteration : {}".format(ep+1))
        SF.render(screen, elapsed_time/1000, ep+1, best_reward, old_reward, current_reward, SF.best_angle, steps)
        space.debug_draw(draw_options)
        
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        action = DQN.get_action(state)
        #print(str(action)+" time:" +str(elapsed_time/1000))
        #SF.eventHandler()
    
        SF.rotate(action)
        next_state = SF.get_state(pygame.time.get_ticks()-elapsed_time)
        current_reward = SF.update(next_state)
        if elapsed_time >= 60*1000: #15seconds episodes
            DQN.train(state, action, next_state, current_reward, True)
            done = True
        DQN.train(state, action, next_state, current_reward, done)
        space.step(1/60)

        clock.tick(60) #fps
        
        elapsed_time = pygame.time.get_ticks() - start_time
        
        state = next_state
        steps += 1
        
    
    rewards.append(current_reward)
    angles.append(SF.best_angle)
    print("Episode: {}, Reward: {:.2f}, Best Angle: {:.2f}".format(ep+1, current_reward, SF.best_angle))
    #print(DQN.sess.run(DQN.q_network.learning_rate))
    if current_reward > best_reward:
        best_reward = current_reward
    old_reward = current_reward
    DQN.q_network.global_step += 1
    SF.destruct()
    

df = DataFrame({'Episode': eps, 'Reward': rewards, 'Angle': angles})
df.to_excel('swing_analysis4.xlsx', sheet_name='sheet1', index=False)
pygame.quit()
