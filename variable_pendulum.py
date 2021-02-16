# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:47:28 2021

@author: remib
"""

#Import modules
import pymunk
import pymunk.pygame_util
import pygame

#Get pendulum config details
import json

def loadConfig(configFile):

    with open(f'{configFile}', 'r') as cfg:
        config = cfg.read()

    return json.loads(config)

config = loadConfig('config.json')["variablePendulumConfig"]
GRAY = (220, 220, 220)
space = pymunk.Space()
space.gravity = [0, config["g"]]
b0 = space.static_body

#Create objects
moment_of_inertia = 0.25*config["flywheelMass"]*config["flywheelRadius"]**2

body = pymunk.Body(mass=config["flywheelMass"], moment=moment_of_inertia)
body.position = config["flywheelInitialPosition"]
circle = pymunk.Circle(body, radius=config["flywheelRadius"])

#Create joints
joint = pymunk.PinJoint(space.static_body, body, config["pivotPosition"])

space.add(body, circle, joint)

#Initialize Simulation
pygame.init()
screen_size = 600, 500
screen = pygame.display.set_mode(screen_size)
draw_options = pymunk.pygame_util.DrawOptions(screen)
running = True
clock = pygame.time.Clock()

#Main Simulation loop
running = True 
upkey_pressed = 0
while running:
    #print(body.position)
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.image.save(screen, 'intro.png')
        
        keys = pygame.key.get_pressed()
        
        #Controls 

        #Rotate flywheel clockwise when right arrow pressed
        if keys[pygame.K_RIGHT]: 
            
            #Applies right-impulse to top of flywheel
            top = body.position[1] + config["flywheelRadius"]
            body.apply_impulse_at_local_point([-config["g"],0], pymunk.Vec2d(body.position[0], top))
        #Rotate flywheel counter-clockwise when left arrow pressed
        elif keys[pygame.K_LEFT]: 
            #Applies left-impulse to top of flywheel
            top = body.position[1] + config["flywheelRadius"]
            body.apply_impulse_at_local_point([config["g"],0], pymunk.Vec2d(body.position[0], top))
        elif keys[pygame.K_UP]: #Up arrow
            #Shortens pendulum
            joint.distance = 250
        elif keys[pygame.K_DOWN]: #Down arrow
            #Lengtens pendulum
            joint.distance = 350
        elif keys[pygame.K_i]: #i key
            #Resets spinning
            print("Pendulum length: ", joint.distance) 
            print("Angular velocity: ", body.angular_velocity)
            print("Position: ", body.position)
            print("Mass: ", body.mass)
        elif keys[pygame.K_SPACE]: #space arrow
            #Resets spinning
            body.angular_velocity = 0
            
    screen.fill(GRAY)
    space.debug_draw(draw_options)
    pygame.display.update()
    #space.step(0.01)
    space.step(0.01)

pygame.quit()