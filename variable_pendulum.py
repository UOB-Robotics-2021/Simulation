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
space.gravity = config["gravity"]
b0 = space.static_body

#Create objects
moment_of_inertia = 0.25*config["flywheelMass"]*config["flywheelRadius"]**2

body = pymunk.Body(mass=config["flywheelMass"], moment=moment_of_inertia)
body.position = config["flywheelInitialPosition"]
circle = pymunk.Circle(body, radius=config["flywheelRadius"])

#Create joints
joint = pymunk.SlideJoint(space.static_body, body, config["pivotPosition"], [0, 0], config["minPendulumLength"], config["maxPendulumLength"])
#joint = pymunk.PinJoint(space.static_body, body, config["pivotPosition"])
#joint = pymunk.PinJoint(space.static_body, body,[300,50], [0, 0])

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
    print(body.angle)
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.image.save(screen, 'intro.png')
        
        keys = pygame.key.get_pressed()
        
        #Controls 

        #Rotate flywheel clockwise when right arrow pressed
        if keys[pygame.K_RIGHT]: 
            print(body.angular_velocity)
            
            #Applies right-impulse to top of flywheel
            top = body.position[1] + config["flywheelRadius"]
            body.apply_impulse_at_local_point([-config["flywheelRadius"],0], pymunk.Vec2d(body.position[0], top))
        #Rotate flywheel counter-clockwise when left arrow pressed
        elif keys[pygame.K_LEFT]: 
            #Applies left-impulse to top of flywheel
            top = body.position[1] + config["flywheelRadius"]
            body.apply_impulse_at_local_point([config["flywheelRadius"],0], pymunk.Vec2d(body.position[0], top))
        elif keys[pygame.K_UP]: #Up arrow
            #Applies impulse torwards pivot
            v = (pymunk.Vec2d(body.position[0], body.position[1])- pymunk.Vec2d(300, 50))
            body.apply_impulse_at_local_point(v,  v)
        elif keys[pygame.K_DOWN]: #Down arrow
            #Applies impulse away from pivot
            v = (pymunk.Vec2d(body.position[0], body.position[1])- pymunk.Vec2d(300, 50))
            body.apply_impulse_at_local_point(v, -v)
        elif keys[pygame.K_SPACE]: #Down arrow
            #Resets spinning
            body.angular_velocity = 0
            
    screen.fill(GRAY)
    space.debug_draw(draw_options)
    pygame.display.update()
    #space.step(0.01)
    space.step(0.1)

pygame.quit()