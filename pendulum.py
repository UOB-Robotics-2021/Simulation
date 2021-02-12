# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:16:52 2021

@author: remib
"""

from intro import pymunk, space, App


b0 = space.static_body 

body = pymunk.Body(mass=1, moment=10)
body.position = (250, 0)
circle = pymunk.Circle(body, radius=20)
circle.angular_velocity = 100


joint = pymunk.PinJoint(b0, body, (350, 100))
space.add(body, circle, joint)

App().run()