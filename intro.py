# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:16:43 2021

@author: remib
"""

import pymunk
import pymunk.pygame_util
import pygame

GRAY = (220, 220, 220)
space = pymunk.Space()
space.gravity = 0, 10
b0 = space.static_body

class App:
    size = 700, 240
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.size)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.running = True

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.image.save(self.screen, 'intro.png')
                
                keys = pygame.key.get_pressed()
                
                #Controls 
                
                if keys[pygame.K_RIGHT]: #Right arrow
                    print("Right key pressed")
                elif keys[pygame.K_LEFT]: #Left arrow
                    print("Left key pressed")
                elif keys[pygame.K_UP]: #Up arrow
                    print("Up key pressed")
                elif keys[pygame.K_DOWN]: #Down arrow
                    print("Down key pressed")

            self.screen.fill(GRAY)
            space.debug_draw(self.draw_options)
            pygame.display.update()
            space.step(0.01)

        pygame.quit()

if __name__ == '__main__':
    p0, p1 = (0, 0), (700, 0)
    segment = pymunk.Segment(b0, p0, p1, 4)
    segment.elasticity = 1

    body = pymunk.Body(mass=1, moment=10)
    body.position = (100, 200)

    circle = pymunk.Circle(body, radius=30)
    circle.elasticity = 0.95
    space.add(body, circle, segment)

    App().run()