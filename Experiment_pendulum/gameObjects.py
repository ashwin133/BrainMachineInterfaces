import os
import pygame as pg
import config
import numpy as np
import math

main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, "images")

class Pendulum(pg.sprite.Sprite):

    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        self.image= pg.image.load(os.path.join(data_dir, "pendulum.png")).convert_alpha()
        # self.image = pg.transform.scale(self.image, (32,34))
        self.rect = self.image.get_rect()
        self.rect.center = pg.display.get_window_size()[0]/2, 300
        self.angle = 0
        self.ang_vel = 0
        self.move = 0
        self.mass = config.PENDULUM['mass']
        self.length = config.PENDULUM['length']
        self.rod_start = self.rect.midbottom

    def update(self):
        screen = pg.display.get_surface()
        pos = pg.mouse.get_pos()
        self.rect.center = pos[0] + config.CART['width']/2 + config.PENDULUM['rod_length']*np.sin((self.angle)), 300 - config.PENDULUM['rod_length']*(1 - np.cos((self.angle)))
        

class Cart(pg.sprite.Sprite):
    
    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        self.image= pg.image.load(os.path.join(data_dir, "cart.png")).convert_alpha()
        self.image = pg.transform.scale(self.image, (config.CART['width'],config.CART['height']))
        self.rect = self.image.get_rect()
        self.rect.midleft = pg.display.get_window_size()[0]/2 - config.CART['width']/2, 300 + config.PENDULUM['rod_length']
        self.pos =  pg.display.get_window_size()[0]/2
        self.vel = 0     
        self.move = 0
        self.mass = config.CART['mass']
        self.rod_end = self.rect.midtop

    def update(self):
        screen = pg.display.get_surface()
        pos = pg.mouse.get_pos()[0]
        self.pos = pos - 640
        self.rect.midleft = pos, 300 + config.PENDULUM['rod_length']
        self.pos = pos - pg.display.get_window_size()[0]/2

    

# class Rod(pg.sprite.Sprite):
    
#     def __init__(self):
#         pg.sprite.Sprite.__init__(self)
#         self.image = pg.draw.line(pg.display.get_surface(), config.BACKGROUND['rod_colour'],  )




