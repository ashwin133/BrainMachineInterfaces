"""
Stores objects required for the game

"""

# import libraries
import pygame
import os
import numpy as np




class Cursor(pygame.sprite.Sprite):
    """
    Spawn a cursor
    """

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.images = []
        #load the pointer image
        try:
            img = pygame.image.load(os.path.join('Experiment_pointer/images', 'dot.png')).convert()
        except FileNotFoundError:
            img = pygame.image.load(os.path.join('images', 'dot.png')).convert()
        self.images.append(img)
        self.image = self.images[0]
        self.rect = self.image.get_rect()
        self.movex = 0 # move along X
        self.movey = 0 # move along Y
        


 
