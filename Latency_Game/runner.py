"""
main file to incorporate pointer based experiment
"""
# information used from opensource.com/article/17/12/game-framework-python


# import all libraries
import pygame
import sys
import os
from multiprocessing import shared_memory
import numpy as np

# import variables and objects used and run setup
from variables import *
from objects import *
from setup import *

sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')



while main:


    
    # get all key presses
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            try:
                sys.exit()
            finally:
                main = False
        if event.type == pygame.KEYDOWN:
            if event.key == ord('g') and pygame.time.get_ticks() > timeCursorAppears and cursorAppearLatch == False:
                time = pygame.time.get_ticks()
                print("Reaction time is {}".format(time - timeCursorAppears))
                cursorAppearLatch = True
                reactionTimes.append(pygame.time.get_ticks() - time)
                timeCursorAppears = pygame.time.get_ticks() + np.random.randint(2000,5000)


        if event.type == pygame.KEYUP:
            if event.key == ord('g'):
                print('G released')
                
            if event.key == ord('q'):
                pygame.quit()
                sys.exit()
                main = False

    """
    DEPENDENT EVENTS
    """
    if cursorAppearLatch == True and pygame.time.get_ticks() > timeCursorAppears:
        cursorAppearLatch = False
    # draw the cursor
    world.fill(BLUE)
    if pygame.time.get_ticks() > timeCursorAppears:
        player_list.draw(world) # draw cursor after
    # advance clock and display
    pygame.display.flip()
    clock.tick(fps)
