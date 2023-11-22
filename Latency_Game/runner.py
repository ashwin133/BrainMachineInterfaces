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
import time as time_

# import variables and objects used and run setup
from variables import *
from objects import *
from setup import *

sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')
avgLoopTimes = []


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
                endTime = time_.perf_counter()
                avgLoopTimes.append(endTime-timeCursorAppears)
                print("Reaction time is {}".format(time - timeCursorAppears))
                cursorAppearLatch = True
                reactionTimes.append(pygame.time.get_ticks() - time)
                timeCursorAppears = pygame.time.get_ticks() + np.random.randint(2000,5000)


        if event.type == pygame.KEYUP:
            if event.key == ord('g'):
                print('G released')
                
            if event.key == ord('q'):
                print(avgLoopTimes)
                print(np.average(np.array(avgLoopTimes)))
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
        timeCursorAppears = time_.perf_counter()
        player_list.draw(world) # draw cursor after
    # advance clock and display
    pygame.display.flip()
    clock.tick(fps)
