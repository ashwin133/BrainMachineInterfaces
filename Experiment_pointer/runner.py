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
    # either get movement from data
    if FETCHDATAFROMREALTIME:
            player.fetchSharedMemoryData()
            if pygame.time.get_ticks() > calibrationTimeEnd:
                if calibrated is False:
                    targetStartTime = player.finishCalibrationStage()

                    calibrated = True
                else:
                    reachedBoxStatus = player.calcCursorPosFromHandData()
                    debugger.disp(4,'X',player.rightHandPos[1])
                    debugger.disp(4,'Y',player.rightHandPos[2])

    # get all key presses
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            endProgram(boxHitTimes,player)
        

        # or get movement from keypresses
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT or event.key == ord('a'):
                print('left')
                player.control(-steps,0)
            if event.key == pygame.K_RIGHT or event.key == ord('d'):
                print('right')
                player.control(steps,0)
            if event.key == pygame.K_UP or event.key == ord('w'):
                print('up')
                player.control(0,steps)
            if event.key == pygame.K_DOWN or event.key == ord('s'):
                print('down')
                player.control(0,-steps)

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == ord('a'):
                print('left stop')
                player.control(0,0)
                
            if event.key == pygame.K_RIGHT or event.key == ord('d'):
                print('right stop')
                player.control(0,0)
            
            if event.key == pygame.K_UP or event.key == ord('w'):
                print('up stop')
                player.control(0,0)
            
            if event.key == pygame.K_DOWN or event.key == ord('s'):
                print('down stop')
                player.control(0,0)
                
            if event.key == ord('q'):
                endProgram(boxHitTimes,player)
    
    if not FETCHDATAFROMREALTIME:
        # update based on requested movement and check if cursor has reached box
        reachedBoxStatus = player.update()

    if FETCHDATAFROMREALTIME:
        pass # just confirming that check for cursor has already taken

    # code does not depend on real time from here

    """
    DEPENDENT EVENTS
    """

    # end the program if the program run time has finished
    if pygame.time.get_ticks() > programRunTime:
        endProgram(boxHitTimes,player)

    # TRIGGERS WHEN CURSOR REACHES TARGET
    if reachedBoxStatus == 1 and reachedBoxLatch == 0:
        debugger.disp(2,'Hit Box','')
        timeToReach = pygame.time.get_ticks()
        boxHitTimes.append(timeToReach)
        targetBox.boxColor = GREEN
        reachedBoxLatch = 1




    # i.e. RESETS THE TARGET 3 SECONDS AFTER USER REACHES
    
    if timeToReach is not None and pygame.time.get_ticks() > timeToReach + 3000:
        # reset timeToReach to None as this is what shows the user has reached the target
        timeToReach = None
        # respawn the box
        targetBox.resetBoxLocation()
        reachedBoxLatch = 0
        # update the dimensions of the new box so the cursor knows when it has reached the box
        player.reset(targetBox)

    # draw the cursor
    world.fill(BLUE)
    player_list.draw(world) # draw player

    if pygame.time.get_ticks() > targetStartTime:
        # draw box

        pygame.draw.rect(world, targetBox.boxColor, pygame.Rect(targetBox.dimensions)) 

    # advance clock and display
    pygame.display.flip()
    clock.tick(fps)
