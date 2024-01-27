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
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')


# import variables and objects used and run setup
from Experiment_pointer.variables import gameEngine
from Experiment_pointer.objects import *
from Experiment_pointer.setup import runSetup, endProgram



def runGame(gameEngine,player,debugger,targetBox,player_list,cursorPredictor = None,clock = None):
    
    while gameEngine.main:
        timestart = time.perf_counter()
        debugger.disp(4,'Time of loop', timestart)
        # either get movement from data
        if gameEngine.FETCHDATAFROMREALTIME:
                player.fetchSharedMemoryData()
                if pygame.time.get_ticks() > gameEngine.calibrationTimeEnd:
                    if gameEngine.calibrated is False:
                        
                        gameEngine.targetStartTime = player.finishCalibrationStage()

                        gameEngine.calibrated = True
                    else:
                        gameEngine.reachedBoxStatus = player.calcCursorPosFromHandData(targetBox)
                        if gameEngine.showCursorPredictor is True:
                            cursorPredictor.calcCursorPosFromHandData(targetBox)
                        debugger.disp(4,'X',player.rightHandPos[1])
                        debugger.disp(4,'Y',player.rightHandPos[2])
        text_surface_x = player.font.render(str(player.rect.x), True, (255, 255, 255))  # White color
        text_surface_y = player.font.render(str(player.rect.y), True, (255, 255, 255))  # White color
# get all key presses
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return endProgram(gameEngine,player,targetBox,debugger)
            

            # or get movement from keypresses
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT or event.key == ord('a'):
                    print('left')
                    player.control(-gameEngine.steps,0)
                if event.key == pygame.K_RIGHT or event.key == ord('d'):
                    print('right')
                    player.control(gameEngine.steps,0)
                if event.key == pygame.K_UP or event.key == ord('w'):
                    print('up')
                    player.control(0,gameEngine.steps)
                if event.key == pygame.K_DOWN or event.key == ord('s'):
                    print('down')
                    player.control(0,-gameEngine.steps)

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
                    return endProgram(gameEngine,player,targetBox,debugger)
        
        if not gameEngine.FETCHDATAFROMREALTIME:
            #print('LOOP EXECUTED')
            # update based on requested movement and check if cursor has reached box
            gameEngine.reachedBoxStatus = player.update(targetBox)

        if gameEngine.FETCHDATAFROMREALTIME:
            pass # just confirming that check for cursor has already taken

        # code does not depend on real time from here

        """
        DEPENDENT EVENTS
        """

        # end the program if the program run time has finished
        if pygame.time.get_ticks() > gameEngine.programRunTime:
            return endProgram(gameEngine,player,targetBox,debugger)

        # TRIGGERS WHEN CURSOR REACHES TARGET
        if gameEngine.reachedBoxStatus == 1 and gameEngine.reachedBoxLatch == 0:
            debugger.disp(2,'Hit Box','')
            gameEngine.timeToReach = pygame.time.get_ticks()
            gameEngine.boxHitTimes.append(gameEngine.timeToReach)
            targetBox.boxColor = gameEngine.colours['GREEN']
            gameEngine.reachedBoxLatch = 1




        # i.e. RESETS THE TARGET 3 SECONDS AFTER USER REACHES
        
        if gameEngine.timeToReach is not None and pygame.time.get_ticks() > gameEngine.timeToReach + 3000:
            # reset timeToReach to None as this is what shows the user has reached the target
            gameEngine.timeToReach = None
            # respawn the box
            targetBox.resetBoxLocation(player)
            gameEngine.reachedBoxLatch = 0
            # update the dimensions of the new box so the cursor knows when it has reached the box
            player.reset(targetBox)

        # draw the cursor
        gameEngine.world.fill(gameEngine.colours['BLUE'])
        debugger.disp(4,'Player X pos',player_list.sprites()[0],frequency = 20)
        player_list.draw(gameEngine.world) # draw player

        
        gameEngine.world.blit(text_surface_x, (0.93 * gameEngine.worldx,0.9* gameEngine.worldy ))
        gameEngine.world.blit(text_surface_y, (0.97 * gameEngine.worldx,0.9* gameEngine.worldy ))
        if pygame.time.get_ticks() > gameEngine.targetStartTime:
            # draw box
            pygame.draw.rect(gameEngine.world, targetBox.boxColor, pygame.Rect(targetBox.dimensions)) 


        # advance clock and display

        
        pygame.display.update()
        clock.tick(gameEngine.fps)

#runGame(gameEngine,player,debugger,targetBox)

if __name__ == "__main__":
    # run setup
    import pygame
    player,targetBox,gameEngine, clock, player_list,debugger,cursorPredictor = runSetup(gameEngine=gameEngine)

    runGame(gameEngine,player,debugger,targetBox,player_list,cursorPredictor=cursorPredictor,clock = clock)