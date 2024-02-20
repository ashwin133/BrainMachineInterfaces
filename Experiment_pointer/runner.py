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
        startTime = time.perf_counter()
        # either get movement from data
        if gameEngine.FETCHDATAFROMREALTIME:
                player.fetchSharedMemoryData()
                if pygame.time.get_ticks() > gameEngine.calibrationTimeEnd:
                    if gameEngine.calibrated is False:
                        
                        gameEngine.targetStartTime = player.finishCalibrationStage()

                        # Initialise ring
                        ring = Ring(center=(targetBox.leftCornerXBoxLoc + gameEngine.boxWidth // 2, targetBox.leftCornerYBoxLoc + gameEngine.boxHeight // 2), radius=20, color=gameEngine.colours['WHITE'], timeToEmpty=gameEngine.timeLimit // 1000,fps = gameEngine.fps,startOnTime = gameEngine.targetStartTime )
                        player.ring = ring

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
        if (gameEngine.reachedBoxStatus == 1 or gameEngine.reachedBoxStatus == "VOID") and gameEngine.reachedBoxLatch == 0:
            debugger.disp(2, "Target removed", gameEngine.reachedBoxStatus)
            gameEngine.timeToReach = pygame.time.get_ticks()
            if gameEngine.reachedBoxStatus == "VOID":
                gameEngine.boxHitTimes.append(-1)
                targetBox.boxColor = gameEngine.colours['BLACK']
                gameEngine.holdLatch = True
                player.ring.freeze = True
                player.ring.number = 0
                player.ring.end_angle = 0
            else:
                debugger.disp(2,'Hit Box','')
                gameEngine.boxHitTimes.append(gameEngine.timeToReach)
                targetBox.boxColor = gameEngine.colours['ORANGE']
                player.ring.freeze = True
                
                gameEngine.scoreMultiplier = 0
            gameEngine.reachedBoxLatch = 1
            


        if gameEngine.timeToReach is not None and gameEngine.holdLatch == False:
            # Reached the target and testing if it can remain in target
            gameEngine.scoreMultiplier += gameEngine.incrementalScoreMultiplier

            # Alter score multiplier in ring
            player.ring.scoreMultiplier = gameEngine.scoreMultiplier

            # Check if cursor in target
            cursorInTarget = pygame.Rect(targetBox.dimensions).colliderect(player.rect)
            if cursorInTarget and gameEngine.scoreMultiplier < gameEngine.maxScoreMultiplier:
                print("Cursor in target")
            else:
                print("Score Multiplier:", gameEngine.scoreMultiplier)
                targetBox.boxColor = gameEngine.colours['GREEN']
                print("Cursor has left target")
                gameEngine.holdLatch = True
                gameEngine.timeToReach = pygame.time.get_ticks()

                
                # Calculate player score
                player.score += int(player.ring.number * gameEngine.scoreMultiplier// 1)
                player.scoreUpdates.append(player.score)
                player.scoreUpdateTimes.append(pygame.time.get_ticks())
                gameEngine.red_bar.addLine(player.scoreUpdates[-1] - player.scoreUpdates[-2])


            # If cursor not in target turn latch off

        # i.e. RESETS THE TARGET 3 SECONDS AFTER USER REACHES
        
        if gameEngine.timeToReach is not None and pygame.time.get_ticks() > gameEngine.timeToReach + 3000 and gameEngine.holdLatch == True:
            # reset timeToReach to None as this is what shows the user has reached the target
            debugger.disp(2, "Reset executed", gameEngine.reachedBoxStatus)
            gameEngine.reachedBoxStatus = None
            gameEngine.timeToReach = None
            # respawn the box
            player = targetBox.resetBoxLocation(player,gameEngine)
            gameEngine.reachedBoxLatch = 0
            # update the dimensions of the new box so the cursor knows when it has reached the box
            player = player.reset(targetBox)

            gameEngine.holdLatch = False
            
        
        
        # Draw the world
        gameEngine.world.fill(gameEngine.colours['BLUE'])

        # Draw the red bar
        gameEngine.red_bar.update()
        gameEngine.red_bar.draw(gameEngine.world)

        debugger.disp(4,'Player X pos',player_list.sprites()[0],frequency = 20)

        # draw the cursor
        player_list.draw(gameEngine.world) # draw player

        
        gameEngine.world.blit(text_surface_x, (0.93 * gameEngine.worldx,0.9* gameEngine.worldy ))
        gameEngine.world.blit(text_surface_y, (0.97 * gameEngine.worldx,0.9* gameEngine.worldy ))

        # Display player score
        playerScoreSurface = player.font.render("Score: " + str(player.score), True, (255, 255, 255))  # White color
        gameEngine.world.blit(playerScoreSurface, (0.8 * gameEngine.worldx,0.1* gameEngine.worldy ))

        if pygame.time.get_ticks() > gameEngine.targetStartTime:
            # draw box
            pygame.draw.rect(gameEngine.world, targetBox.boxColor, pygame.Rect(targetBox.dimensions)) 


        # Draw the ring if it is there
        if hasattr(player,"ring"):
            player.ring.update()
            player.ring.draw(gameEngine.world)

        

        # advance clock and display
    
        
        pygame.display.update()
        endTime = time.perf_counter()
                    
        #print('loop time: ',endTime-startTime)
        clock.tick(gameEngine.fps)

#runGame(gameEngine,player,debugger,targetBox)

if __name__ == "__main__":
    # run setup
    import pygame
    player,targetBox,gameEngine, clock, player_list,debugger,cursorPredictor = runSetup(gameEngine=gameEngine)

    runGame(gameEngine,player,debugger,targetBox,player_list,cursorPredictor=cursorPredictor,clock = clock)