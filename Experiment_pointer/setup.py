"""
Setup

"""
import sys
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')

from Experiment_pointer.objects import *
from Experiment_pointer.variables import gameEngine
from multiprocessing import shared_memory
import time
import sys
import pygame
import numpy as np


def runSetup(gameEngine):
    # test that location is readable

    if gameEngine.readData:
        try:
            np.load(gameEngine.readLocation)
        except FileNotFoundError:
            try:
                gameEngine.readLocation_ = '../' + gameEngine.readLocation
                np.load(gameEngine.readLocation_)
                gameEngine.readLocation = gameEngine.readLocation_
            except FileNotFoundError:
                gameEngine.readLocation = 'Experiment_pointer/' + gameEngine.readLocation
                np.load(gameEngine.readLocation)
                

    rightHandIndex = gameEngine.processedRigidBodyParts.index('RHand')


    debugger = Debugger(3)
    debugger.disp(2,'Debug Level',debugger.debugLevel)
    debugger.test = gameEngine.testMode

    clock = pygame.time.Clock()
    pygame.init()
    gameEngine.world = pygame.display.set_mode([gameEngine.worldx,gameEngine.worldy]) # this is the surface
    gameEngine.targetStartTime = 100000
    targetBox = Box(gameEngine.leftCornerXBoxLoc,gameEngine.leftCornerYBoxLoc,gameEngine.boxWidth,gameEngine.boxHeight,gameEngine.colours['RED'],debugger)


    player = Player(targetBox,gameEngine.colours, gameEngine.targetStartTime,gameEngine.worldx,gameEngine.worldy,debugger)   # spawn player
    player.rect.x = gameEngine.worldx // 2   # go to x
    player.rect.y = gameEngine.worldy // 2   # go to y
    # put shared memory in player
    if gameEngine.readData: # tell the cursor to read data
        player.prepareForDataRead(gameEngine.readLocation,gameEngine.handDataReadVarName)
        leftCornerXBoxLoc,leftCornerYBoxLoc = targetBox.prepareForDataRead(gameEngine.readLocation,gameEngine.targetBoxReadVarName,player)
        debugger.disp(3,'New Box seen in cursor object, x loc',player.targetBoxXmin)
        debugger.disp(3,'New Box seen in cursor object, y loc',player.targetBoxYmin)
    if gameEngine.FETCHDATAFROMREALTIME:
        player.initSharedMemory(BODY_PART_MEM = 'Test Rigid Body',noDataTypes = 7,noBodyParts = 51)


    gameEngine.programRunTime = gameEngine.timeProgram * 1000 # in ms
    gameEngine.noTimeStamps = gameEngine.timeProgram * gameEngine.fps  # 40 fps times time in seconds


    if gameEngine.recordData or debugger.test:
        
        player.prepareForDataWrite(gameEngine.noTimeStamps)
        boxLocs = np.zeros((100,2))
        targetBox.prepareForDataWrite(boxLocs)
        


    player_list = pygame.sprite.Group()
    player_list.add(player)

    




    gameEngine.steps =  4 # speed at which the cursor moves

    if gameEngine.calibrated is False and gameEngine.FETCHDATAFROMREALTIME:
        player.enterCalibrationStage()
        gameEngine.calibrationTimeEnd = pygame.time.get_ticks() + 7000
        gameEngine.targetStartTime = gameEngine.calibrationTimeEnd + 2000
        player.targetStartTime = gameEngine.targetStartTime

    else:
        gameEngine.targetStartTime = 2000
        player.targetStartTime = 2000

    if gameEngine.LATENCY_TEST:
        player.latencyTestActivated = True
        targetBox.latencyTestActivated = True
    return player,targetBox,gameEngine, clock, player_list,debugger, player_list

def endProgram(gameEngine,player,targetBox,debugger):
        gameEngine.boxHitTimes = np.array(gameEngine.boxHitTimes)
        
        if gameEngine.recordData is True:
            np.savez('test',gameEngine.boxHitTimes)
            print('box hit times:', gameEngine.boxHitTimes)
            player.processData()
            del gameEngine.world
            np.savez(gameEngine.writeDataLocation,dataStore = player.datastore,targetBoxLocs = targetBox.writeDatastore,
                    targetBoxHitTimes = gameEngine.boxHitTimes,targetBoxAppearTimes = player.targetAppearTimes,
                    allBodyPartsData = player.allBodyPartsDatastore,boxSizeVarName = (gameEngine.boxHeight,gameEngine.boxWidth),
                    metadataLocation = gameEngine.metadataLocation,cursorMotionDatastoreLocation = player.cursorDatastore,gameEngineLocation = gameEngine)
        

        if debugger.test:
            outputDict = debugger.returnDebuggingOutput(player.datastore,targetBox.writeDatastore, gameEngine.boxHitTimes ,player.targetAppearTimes,player.allBodyPartsDatastore,
                                        (gameEngine.boxHeight,gameEngine.boxWidth),gameEngine.metadataLocation,player.cursorDatastore,gameEngine = gameEngine)
            pygame.quit()
            gameEngine.main = False
            return outputDict
        else:
            sys.exit()
        