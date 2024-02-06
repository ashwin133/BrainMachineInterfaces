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
import pickle


def runSetup(gameEngine):
    # test that location is readable
    # change this function after it works
    
                    

     
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
    gameEngine.world = pygame.display.set_mode([gameEngine.worldx,gameEngine.worldy],pygame.DOUBLEBUF) # this is the surface
    gameEngine.targetStartTime = 100000
    targetBox = Box(gameEngine.leftCornerXBoxLoc,gameEngine.leftCornerYBoxLoc,gameEngine.boxWidth,gameEngine.boxHeight,gameEngine.colours['RED'],debugger)
    

    player = Player(targetBox,gameEngine.colours, gameEngine.targetStartTime,gameEngine.worldx,gameEngine.worldy,debugger)   # spawn player
    
    # Set ability for player to use rotation of control body instead of position
    if gameEngine.useRotation:
        player.useRotation = True
    else:
        player.useRotation = False

    player.rect.x = gameEngine.worldx // 2   # go to x
    player.rect.y = gameEngine.worldy // 2   # go to y
    # put shared memory in player
    if gameEngine.readData: # tell the cursor to read data
        if gameEngine.readRigidBodies:
            player.simulateSharedMemoryOn = True
            if gameEngine.readAdjustedRigidBodies:
                player.readAdjustedRigidBodies = True
        if gameEngine.showCursorPredictor:
            if gameEngine.retrieveCursorDataFromModelFile == True:
                gameEngine.cursorMotionDatastoreLocation = gameEngine.modelReadLocation
            try:
                np.load(gameEngine.cursorMotionDatastoreLocation)
            except FileNotFoundError:
                try:
                    gameEngine.cursorMotionDatastoreLocation_ = '../' + gameEngine.cursorMotionDatastoreLocation
                    np.load(gameEngine.cursorMotionDatastoreLocation)
                    gameEngine.cursorMotionDatastoreLocation = gameEngine.cursorMotionDatastoreLocation_
                except FileNotFoundError:
                    gameEngine.cursorMotionDatastoreLocation = 'Experiment_pointer/' + gameEngine.cursorMotionDatastoreLocation
                    np.load(gameEngine.cursorMotionDatastoreLocation)
            if gameEngine.retrieveCursorDataFromModelFile == True:
                dataLocation = np.load(gameEngine.cursorMotionDatastoreLocation)['predCursorPos']
            else:
                dataLocation = np.load(gameEngine.cursorMotionDatastoreLocation)['cursorPred']
            
            cursorPredictor = Player(targetBox,gameEngine.colours,100000,gameEngine.worldx,gameEngine.worldy,debugger,gameEngine.showCursorPredictor,dataLocation) # never generate box
                    
        player.prepareForDataRead(gameEngine.readLocation,gameEngine.handDataReadVarName,gameEngine.allBodyDataVarName)
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
    
    if gameEngine.runDecoderInLoop:
        gameEngine.decoderStartTime = 10000
        gameEngine.decodeFromPreviousData = False # 
        if gameEngine.decodeFromPreviousData is False:
            try:
                np.load(gameEngine.modelReadLocation)
            except FileNotFoundError:
                try:
                    gameEngine.modelReadLocation_ = '../' + gameEngine.modelReadLocation
                    np.load(gameEngine.modelReadLocation_)
                    gameEngine.modelReadLocation = gameEngine.modelReadLocation_
                except FileNotFoundError:
                    gameEngine.modelReadLocation = 'Experiment_pointer/' + gameEngine.modelReadLocation
                    np.load(gameEngine.modelReadLocation)
            player.setupLiveDecoding(gameEngine)
    
    print('111')

    if gameEngine.showCursorPredictor:
        player_list.add(cursorPredictor)
        print('222')
        return player,targetBox,gameEngine, clock, player_list,debugger, cursorPredictor
    else:
        return player,targetBox,gameEngine, clock, player_list,debugger, None

def endProgram(gameEngine,player,targetBox,debugger):
        gameEngine.boxHitTimes = np.array(gameEngine.boxHitTimes)
        
        if gameEngine.recordData is True:
            np.savez('test',gameEngine.boxHitTimes)
            print('box hit times:', gameEngine.boxHitTimes)
            player.processData()
            del gameEngine.world
            del player.images
            del player.image
            del player.font
            if os.getcwd() == "/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces":
                if "Experiment_pointer" not in gameEngine.writeDataLocation:
                    os.chdir(os.getcwd() + "/Experiment_pointer")

            np.savez(gameEngine.writeDataLocation,dataStore = player.datastore,targetBoxLocs = targetBox.writeDatastore,
                    targetBoxHitTimes = gameEngine.boxHitTimes,targetBoxAppearTimes = player.targetAppearTimes,
                    allBodyPartsData = player.allBodyPartsDatastore,boxSizeVarName = (gameEngine.boxHeight,gameEngine.boxWidth),
                    metadataLocation = gameEngine.metadataLocation,cursorMotionDatastoreLocation = player.cursorDatastore,gameEngineLocation = gameEngine)

            try:
                with open(gameEngine.writeDataLocationPkl, 'wb') as file:
                    pickle.dump([gameEngine,player], file)
            except:
                del player.model
                with open(gameEngine.writeDataLocationPkl, 'wb') as file:
                    pickle.dump([gameEngine,player], file)
        if debugger.test:
            outputDict = debugger.returnDebuggingOutput(player.datastore,targetBox.writeDatastore, gameEngine.boxHitTimes ,player.targetAppearTimes,player.allBodyPartsDatastore,
                                        (gameEngine.boxHeight,gameEngine.boxWidth),gameEngine.metadataLocation,player.cursorDatastore,gameEngine = gameEngine)
            pygame.quit()
            gameEngine.main = False
            return outputDict
        else:
            pygame.quit()
            gameEngine.main = False
            return 
        