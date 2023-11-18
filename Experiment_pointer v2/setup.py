"""
Setup

"""
from objects import *
from variables import *
from multiprocessing import shared_memory
import time
import sys
import pygame
import numpy as np

# test that location is readable
try:
    np.load(readLocation)
except FileNotFoundError:
    readLocation = '../' + readLocation
    np.load(readLocation)
    

rightHandIndex = processedRigidBodyParts.index('RHand')


debugger = Debugger(3)
debugger.disp(2,'Debug Level',debugger.debugLevel)

clock = pygame.time.Clock()
pygame.init()
world = pygame.display.set_mode([worldx,worldy]) # this is the surface
targetStartTime = 100000
targetBox = Box(leftCornerXBoxLoc,leftCornerYBoxLoc,boxWidth,boxHeight,RED,debugger)


player = Player(targetBox,colours, targetStartTime,worldx,worldy,debugger)   # spawn player
player.rect.x = worldx // 2   # go to x
player.rect.y = worldy // 2   # go to y
# put shared memory in player
if readData: # tell the cursor to read data
    player.prepareForDataRead(readLocation,handDataReadVarName)
    leftCornerXBoxLoc,leftCornerYBoxLoc = targetBox.prepareForDataRead(readLocation,targetBoxReadVarName,player)
    debugger.disp(3,'New Box seen in cursor object, x loc',player.targetBoxXmin)
    debugger.disp(3,'New Box seen in cursor object, y loc',player.targetBoxYmin)
if FETCHDATAFROMREALTIME:
    player.initSharedMemory(BODY_PART_MEM = 'Test Rigid Body',noDataTypes = 7,noBodyParts = 51)


programRunTime = timeProgram * 1000 # in ms
noTimeStamps = timeProgram * fps  # 40 fps times time in seconds


if recordData:
    
    player.prepareForDataWrite(noTimeStamps)
    boxLocs = np.zeros((100,2))
    targetBox.prepareForDataWrite(boxLocs)
    


player_list = pygame.sprite.Group()
player_list.add(player)

def endProgram(boxHitTimes,player):
    boxHitTimes = np.array(boxHitTimes)
    
    if player.datastore is not None:
        np.savez('test',boxHitTimes)
        print('box hit times:', boxHitTimes)
        player.processData()
        
        np.savez(writeDataLocation,dataStore = player.datastore,targetBoxLocs = targetBox.writeDatastore,
                 targetBoxHitTimes = boxHitTimes,targetBoxAppearTimes = player.targetAppearTimes,
                 allBodyPartsData = player.allBodyPartsDatastore)
    pygame.quit()
    sys.exit()
    main = False




steps = 20 # speed at which the cursor moves

if calibrated is False and FETCHDATAFROMREALTIME:
    player.enterCalibrationStage()
    calibrationTimeEnd = pygame.time.get_ticks() + 7000
    targetStartTime = calibrationTimeEnd + 2000
    player.targetStartTime = targetStartTime

else:
    targetStartTime = 2000
    player.targetStartTime = 2000
