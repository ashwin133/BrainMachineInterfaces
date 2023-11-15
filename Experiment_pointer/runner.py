"""
main file to incorporate pointer based experiment
"""
# information used from opensource.com/article/17/12/game-framework-python

import pygame
import sys
import os
from multiprocessing import shared_memory
import numpy as np

sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')

"""
Variables
"""
#toggle screen size
worldx = 960
worldy = 720

fps = 40 # frame rate
ani = 4 # animation cycles # animate simple movements repeatedly

# colours
BLUE = (25, 25, 200)
BLACK = (23, 23, 23)
WHITE = (254, 254, 254)
RED  = (255,0,0)
GREEN = (0,255,0)

main = True

# box properties
leftCornerXBoxLoc = np.random.randint(100,500)
leftCornerYBoxLoc = np.random.randint(100,400)
boxWidth = 150
boxHeight = 150
boxColor = RED
targetStartTime = 2000
timeToReach = None
FETCHDATAFROMREALTIME = True


enforce = True
offline = False
positions = True
"""
Objects
"""

class Box():
    """
    Contains properties for a box
    """
    def __init__(self,leftCornerXBoxLoc,leftCornerYBoxLoc,boxWidth,boxHeight):
        self.leftCornerXBoxLoc = leftCornerXBoxLoc
        self.leftCornerYBoxLoc = leftCornerYBoxLoc
        self.boxWidth = boxWidth
        self.boxHeight = boxHeight

class Player(pygame.sprite.Sprite):
    """
    Spawn a player
    """

    def __init__(self,targetBox):
        pygame.sprite.Sprite.__init__(self)
        self.images = []
        try:
            img = pygame.image.load(os.path.join('Experiment_pointer/images', 'dot.png')).convert()
        except FileNotFoundError:
            img = pygame.image.load(os.path.join('images', 'dot.png')).convert()
        self.images.append(img)
        self.image = self.images[0]
        self.rect = self.image.get_rect()
        self.movex = 0 # move along X
        self.movey = 0 # move along Y
        self.targetBoxXmin = targetBox.leftCornerXBoxLoc
        self.targetBoxXmax = targetBox.leftCornerXBoxLoc + targetBox.boxWidth
        self.targetBoxYmin = targetBox.leftCornerYBoxLoc
        self.targetBoxYmax = targetBox.leftCornerYBoxLoc + targetBox.boxHeight
    
    def control(self,x,y):
        """
        control player movement
        """
        self.movex = x
        self.movey = -y
    
    def update(self):
        """
        Update sprite position
        """
        self.rect.x = self.rect.x + self.movex
        self.rect.y = self.rect.y + self.movey
    
    def updatepos(self,x,y):
        # note will need to enforce 0<x<960 and 0 < y < 720
        global enforce
        global offline
        global positions
        if enforce and offline:
            y = 720 * (y/2000)
            x = (x + 600)
        elif enforce and not offline and not positions:
            y = (y+1)/1 * 360 # 
            x = (x+0.25) * 1200
        elif enforce and not offline and positions:
            y = (y+1)/1 * 600 # 
            x = (x-1)/4 * 800   # 1600 - 2400
        self.rect.x = x
        self.rect.y = y
        global targetStartTime
        if self.targetBoxXmin <= self.rect.x <= self.targetBoxXmax and self.targetBoxYmin <= self.rect.y <= self.targetBoxYmax and pygame.time.get_ticks() > targetStartTime:
            global boxColor
            global GREEN
            global timeToReach
            boxColor = GREEN
            timeToReach = pygame.time.get_ticks()
            print(timeToReach)

    def reset(self,targetBox):
        self.targetBoxXmin = targetBox.leftCornerXBoxLoc
        self.targetBoxXmax = targetBox.leftCornerXBoxLoc + targetBox.boxWidth
        self.targetBoxYmin = targetBox.leftCornerYBoxLoc
        self.targetBoxYmax = targetBox.leftCornerYBoxLoc + targetBox.boxHeight
        global boxColor
        global RED
        boxColor = RED
    

            



"""
Setup

"""

processedRigidBodyParts = ['Pelvis', 'Ab', 'Chest', 'Neck', 'Head', 'LShoulder', 'LUArm', 
                      'LFArm', 'LHand', 'RShoulder', 'RUArm', 'RFArm', 'RHand']

rightHandIndex = processedRigidBodyParts.index('RHand')

if FETCHDATAFROMREALTIME:
    # create new shared memory
    BODY_PART_MEM = 'Body Parts'
    noDataTypes = 6 # 3 for pos and 3 for vector pos
    noBodyParts = 51
    dataEntries = noDataTypes * noBodyParts
    shared_block = shared_memory.SharedMemory(size= dataEntries * 8, name=BODY_PART_MEM, create=False)
    shared_array = np.ndarray(shape=(noBodyParts,noDataTypes), dtype=np.float64, buffer=shared_block.buf)



clock = pygame.time.Clock()
pygame.init()
world = pygame.display.set_mode([worldx,worldy]) # this is the surface

targetBox = Box(leftCornerXBoxLoc,leftCornerYBoxLoc,boxWidth,boxHeight)
player = Player(targetBox)   # spawn player
player.rect.x = 0   # go to x
player.rect.y = 0   # go to y
player_list = pygame.sprite.Group()
player_list.add(player)
steps = 20





"""
Main Loop
"""


while main:

    if FETCHDATAFROMREALTIME:
            #player.updatepos(shared_array[rightHandIndex][4],shared_array[rightHandIndex][5]) # get index 1 and 2 for pos
            player.updatepos(-shared_array[rightHandIndex][1],-shared_array[rightHandIndex][2])
            print('X:',player.rect.x)
            print('Y:',player.rect.y)
            print('dirX:',shared_array[rightHandIndex][0] ) 
            print('dirY:',shared_array[rightHandIndex][1] ) # this is -x
            print('dirZ:',shared_array[rightHandIndex][2] ) # this is -y
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            try:
                sys.exit()
            finally:
                main = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == ord('q'):
                pygame.quit()
                try:
                    sys.exit()
                finally:
                    main = False
        
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
                pygame.quit()
                sys.exit()
                main = False


    player.update()
    world.fill(BLUE)
    player_list.draw(world) # draw player

    if timeToReach is not None and pygame.time.get_ticks() > timeToReach + 2000:
        # reset
        timeToReach = None
        leftCornerXBoxLoc = np.random.randint(100,500)
        leftCornerYBoxLoc = np.random.randint(100,400)
        targetBox = Box(leftCornerXBoxLoc,leftCornerYBoxLoc,boxWidth,boxHeight)
        player.reset(targetBox)
    if pygame.time.get_ticks() > targetStartTime:
        pygame.draw.rect(world, boxColor, pygame.Rect(leftCornerXBoxLoc, leftCornerYBoxLoc, boxWidth, boxHeight)) # draw red box : left, top, width, height (600 to 750, 240, 390)
    pygame.display.flip()
    clock.tick(fps)
    #print(pygame.time.get_ticks()) # get time in ms
    pygame.display.flip()
    clock.tick(fps)
