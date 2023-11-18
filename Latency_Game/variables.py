"""
Stores all variables needed for pointer experiment
"""

# import necessary libraries
import numpy as np

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
colours = {'BLUE': BLUE , 'BLACK': BLACK, 'WHITE':WHITE, 'RED':RED, 'GREEN':GREEN

}

main = True

timeCursorAppears = np.random.randint(2000,5000)
reactionTimes = []
cursorAppearLatch = False
latchResetTime = None
FETCHDATAFROMREALTIME = False




