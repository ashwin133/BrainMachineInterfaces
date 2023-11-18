"""
Setup

"""
from objects import *
from variables import *
from multiprocessing import shared_memory



# if FETCHDATAFROMREALTIME:
#     # create new shared memory
#     BODY_PART_MEM = 'Body Parts'
#     noDataTypes = 6 # 3 for pos and 3 for vector pos
#     noBodyParts = 51
#     dataEntries = noDataTypes * noBodyParts
#     shared_block = shared_memory.SharedMemory(size= dataEntries * 8, name=BODY_PART_MEM, create=False)
#     shared_array = np.ndarray(shape=(noBodyParts,noDataTypes), dtype=np.float64, buffer=shared_block.buf)



clock = pygame.time.Clock()
pygame.init()
world = pygame.display.set_mode([worldx,worldy]) # this is the surface

cursor = Cursor()   # spawn cursor
cursor.rect.x = worldx // 2   # go to x
cursor.rect.y = worldy // 2   # go to y

player_list = pygame.sprite.Group()
player_list.add(cursor)

steps = 20 # speed at which the cursor moves