import os
import pygame as pg
from gameObjects import Pendulum, Cart
from setUp import GUI
from dynamics import Dynamics



def main():
    """this function is called when the program starts.
    it initializes everything it needs, then runs in
    a loop until the function returns."""
    # # Initialize Everything
    interface = GUI()
    state_object = Dynamics()
    interface.set_state(state_object)
    interface.clean_background()
    

    # Prepare Game Objects
    interface.draw()
    clock = pg.time.Clock()

    # Main Loop
    going = True
    start = False
    while going:
        clock.tick(10)

        # Handle Input Events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                going = False
            elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                going = False

            if event.type == pg.MOUSEBUTTONDOWN:
                start = True

            if event.type == pg.MOUSEMOTION and start==True:
                print(event.rel)
                interface.clean_background()
                interface.update_scene(event.rel[0])
                

            if event.type == pg.MOUSEBUTTONUP:
                start = False
        
        
        

        # Draw Everything
        interface.draw()
        pg.display.flip()

        # if start == True:
        #     state.update()
        #     allsprites.update()

    pg.quit()


# Game Over


# this calls the 'main' function when this script is executed
if __name__ == "__main__":
    main()