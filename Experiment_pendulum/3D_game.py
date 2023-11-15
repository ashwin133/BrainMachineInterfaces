import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math
from gameObjects import Pendulum, Cart

def update_view(camera_position, camera_rotation):
    glLoadIdentity()
    glTranslatef(*camera_position)
    glRotatef(camera_rotation[0], 1, 0, 0)
    glRotatef(camera_rotation[1], 0, 1, 0)

def main():
    
    # Initialize Pygame
    pygame.init()
    display = (800, 600)

    # Initial camera position and orientation
    camera_position = [0, 0, 0]
    camera_rotation = [0, 0, 0]
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    # Set up the perspective
    glMatrixMode(GL_PROJECTION)
    gluPerspective(120, (4/3), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glTranslatef(0.0, 0.0, -10)
    clock = pygame.time.Clock()

    # Simulation parameters
    dt = 0.02
    previous_time = pygame.time.get_ticks() / 1000.0

    # Initialise game objects
    cart = Cart()
    pendulum = Pendulum()
    pendulum.previous_time = previous_time
    cart.previous_time = previous_time
    pendulum.set_cart(cart)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        keys = pygame.key.get_pressed()

        if keys[K_1]:
            camera_rotation[1] += 0.1
            update_view(camera_position, camera_rotation)
        elif keys[K_2]:
            camera_rotation[1] -= 0.1
            update_view(camera_position, camera_rotation)
        elif keys[K_3]:
            camera_rotation[0] += 0.1
            update_view(camera_position, camera_rotation)
        elif keys[K_4]:
            camera_rotation[0] -= 0.1
            update_view(camera_position, camera_rotation)
        elif keys[K_5]:
            camera_position[2] += 0.05
            update_view(camera_position, camera_rotation)
        elif keys[K_6]:
            camera_position[2] -= 0.05
            update_view(camera_position, camera_rotation)

        elif keys[K_LEFT]:
            cart.cart_velocity = -0.0002
            cart.update_cart()
                        
        elif keys[K_RIGHT]:
            cart.cart_velocity = 0.0002
            cart.update_cart()
            

        elif keys[K_SPACE]:
            cart.cart_velocity = 0
            cart.update_cart()
            
            
            # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            # cart.draw_cart_3d()
            # pendulum.draw_pendulum_3d()
            # glutSwapBuffers()

        else:
            cart.cart_velocity = 0
            cart.update_cart()
        
        pendulum.update_pendulum_state()


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        cart.draw_cart_3d()
        pendulum.draw_pendulum_3d()
        glutSwapBuffers()
        pygame.display.flip()
    

if __name__ == "__main__":
    main()
