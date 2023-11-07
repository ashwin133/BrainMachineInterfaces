import pygame
import sys
import math
import config
from gameObjects import Pendulum, Cart


def main():

    # Initialize Pygame
    pygame.init()

    # Create the display
    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
    pygame.display.set_caption("Balancing Inverted Pendulum")

    # Simulation parameters
    dt = 0.02
    previous_time = pygame.time.get_ticks() / 1000.0

    # Initialise game objects
    cart = Cart()
    pendulum = Pendulum()
    pendulum.previous_time = previous_time
    cart.previous_time = previous_time
    pendulum.set_cart(cart)
    

    

    # Main game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            cart.cart_velocity = -0.5
            pendulum.update_pendulum_state()
        elif keys[pygame.K_RIGHT]:
            cart.cart_velocity = 0.5
            pendulum.update_pendulum_state()
        else:
            cart.cart_velocity = 0
            pendulum.update_pendulum_state()

    

        # Clear the screen
        screen.fill(config.BACKGROUND_COLOR)

        # Draw cart
        cart.draw_cart(screen)

        # Draw pendulum
        pendulum.draw_pendulum(screen)

        pygame.display.flip()

    # Quit Pygame
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

