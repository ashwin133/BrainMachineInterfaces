import pygame
import sys
import math
import config

class Pendulum():

    def __init__(self):

       # Initial state
        self.pendulum_angle = 0
        self.pendulum_angular_velocity = 0
        self.cart = None
        self.previous_time = 0
    
    def set_cart(self, cart):
        self.cart = cart

    def draw_pendulum(self, screen):
        # Calculate pendulum position
        pendulum_center_x = self.cart.cart_x + config.CART_WIDTH // 2
        pendulum_center_y = config.HEIGHT - config.CART_HEIGHT
        pendulum_end_x = pendulum_center_x + config.PEN_LENGTH * math.sin(self.pendulum_angle)
        pendulum_end_y = pendulum_center_y - config.PEN_LENGTH * math.cos(self.pendulum_angle)

        # Draw pendulum
        pygame.draw.line(screen, config.PENDULUM_COLOR, (pendulum_center_x, pendulum_center_y), (pendulum_end_x, pendulum_end_y), config.PENDULUM_WIDTH)



    def update_pendulum_state(self):
    
        current_time = pygame.time.get_ticks() / 1000.0
        elapsed_time = current_time - self.previous_time

        # Apply control here (adjust cart_velocity based on pendulum_angle)

        # Update pendulum state
        pendulum_acceleration = (
            config.G * math.sin(self.pendulum_angle) - 
            math.cos(self.pendulum_angle) * (self.cart.cart_velocity ** 2) / config.PEN_LENGTH
        )
        self.pendulum_angular_velocity += pendulum_acceleration * elapsed_time
        self.pendulum_angle += self.pendulum_angular_velocity * elapsed_time



        # Update cart position

        self.cart.update_cart()

        self.previous_time = current_time


class Cart():

    def __init__(self):

       # Initial state
        self.cart_x = (config.WIDTH - config.CART_WIDTH) // 2
        self.cart_velocity = 0
        self.previous_time = 0

    def draw_cart(self, screen):
        
        # Draw cart
        pygame.draw.rect(screen, config.CART_COLOR, (self.cart_x, config.HEIGHT - config.CART_HEIGHT, config.CART_WIDTH, config.CART_HEIGHT))

    def update_cart(self):

        current_time = pygame.time.get_ticks() / 1000.0
        elapsed_time = current_time - self.previous_time
        sensitivity = 150
        # Update cart position
        
        self.cart_x += sensitivity*self.cart_velocity * elapsed_time

        self.previous_time = current_time

