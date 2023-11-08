import config
import numpy as np
from control import StateSpace
import pygame as pg
from gameObjects import Pendulum, Cart
import math

class Dynamics:

    def __init__(self):
        m = config.PENDULUM['mass']
        M = config.CART['mass']
        g = config.GRAVITY
        I = config.PENDULUM['MOI']
        l = config.PENDULUM['length']
        p = I*(M+m)+M*m*l^2 
        self.A = np.array([[0,1,0,0],
                  [0,0,((m**2)*g*(l**2))/p,0],
                  [0,0,0,1],
                  [0,0,m*g*l*(M+m)/p,0]]) #4x4 matrix
        self.B = np.array([[0], [(I+m*(l**2))/p], [0], [m*l/p]])
        self.C = [[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]]
        self.D = [[0],[0], [0], [0]]
        self.first_state = True
        self.pendulum = Pendulum()
        self.cart = Cart()
        pass
    
    def update(self):
        # get current state
        current_cart = self.get_cart()
        current_pendulum = self.get_pendulum()
        print(current_cart.pos, current_cart.vel, current_pendulum.angle, current_pendulum.ang_vel)
        current_state = np.array([[current_cart.pos], [current_cart.vel], [current_pendulum.angle], [current_pendulum.ang_vel]])
        
        # calculate next state
        next_state = self.get_new_state(current_state)

        #update current state
        self.pendulum.ang_vel = math.radians(float(next_state[3]))
        self.pendulum.angle = math.radians(float(next_state[2]))
        pass

    def get_new_state(self, current_state): # 4x1 matrux
        
        time = pg.time.get_ticks()
        state_space = StateSpace(self.A, self.B, self.C, self.D)

        # delta = np.matmul(self.A,current_state) + self.B
        # new_state = current_state + delta
        # get impulse response of system
        if not self.first_state:
            new_state = state_space.output(time, x=current_state, u=0)

        else:
            new_state = state_space.output(time+100,x=current_state, u=1) 

        return new_state
    
    def set_cart_vel(self, vel):
        self.cart.vel = vel
    
    def get_pendulum(self):
        return self.pendulum

    def get_cart(self):
        return self.cart

    def get_current_state(self):
        current_state = np.transpose(np.array([self.cart.pos, self.cart.vel, self.pendulum.angle, self.pendulum.ang_vel]))
        return current_state
