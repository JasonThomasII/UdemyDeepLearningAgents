#Importing Dependencies for Mario
import tensorflow as tf
import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import gym_super_mario_bros
#This makes mario Travel in the "Right" Direction only. 
#Other options are "Simple Movement" Which allows mario to move in any directions
#starting with RIGHT_ONLY will make training easier, but our Model won't be as good.
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from IPython.display import clear_output
from keras.models import save_model
from keras.models import load_model
import time

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, RIGHT_ONLY)

total_reward = 0
done = True
for step in range(100000):
    env.render()
    if done: 
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    print(info)
    total_reward += reward
    clear_output(wait=True)

class DQNAgent:
    def __init__(self,state_size, action_size):
        #CreateVars
        self.state_space = state_size
        state.action_space = action_size
        self.memory = deque(maxlen=5000)

        #Exploration vs explotation
        self.epsilon = 1 
        self.max_epsilon = 1
        self.min_epslion = 0.01
        self.decay_epslion = 0.0001

        #Build Neural Networks for Agent
        self.main_network = self.build_network()
        self.target_network = self.build_network()
        self.update_target_network()

    def build_network(self):
        model = Sequential()
        model.add(Conv2D(64, (4,4), strides=4, padding='same', input_shape = self.state_space))
        model.add(Activation('relu'))

        model.add(Conv2D(64,(4,4),strides=2, padding='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(64,(3,3),strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))

        model.compile(loss ='mse', optimizers=Adam())
        
    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def act(self, state):
        if random.uniform(0,1)< self.epsilon:
            return np.random.randint(self.action_space)

env.close()