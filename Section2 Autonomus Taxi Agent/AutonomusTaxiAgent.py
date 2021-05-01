import gym
import numpy
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3')

episodes = 10
for episode in range(1, episodes):
    state = env.reset()
    done = False
    score = 0

    while not done: 
        env.render()
        state, reward, done, info = env.step(env.action_space.sample())
        score += reward
    print('Episode: {}\nScore: {}'.format(episode, score))
env.close()
