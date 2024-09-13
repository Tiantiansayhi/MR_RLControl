import gymnasium as gym

env = gym.make('CartPole-v1')
observation, info = env.reset()
