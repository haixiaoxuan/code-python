from nes_py.wrappers import JoypadSpace
import gym
from Contra.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import time


"""
    pip install gym-contra
    https://github.com/OuYanghaoyue/gym_contra
"""


env = gym.make('Contra-v0')
env = JoypadSpace(env, RIGHT_ONLY)

print("actions", env.action_space)
print("observation_space ", env.observation_space.shape[0])

done = False
obs = env.reset()
print(obs)

for step in range(5000):
    if done:
        print("Over")
        break
    time.sleep(0.02)

    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    print(reward, info, action)
    env.render()

env.close()