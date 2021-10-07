import gym


env = gym.make('SpaceInvaders-v0')
state = env.reset()

for i in range(1000):
    env.render()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    if done:
        print("game over")
        break


import time
time.sleep(100)

