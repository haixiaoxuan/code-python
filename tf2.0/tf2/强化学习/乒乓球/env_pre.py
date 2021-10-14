import gym
import matplotlib.pyplot as plt


env = gym.make("PongDeterministic-v4")
obs = observation = env.reset()
print(obs)
print(env.action_space.n)
print(env.action_space.shape)

plt.figure("Image") 
plt.imshow(obs)
plt.show()


obs = obs[35:195]  # crop
plt.imshow(obs)
plt.show()


obs = obs[::2, ::2, 0]  # downsample by factor of 2
plt.imshow(obs)
plt.show()


obs[obs == 144] = 0  # erase background (background type 1)
plt.imshow(obs)
plt.show()

obs[obs == 109] = 0  # erase background (background type 2)
plt.imshow(obs)
plt.show()


obs[obs != 0] = 1    # everything else (paddles, ball) just set to 1
plt.imshow(obs)
plt.show()


