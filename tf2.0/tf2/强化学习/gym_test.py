import gym

"""
    pip install gym==0.17.3
"""

# 获取所有注册游戏
for g in gym.envs.registry.all():
    print(g)


env = gym.make("CartPole-v1")
env.render()

print(env.action_space)
print([env.action_space.sample() for i in range(10)])

print(env.observation_space.shape)

