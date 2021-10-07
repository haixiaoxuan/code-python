import tensorflow as tf
import numpy as np
import gym
import random
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
import os
import time


optimiser_learning_rate = 0.00025
observe_step_num = 100000
batch_size = 32

initial_epsilon = 1.0
epsilon_anneal_num = 500000     # 要跑多少step可以将 epsilon 从1降到0.01
final_epsilon = 0.01

gamma = 0.99
replay_memory = 200000
n_episodes = 100000
no_op_steps = 2

# target网络落后 Q网络的step
update_target_model_steps = 100000
train_dir = ""
render = False


input_shape = (84, 84, 4)       # 取最近4帧图片
action_size = 4


if __name__ == "__main__":
    env = gym.make("PongDeterministic-v4")

    memory = deque(maxlen=replay_memory)

    # 计数
    episode_num = 0

    epsilon = initial_epsilon
    epsilon_decay = (initial_epsilon - final_epsilon) / epsilon_anneal_num


