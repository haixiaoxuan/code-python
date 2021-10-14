import tensorflow as tf
import numpy as np
import gym
import random
from collections import deque


"""
    https://tf.wiki/zh/basic/models.html
    DQN 平衡锤
"""

env = gym.make('CartPole-v1')
state = env.reset()
print(env.action_space.sample())
print(state)


# while True:
#     env.render()
#     action = model.predict(state)
#     next_state, reward, done, info = env.step(action)
#     if done:
#         break


num_episodes = 50000            # 游戏训练的总episode数量
num_exploration_episodes = 100  # 探索过程所占的episode数量
max_len_episode = 1000000       # 每个episode的最大step

batch_size = 256
learning_rate = 1e-3
gamma = 1.

initial_epsilon = 1.            # 探索起始时的探索率
final_epsilon = 0.01            # 探索终止时的探索率

pool_size = 10000               # 经验回放池大小


class QNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=2)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def predict(self, inputs):
        q_values = self(inputs)
        return tf.argmax(q_values, axis=-1)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    model = QNetwork()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    replay_buffer = deque(maxlen=pool_size)

    epsilon = initial_epsilon

    for episode_id in range(num_episodes):
        state = env.reset()
        epsilon = max(                  # 计算当前探索率, epsilon 随着场数的增加逐渐减小
            initial_epsilon * (num_exploration_episodes - episode_id) / num_exploration_episodes,
            final_epsilon)

        for t in range(max_len_episode):
            env.render()

            # select action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = model.predict(np.expand_dims(state, axis=0)).numpy()
                action = action[0]

            next_state, reward, done, info = env.step(action)

            # 如果游戏Game Over，给予大的负奖励
            reward = -10. if done else reward
            replay_buffer.append((state, action, reward, next_state, 1 if done else 0))

            state = next_state

            if done:
                print("game_over, episode %d, epsilon %f, score %d" % (episode_id, epsilon, t))
                break

            if len(replay_buffer) >= batch_size:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(
                    *random.sample(replay_buffer, batch_size))

                batch_state, batch_reward, batch_next_state, batch_done = \
                    [np.array(a, dtype=np.float32) for a in [batch_state, batch_reward, batch_next_state, batch_done]]
                batch_action = np.array(batch_action, dtype=np.int32)

                q_value = model(batch_next_state)

                # 游戏结束不需要结算后面的reward
                y = batch_reward + (gamma * tf.reduce_max(q_value, axis=1)) * (1 - batch_done)
                with tf.GradientTape() as tape:
                    loss = tf.keras.losses.mean_squared_error(
                        y_true=y,
                        y_pred=tf.reduce_sum(model(batch_state) * tf.one_hot(batch_action, depth=2), axis=1)
                    )
                grads = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))


