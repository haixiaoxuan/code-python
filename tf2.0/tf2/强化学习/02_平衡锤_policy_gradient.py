import gym
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


GAMMA = 0.95
LEARNING_RATE = 0.01

ENV_NAME = 'CartPole-v0'
EPISODE = 3000      # Episode limitation
STEP = 3000         # Step limitation in an episode
TEST = 100          # The number of experiment test every 100 episode


class PolicyGradient():
    def __init__(self, env):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # state action reword
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.model = self.create_network()
        self.optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    def create_network(self):
        status = keras.layers.Input(shape=self.state_dim)
        h1 = keras.layers.Dense(units=24, activation=tf.nn.relu)(status)
        h2 = keras.layers.Dense(units=self.action_dim, activation=tf.nn.relu)(h1)
        action_prob = keras.layers.Softmax()(h2)
        model = keras.models.Model(inputs=status, outputs=action_prob)
        model.summary()
        return model

    def choose_action(self, observation):
        prob_weights = self.model(observation[np.newaxis, :])
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.numpy().ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # 积累够一个 episode 开始训练
        discounted_ep_rs = np.zeros_like(self.ep_rs)  # 反向折扣奖励
        running_add = 0
        # 从最后一步奖励向前更新
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * GAMMA + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # 对折扣奖励进行归一化
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)

        discounted_reward = tf.Variable(initial_value=discounted_ep_rs, dtype=tf.float32)
        var = [discounted_reward, *self.model.variables]

        with tf.GradientTape() as tape:
            action_prob = self.model(np.vstack(self.ep_obs))
            neg_log_prob = keras.losses.sparse_categorical_crossentropy(np.array(self.ep_as), action_prob)
            loss = tf.reduce_mean(neg_log_prob * discounted_reward)

        gradients = tape.gradient(loss, var)
        self.optimizer.apply_gradients(grads_and_vars=zip(gradients, var))

        # 清空记录
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data


def main():
    # 初始化环境
    env = gym.make(ENV_NAME)
    agent = PolicyGradient(env)

    for episode in range(EPISODE):
        state = env.reset()

        for step in range(STEP):
            action = agent.choose_action(state)  # e-greedy action for train

            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward)
            state = next_state
            if done:
                print("game over ", step, " steps")
                agent.learn()
                break

        # Test every 100 episodes
        if episode % TEST == 0:
            total_reward = 0
            state = env.reset()
            for _ in range(STEP):
                env.render()
                action = agent.choose_action(state)  # direct action for test
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
            ave_reward = total_reward
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)


if __name__ == '__main__':
    main()
