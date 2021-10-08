import gym
import tensorflow as tf
import numpy as np


GAMMA = 0.95            # 折扣系数
LEARNING_RATE = 0.01    # 学习率

ENV_NAME = 'CartPole-v0'    # 游戏名称
EPISODE = 3000              # Episode limitation
STEP = 3000                 # Step limitation in an episode
TEST = 100                  # The number of experiment test every 100 episode


class PolicyGradient():
    def __init__(self, env):
        # state 维度
        self.state_dim = env.observation_space.shape[0]
        # action 维度
        self.action_dim = env.action_space.n

        # state action reword
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self.create_softmax_network()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_softmax_network(self):
        # network weights
        W1 = self.weight_variable([self.state_dim, 20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20, self.action_dim])
        b2 = self.bias_variable([self.action_dim])

        # input layer
        self.state_input = tf.placeholder("float", [None, self.state_dim])          # state
        self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")       # action
        self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")     # discount reword

        h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        self.softmax_input = tf.matmul(h_layer, W2) + b2

        # 输出所有行为的概率
        self.all_act_prob = tf.nn.softmax(self.softmax_input, name='act_prob')

        # TODO
        # 损失函数: 交叉熵损失函数和状态价值函数的乘机
        self.neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.softmax_input, labels=self.tf_acts)
        self.loss = tf.reduce_mean(self.neg_log_prob * self.tf_vt)  # reward guided loss

        # 定义优化器
        self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

    def weight_variable(self, shape):
        # 初始化权重w
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        # 初始化 bias
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def choose_action(self, observation):
        # 根据输出概率选择最佳行为
        prob_weights = self.session.run(self.all_act_prob, feed_dict={self.state_input: observation[np.newaxis, :]})
        # 根据概率分布随机选择行为返回
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def store_transition(self, s, a, r):
        # state action reword
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # 积累够一个 episode 开始训练
        discounted_ep_rs = np.zeros_like(self.ep_rs)        # 反向折扣奖励
        running_add = 0
        # 从最后一步奖励向前更新
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * GAMMA + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # 对折扣奖励进行归一化
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)

        # train on episode
        self.session.run(self.train_op, feed_dict={
            self.state_input: np.vstack(self.ep_obs),
            self.tf_acts: np.array(self.ep_as),
            self.tf_vt: discounted_ep_rs,
        })

        # 清空记录
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data


def main():
    # 初始化环境
    env = gym.make(ENV_NAME)
    agent = PolicyGradient(env)

    for episode in range(EPISODE):
        state = env.reset()

        # Train (循环直到一局游戏结束)
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
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = agent.choose_action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)


if __name__ == '__main__':
    main()