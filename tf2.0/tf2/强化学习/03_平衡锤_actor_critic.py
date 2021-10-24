import gym
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os


GAMMA = 0.95
LEARNING_RATE = 0.01

EPSILON = 0.01              # final value of epsilon
REPLAY_SIZE = 10000         # experience replay buffer size


ENV_NAME = 'CartPole-v0'
EPISODE = 3000  # Episode limitation
STEP = 3000     # Step limitation in an episode

TEST = 100      # The number of experiment test every 100 episode

render = False


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)


class Actor():
    def __init__(self, env):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

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

    def learn(self, state, action, td_error):
        td_error = tf.Variable(initial_value=td_error, dtype=tf.float32)
        var = [td_error, *self.model.variables]

        with tf.GradientTape() as tape:
            action_prob = self.model(state[np.newaxis, :])
            neg_log_prob = keras.losses.sparse_categorical_crossentropy(np.array(action), action_prob)
            loss = -tf.reduce_mean(neg_log_prob * td_error)

        gradients = tape.gradient(loss, var)
        self.optimizer.apply_gradients(grads_and_vars=zip(gradients, var))


class Critic():
    def __init__(self, env):
        self.epsilon = EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.model = self.create_q_network()
        self.optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    def create_q_network(self):
        status = keras.layers.Input(shape=self.state_dim)
        h1 = keras.layers.Dense(units=24, activation=tf.nn.relu)(status)
        q_value = keras.layers.Dense(units=1, activation=None)(h1)
        model = keras.models.Model(inputs=status, outputs=q_value)
        model.summary()
        return model

    def train_q_network(self, state, reward, next_state):
        s, s_ = state[np.newaxis, :], next_state[np.newaxis, :]

        with tf.GradientTape() as tape:
            td_error = reward + GAMMA * tf.squeeze(self.model(s_)) - tf.squeeze(self.model(s))
            loss = tf.square(td_error)

        gradients = tape.gradient(loss, self.model.variables)
        self.optimizer.apply_gradients(grads_and_vars=zip(gradients, self.model.variables))
        return td_error


def main():
    env = gym.make(ENV_NAME)
    actor = Actor(env)
    critic = Critic(env)

    for episode in range(EPISODE):
        state = env.reset()

        for step in range(STEP):
            action = actor.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            td_error = critic.train_q_network(state, reward, next_state)
            actor.learn(state, action, td_error)
            state = next_state
            if done:
                break

        # Test every 100 episodes
        if episode % TEST == 0:
            total_reward = 0
            state = env.reset()
            for _ in range(STEP):
                if render:
                    env.render()
                action = actor.choose_action(state)  # direct action for test
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
            ave_reward = total_reward
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)


if __name__ == '__main__':
    main()
