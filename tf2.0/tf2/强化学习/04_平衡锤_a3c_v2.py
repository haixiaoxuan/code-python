#!/usr/bin/env Python
# coding=utf-8

import ray
import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time


GAMMA = 0.95
LEARNING_RATE = 0.0001

ENV_NAME = 'CartPole-v0'

ENTROPY_BETA = 0.001       # 策略的熵项权重系数
UPDATE_GLOBAL_ITER = 256

RENDER = False


ray.init(address='ray://localhost:10001')
# ray.init(local_mode=True)


@ray.remote
class ActorCritic():
    import tensorflow as tf
    from tensorflow import keras
    def __init__(self, global_ac=None):

        self.env = gym.make(ENV_NAME)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.actor = self.create_actor_network()
        self.critic = self.create_critic_network()

        self.optimizer_actor = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.optimizer_critic = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        if global_ac is not None:
            self.global_ac = global_ac

    def create_actor_network(self):
        status = keras.layers.Input(shape=self.state_dim)
        h1 = keras.layers.Dense(units=24, activation=keras.activations.relu)(status)
        h2 = keras.layers.Dense(units=self.action_dim, activation=keras.activations.relu)(h1)
        action_prob = keras.layers.Softmax()(h2)
        model = keras.models.Model(inputs=status, outputs=action_prob)
        model.summary()
        return model

    def create_critic_network(self):
        status = keras.layers.Input(shape=self.state_dim)
        h1 = keras.layers.Dense(units=24, activation=keras.activations.relu)(status)
        q_value = keras.layers.Dense(units=1, activation=None)(h1)
        model = keras.models.Model(inputs=status, outputs=q_value)
        model.summary()
        return model

    def choose_action(self, observation):
        prob_weights = self.actor(observation[np.newaxis, :])
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.numpy().ravel())
        return action

    def start(self):
        total_step = 0
        # state, action, reward
        buffer_s, buffer_a, buffer_r = [], [], []
        s = self.env.reset()
        episode_step = 0

        while True:
            a = self.choose_action(s)
            s_, r, done, info = self.env.step(a)
            total_step += 1
            episode_step += 1

            if done:
                r = -5

            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append(r)

            if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                if done:
                    # print("total_steps:", episode_step)
                    v_s_ = 0        # next state reward
                    self.env.reset()
                    episode_step = 0
                else:
                    v_s_ = self.critic(s_[np.newaxis, :])[0, 0]

                buffer_v_target = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    buffer_v_target.append(v_s_)
                buffer_v_target.reverse()

                buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)

                critic_grads, td_error = self.get_critic_grad(buffer_s, buffer_v_target)
                actor_grads = self.get_actor_grad(buffer_s, buffer_a, td_error)
                self.global_ac.set_grads.remote(actor_grads, critic_grads)
                actor_params, critic_params = self.global_ac.get_params.remote()
                self.set_params(ray.get(actor_params), ray.get(critic_params))

                buffer_s, buffer_a, buffer_r = [], [], []

    def get_actor_grad(self, state, action, td_error):
        td_error = tf.constant(td_error, dtype=tf.float32)
        with tf.GradientTape() as tape:
            action_prob = self.actor(state)
            neg_log_prob = keras.losses.sparse_categorical_crossentropy(np.array(action), action_prob)

            polity_entropy = -tf.reduce_sum(action_prob * tf.math.log(action_prob + 1e-5))
            loss = -tf.reduce_mean(neg_log_prob * td_error + ENTROPY_BETA * polity_entropy)

        gradients = tape.gradient(loss, self.actor.variables)
        return gradients

    def get_critic_grad(self, state, reward):
        with tf.GradientTape() as tape:
            td_error = reward - self.critic(state)
            loss = tf.square(td_error)

        gradients = tape.gradient(loss, self.critic.variables)
        return gradients, td_error

    def set_grads(self, actor_gradients, critic_gradients):
        self.optimizer_actor.apply_gradients(grads_and_vars=zip(actor_gradients, self.actor.variables))
        self.optimizer_critic.apply_gradients(grads_and_vars=zip(critic_gradients, self.critic.variables))

    @ray.method(num_returns=2)
    def get_params(self):
        return self.actor.get_weights(), self.critic.get_weights()

    def set_params(self, actor_params, critic_params):
        self.actor.set_weights(actor_params)
        self.critic.set_weights(critic_params)

    def test(self):
        while True:
            step = 0
            obs = self.env.reset()
            actor_params, critic_params = self.global_ac.get_params.remote()
            self.set_params(ray.get(actor_params), ray.get(critic_params))

            while True:
                if RENDER:
                    self.env.render()
                output = self.actor(obs[np.newaxis, :])[0]
                action = tf.argmax(output).numpy()
                next_state, reward, done, info = self.env.step(action)
                step += 1
                if done:
                    print("total_step", step)
                    time.sleep(10)
                    break


if __name__ == '__main__':
    global_ac = ActorCritic.remote()
    # global_ac.test.remote()

    test = ActorCritic.remote(global_ac)
    test.test.remote()

    workers = [ActorCritic.remote(global_ac) for _ in range(10)]
    ids = [worker.start.remote() for worker in workers]
    [ray.get(i) for i in ids]







