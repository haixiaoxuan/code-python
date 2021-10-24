import numpy as np
import pickle
import gym
import time


resume = True
render = False

hidden_unit = 200
batch_size = 256
learning_rate = 1e-4
gamma = 0.99
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2

input_dim = 80 * 80  # input dimensionality: 80x80 grid

if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(hidden_unit, input_dim) / np.sqrt(input_dim)
    model['W2'] = np.random.randn(hidden_unit) / np.sqrt(hidden_unit)

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def preprocess(obs):
    """
        210x160x3 uint8 frame into 6400 (80x80) input_dim float vector
    """
    obs = obs[35:195]  # crop
    obs = obs[::2, ::2, 0]  # downsample by factor of 2
    obs[obs == 144] = 0  # erase background (background type 1)
    obs[obs == 109] = 0  # erase background (background type 2)
    obs[obs != 0] = 1  # everything else (paddles, ball) just set to 1
    return obs.astype(np.float).ravel()


def discount_rewards(r):
    """ compute discounted reward, r = array([r1, r2, r3, ...]) """
    discounted_r = np.zeros_like(r, dtype=np.float)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  # return probability of taking action 2, and hidden state


def policy_backward(eph, epdlogp, epx):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


env = gym.make("Pong-v0")
observation = env.reset()


def train():
    prev_x = None  # 前一状态

    # status hidden_output (y-y^) reward
    xs, hs, dlogps, drs = [], [], [], []
    reward_sum = 0
    episode_number = 0

    while True:
        if render:
            env.render()

        cur_x = preprocess(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(input_dim)
        prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        aprob, h = policy_forward(x)
        action = 2 if np.random.uniform() < aprob else 3

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        xs.append(x)  # state
        hs.append(h)  # hidden_output
        y = 1 if action == 2 else 0
        dlogps.append(y - aprob)
        drs.append(reward)

        if done:
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)

            xs, hs, dlogps, drs = [], [], [], []  # reset array memory

            # discount reward
            discounted_epr = discount_rewards(epr)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            # TODO
            # modulate the gradient with advantage (PG magic happens right here.)
            epdlogp *= discounted_epr

            grad = policy_backward(eph, epdlogp, epx)
            for k in model:
                grad_buffer[k] += grad[k]  # accumulate grad over batch

            # perform rmsprop parameter update every batch_size episodes
            if episode_number % batch_size == 0:
                for k, v in model.items():
                    g = grad_buffer[k]  # gradient

                    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)

                    # reset batch gradient buffer
                    grad_buffer[k] = np.zeros_like(v)

            print('resetting env. episode %d, reward total was %d.' % (episode_number, reward_sum))

            if episode_number % 100 == 0:
                pickle.dump(model, open('save.p', 'wb'))
            reward_sum = 0
            observation = env.reset()
            prev_x = None


def test():
    env = gym.make("Pong-v0")
    observation = env.reset()
    prev_x = None
    reward_sum = 0
    while True:
        time.sleep(0.005)
        env.render()
        cur_x = preprocess(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(input_dim)
        prev_x = cur_x

        aprob, h = policy_forward(x)
        action = 2 if np.random.uniform() < aprob else 3

        observation, reward, done, info = env.step(action)
        reward_sum += reward
        if done:
            print("total_reward", reward_sum)
            break


if __name__ == "__main__":
    test()


