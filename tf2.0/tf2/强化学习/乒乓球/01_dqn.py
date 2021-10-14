import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import gym
import random
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
import os
import time
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


params = {
    "optimiser_learning_rate": 0.00025,
    "observe_step_num": 100000,
    "batch_size": 256,
    
    "initial_epsilon": 1,
    "epsilon_anneal_num": 500000,
    "final_epsilon": 0.01,
    
    "gamma": 0.99,
    "replay_memory": 400000,     # 40w=20GB
    "n_episodes": 100000,       # 训练总局数
    "no_op_steps": 2,
    "render": True,
    "update_target_model_steps": 100000,        # 每隔多少步，target network 同步一次网络权重
    "train_dir": "log"
}


Input_shape = (84, 84, 4)
Action_size = 3


def pre_processing(observation):
    """ 预处理 """
    # 以整数存储节省空间，取消归一化
    processed_observation = rgb2gray(observation) * 255
    processed_observation = resize(processed_observation, (84, 84), mode="constant")
    processed_observation = np.uint8(processed_observation)
    return processed_observation


def huber_loss(y, q_value):
    error = tf.abs(y - q_value)
    quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)
    return loss


def atari_model():
    frames_input = keras.Input(shape=Input_shape, name="frames")
    actions_input = keras.layers.Input(shape=(Action_size,), name="action_mask")        # (1,1,1), (1,0,0)

    normalised = keras.layers.Lambda(lambda x: x / 255.0, name="normalised")(frames_input)
    conv1 = keras.layers.Conv2D(16, 8, 4, activation="relu")(normalised)
    conv2 = keras.layers.Conv2D(32, 4, 2, activation="relu")(conv1)
    conv2_flatten = keras.layers.Flatten()(conv2)
    dense1 = keras.layers.Dense(256, activation="relu")(conv2_flatten)
    output = keras.layers.Dense(Action_size)(dense1)
    # Then we multiply the output by the action mask
    # When trying to find the value of all the actions this will be a mask full of 1s
    # When trying to find the value of a specific action, the mask will only be 1 for a single action
    filtered_output = keras.layers.Multiply(name="Qvalue")([output, actions_input])

    model = keras.Model(inputs=[frames_input, actions_input], outputs=filtered_output)
    model.summary()
    optimiser = keras.optimizers.Adam()
    model.compile(optimizer=optimiser, loss=huber_loss)
    return model


def atari_model_target():
    frames_input = keras.Input(shape=Input_shape, name="frames")
    actions_input = keras.layers.Input(shape=(Action_size,), name="action_mask")

    normalised = keras.layers.Lambda(lambda x: x / 255.0, name="normalised")(frames_input)
    conv1 = keras.layers.Conv2D(16, 8, 4, activation="relu")(normalised)
    conv2 = keras.layers.Conv2D(32, 4, 2, activation="relu")(conv1)
    conv2_flatten = keras.layers.Flatten()(conv2)
    dense1 = keras.layers.Dense(256, activation="relu")(conv2_flatten)
    output = keras.layers.Dense(Action_size)(dense1)
    filtered_output = keras.layers.Multiply(name="Qvalue")([output, actions_input])

    model = keras.Model(inputs=[frames_input, actions_input], outputs=filtered_output)
    model.summary()
    optimiser = keras.optimizers.Adam()
    model.compile(optimizer=optimiser, loss=huber_loss)
    return model


def get_action(history, epsilon, step, model):
    # observe_step_num 都属于探索阶段
    if np.random.rand() <= epsilon or step <= params.get("observe_step_num"):
        return random.randrange(Action_size)
    else:
        q_value = model.predict([history, np.ones(Action_size).reshape(1, Action_size)])
        return np.argmax(q_value[0])


def store_memory(memory, history, action, reward, next_history):
    # save sample <s,a,r,s"> to the replay memory
    memory.append((history, action, reward, next_history))


def get_one_hot(targets, nb_classes):
    # 对 [1, 2, 1, 2] 这样的target进行编码，nb_classes为类别数
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def train_memory_batch(memory, model):
    mini_batch = random.sample(memory, params.get("batch_size"))
    state = np.zeros((params.get("batch_size"), Input_shape[0], Input_shape[1], Input_shape[2]))
    next_state = np.zeros((params.get("batch_size"), Input_shape[0], Input_shape[1], Input_shape[2]))
    action = []
    reward = []

    # Create an array that will carry what the target q values will be - based on our target networks weights
    target_q = np.zeros((params.get("batch_size"),))

    # Fill up our arrays with our minibatch
    for id, val in enumerate(mini_batch):
        state[id] = val[0]
        next_state[id] = val[3]
        action.append(val[1])
        reward.append(val[2])

    # 对所有的action预测Q值
    actions_mask = np.ones((params.get("batch_size"), Action_size))
    next_q_values = model.predict([next_state, actions_mask])

    # Fill out target q values based on the max q value in the next state
    for i in range(params.get("batch_size")):
        # TD_target
        target_q[i] = reward[i] + params.get("gamma") * np.amax(next_q_values[i])

    action_one_hot = get_one_hot(action, Action_size)
    # Apply one hot mask onto target vector
    # This results in a vector that has the max q value in the position corresponding to the action
    target_one_hot = action_one_hot * target_q[:, None]     # (128, 3)

    # Then we fit the model
    # We map the state and the action from the memory bank to the q value of that state action pair
    # s,a -> q(s,a|w)
    h = model.fit([state, action_one_hot], target_one_hot, epochs=1, batch_size=params.get("batch_size"), verbose=0)

    return h.history["loss"][0]


def train():
    env = gym.make("PongDeterministic-v4")

    memory = deque(maxlen=params.get("replay_memory"))

    # Start episode counter
    episode_number = 0

    epsilon = params.get("initial_epsilon")
    epsilon_decay = (params.get("initial_epsilon") - params.get("final_epsilon")) / params.get("epsilon_anneal_num")

    # Start global step
    global_step = 0

    model = atari_model()
    model_target = atari_model_target()

    log_dir = "{}/run-{}-log".format(params.get("train_dir"), "MK10")
    file_writer = tf.summary.create_file_writer(log_dir)

    while episode_number < params.get("n_episodes"):
        done = False

        # Initialise other values
        step = 0
        score = 0
        loss = 0.0

        start_time = time.time()

        # Initialise environment
        observation = env.reset()

        # For the very start of the episode, we will do nothing but observe
        # This way we can get a sense of whats going on
        for _ in range(random.randint(1, params.get("no_op_steps"))):
            observation, _, _, _ = env.step(1)

        # At the start of the episode there are no preceding frames
        # So we just copy the initial states into a stack to make the state history
        state = pre_processing(observation)
        state_history = np.stack((state, state, state, state), axis=2)
        state_history = np.reshape([state_history], (1, 84, 84, 4))

        while not done:
            if params.get("render"):
                env.render()

            # 使用target_model获取action
            action = get_action(state_history, epsilon, global_step, model_target)
            real_action = action + 1

            # 在观测步数之内，会一直进行探索。
            if global_step > params.get("observe_step_num") and epsilon > params.get("final_epsilon"):
                epsilon -= epsilon_decay

            observation, reward, done, info = env.step(real_action)

            next_state = pre_processing(observation)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            state_history_w_next = np.append(next_state, state_history[:, :, :, :3], axis=3)

            # Update score
            score += reward

            # Save the (s, a, r, s") set to memory
            store_memory(memory, state_history, action, reward, state_history_w_next)

            # 观测结束则开始训练
            if global_step > params.get("observe_step_num"):
                loss = loss + train_memory_batch(memory, model)
                # 更新target_model
                if global_step % params.get("update_target_model_steps") == 0:
                    model_target.set_weights(model.get_weights())
                    print("UPDATING TARGET WEIGHTS")

            state_history = state_history_w_next

            global_step += 1
            step += 1

            # game over
            if done:
                if global_step <= params.get("observe_step_num"):
                    # 观测阶段
                    current_position = "observe"

                elif params.get("observe_step_num") < global_step <= params.get("observe_step_num") + params.get("epsilon_anneal_num"):
                    # 使用 epsilon 探索阶段
                    current_position = "explore"
                else:
                    current_position = "train"

                spend_time = round(time.time() - start_time, 2)

                print("current stage: {}, "
                      "epsilon: {} , "
                      "episode: {}, "
                      "score: {}, "
                      "global_step: {}, "
                      "avg loss: {}, "
                      "step: {}, "
                      "memory length: {}, "
                      "spend_time: {} ".format(current_position, epsilon, episode_number, score, global_step,
                                                 loss / float(step), step, len(memory), spend_time))

                # Save model
                if episode_number % 100 == 0 or (episode_number + 1) == params.get("n_episodes"):
                    file_name = "pong_model_{}.h5".format(episode_number)
                    model_path = os.path.join(params.get("train_dir"), file_name)
                    model.save(model_path)

                with file_writer.as_default():
                    tf.summary.scalar("loss", loss / float(step), episode_number)
                    tf.summary.scalar("score", score, episode_number)
                    tf.summary.scalar("spend_time", spend_time, episode_number)

                if episode_number % 100 == 0:
                    params["render"] = True
                else:
                    params["render"] = False

                episode_number += 1

    file_writer.close()


def test_env():
    env = gym.make("PongDeterministic-v4")
    obs = env.reset()
    print(obs)
    print(obs.shape)        # (210, 160, 3)
    env.render()


if __name__ == "__main__":
    train()
    # test_env()

