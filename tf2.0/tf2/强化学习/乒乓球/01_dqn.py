import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import gym
import random
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
import os


params = {
    "optimiser_learning_rate": 0.00025,
    "observe_step_num": 100000,
    "batch_size": 32,
    
    "initial_epsilon": 1,
    "epsilon_anneal_num": 500000,
    "final_epsilon": 0.01,
    
    "gamma": 0.99,
    "replay_memory": 400000,    # 40w=20GB
    "n_episodes": 100000,
    "no_op_steps": 2,
    "render": True,
    "update_target_model_steps": 100000,
    "train_dir": "log"
}


Input_shape = (84, 84, 4)
Action_size = 3


def pre_processing(observation):
    """ 预处理 """
    processed_observation = rgb2gray(observation) * 255
    # Convering to grey converts it to normalised values [0,255] -> [0,1]
    # We need to store the values as integers in the next step to save space, so we need to undo the normalisation
    processed_observation = resize(processed_observation, (84, 84), mode="constant")
    # Convert from floating point to integers ranging from [0,255]
    processed_observation = np.uint8(processed_observation)
    return processed_observation


# Define the huber loss function
# Used for the Keras model compile function
# Needs to take y and yhat and output loss
# Needs investigation
def huber_loss(y, q_value):
    error = tf.abs(y - q_value)
    quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)
    return loss


# Create the model to approximate Qvalues for a given set of states and actions
def atari_model():
    # Define the inputs
    frames_input = keras.Input(shape=Input_shape, name="frames")
    actions_input = keras.layers.Input(shape=(Action_size,), name="action_mask")

    # Normalise the inputs from [0,255] to [0,1] - to make processing easier
    normalised = keras.layers.Lambda(lambda x: x / 255.0, name="normalised")(frames_input)
    # Conv1 is 16 8x8 filters with a stride of 4 and a ReLU
    conv1 = keras.layers.Conv2D(16, 8, 4, activation="relu")(normalised)
    # Conv2 is 32 4x4 filters with a stride of 2 and a ReLU
    conv2 = keras.layers.Conv2D(32, 4, 2, activation="relu")(conv1)
    # Flatten the output from Conv2
    conv2_flatten = keras.layers.Flatten()(conv2)
    # Then a fully connected layer with 256 ReLU units
    dense1 = keras.layers.Dense(256, activation="relu")(conv2_flatten)
    # Then a fully connected layer with a unit to map to each of the actions and no activation
    output = keras.layers.Dense(Action_size)(dense1)
    # Then we multiply the output by the action mask
    # When trying to find the value of all the actions this will be a mask full of 1s
    # When trying to find the value of a specific action, the mask will only be 1 for a single action
    filtered_output = keras.layers.Multiply(name="Qvalue")([output, actions_input])

    # Create the model
    # Create a model that maps frames and actions to the filtered output
    model = keras.Model(inputs=[frames_input, actions_input], outputs=filtered_output)
    # Print a summary of the model
    model.summary()
    # Define optimiser
    optimiser = keras.optimizers.Adam()
    # Compile model
    model.compile(optimizer=optimiser, loss=huber_loss)
    # Return the model
    return model


# Create an identical model to use as a target
def atari_model_target():
    # Define the inputs
    frames_input = keras.Input(shape=Input_shape, name="frames")
    actions_input = keras.layers.Input(shape=(Action_size,), name="action_mask")

    # Normalise the inputs from [0,255] to [0,1] - to make processing easier
    normalised = keras.layers.Lambda(lambda x: x / 255.0, name="normalised")(frames_input)
    # Conv1 is 16 8x8 filters with a stride of 4 and a ReLU
    conv1 = keras.layers.Conv2D(16, 8, 4, activation="relu")(normalised)
    # Conv2 is 32 4x4 filters with a stride of 2 and a ReLU
    conv2 = keras.layers.Conv2D(32, 4, 2, activation="relu")(conv1)
    # Flatten the output from Conv2
    conv2_flatten = keras.layers.Flatten()(conv2)
    # Then a fully connected layer with 256 ReLU units
    dense1 = keras.layers.Dense(256, activation="relu")(conv2_flatten)
    # Then a fully connected layer with a unit to map to each of the actions and no activation
    output = keras.layers.Dense(Action_size)(dense1)
    # Then we multiply the output by the action mask
    # When trying to find the value of all the actions this will be a mask full of 1s
    # When trying to find the value of a specific action, the mask will only be 1 for a single action
    filtered_output = keras.layers.Multiply(name="Qvalue")([output, actions_input])

    # Create the model
    # Create a model that maps frames and actions to the filtered output
    model = keras.Model(inputs=[frames_input, actions_input], outputs=filtered_output)
    # Print a summary of the model
    model.summary()
    # Define optimiser
    optimiser = keras.optimizers.Adam()
    # Compile model
    model.compile(optimizer=optimiser, loss=huber_loss)
    # Return the model
    return model


# get action from model using epsilon-greedy policy
def get_action(history, epsilon, step, model):
    if np.random.rand() <= epsilon or step <= params.get("observe_step_num"):
        return random.randrange(Action_size)
    else:
        q_value = model.predict([history, np.ones(Action_size).reshape(1, Action_size)])
        return np.argmax(q_value[0])


# save sample <s,a,r,s"> to the replay memory
def store_memory(memory, history, action, reward, next_history):
    memory.append((history, action, reward, next_history))


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


# train model by random batch
def train_memory_batch(memory, model):
    # Sample a minibatch
    mini_batch = random.sample(memory, params.get("batch_size"))
    # Create empty arrays to load our minibatch into
    # These objects have multiple values hence need defined shapes
    state = np.zeros((params.get("batch_size"), Input_shape[0], Input_shape[1], Input_shape[2]))
    next_state = np.zeros((params.get("batch_size"), Input_shape[0], Input_shape[1], Input_shape[2]))
    # These objects have a single value, so we can just create a list that we append later
    action = []
    reward = []
    # Create an array that will carry what the target q values will be - based on our target networks weights
    target_q = np.zeros((params.get("batch_size"),))

    # Fill up our arrays with our minibatch
    for id, val in enumerate(mini_batch):
        state[id] = val[0]
        print(val[0].shape)
        next_state[id] = val[3]
        action.append(val[1])
        reward.append(val[2])

    # We want the model to predict the q value for all actions hence:
    actions_mask = np.ones((params.get("batch_size"), Action_size))
    # Get the target model to predict the q values for all actions
    next_q_values = model.predict([next_state, actions_mask])

    # Fill out target q values based on the max q value in the next state
    for i in range(params.get("batch_size")):
        # Standard discounted reward formula
        # q(s,a) = r + discount * cumulative future rewards
        target_q[i] = reward[i] + params.get("gamma") * np.amax(next_q_values[i])

    # Convert all the actions into one hot vectors
    action_one_hot = get_one_hot(action, Action_size)
    # Apply one hot mask onto target vector
    # This results in a vector that has the max q value in the position corresponding to the action
    target_one_hot = action_one_hot * target_q[:, None]

    # Then we fit the model
    # We map the state and the action from the memory bank to the q value of that state action pair
    # s,a -> q(s,a|w)
    h = model.fit([state, action_one_hot], target_one_hot, epochs=1, batch_size=params.get("batch_size"), verbose=0)

    # Return the loss
    # Its just for monitoring progress
    return h.history["loss"][0]


def train():
    # Define which game to play
    env = gym.make("PongDeterministic-v4")

    # Create a space for our memory
    # We will use a deque - double ended que
    # This will result in a max size so that once all the space is filled,
    # older entries will be removed to make room for new
    memory = deque(maxlen=params.get("replay_memory"))

    # Start episode counter
    episode_number = 0

    # Set epsilon
    epsilon = params.get("initial_epsilon")
    # Define epsilon decay
    epsilon_decay = (params.get("initial_epsilon") - params.get("final_epsilon")) / params.get("epsilon_anneal_num")

    # Start global step
    global_step = 0

    # Define model
    model = atari_model()

    # Define target model
    model_target = atari_model_target()

    # Define where to store logs
    log_dir = "{}/run-{}-log".format(params.get("train_dir"), "MK10")
    # Pass graph to TensorBoard

    file_writer = tf.summary.create_file_writer(log_dir)

    # Start the optimisation loop
    while episode_number < params.get("n_episodes"):

        # Initialisise done as false
        done = False

        # Initialise other values
        step = 0
        score = 0
        loss = 0.0

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

        # Perform while we still have lives
        while not done:
            # Render the image if selected to do so
            if params.get("render"):
                env.render()

            # Select an action based on our current model
            action = get_action(state_history, epsilon, global_step, model_target)
            # Convert action from array numbers to real numbers
            real_action = action + 1

            # After we"re done observing, start scaling down epsilon
            if global_step > params.get("observe_step_num") and epsilon > params.get("final_epsilon"):
                epsilon -= epsilon_decay

            # Record output from the environment
            observation, reward, done, info = env.step(real_action)

            # Process the observation
            next_state = pre_processing(observation)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            # Update the history with the next state - also remove oldest state
            state_history_w_next = np.append(next_state, state_history[:, :, :, :3], axis=3)

            # Update score
            score += reward

            # Save the (s, a, r, s") set to memory
            store_memory(memory, state_history, action, reward, state_history_w_next)

            # Train model
            # Check if we are done observing
            if global_step > params.get("observe_step_num"):
                loss = loss + train_memory_batch(memory, model)
                # Check if we are ready to update target model with the model we have been training
                if global_step % params.get("update_target_model_steps") == 0:
                    model_target.set_weights(model.get_weights())
                    print("UPDATING TARGET WEIGHTS")
            state_history = state_history_w_next

            # print("step: ", global_step)
            global_step += 1
            step += 1

            # Check if episode is over - lost all lives in breakout
            if done:
                # Check if we are still observing
                if global_step <= params.get("observe_step_num"):
                    current_position = "observe"
                # Check if we are still annealing epsilon
                elif params.get("observe_step_num") < global_step <= params.get("observe_step_num") + params.get("epsilon_anneal_num"):
                    current_position = "explore"
                else:
                    current_position = "train"
                # Print status
                print(
                    "current position: {}, epsilon: {} , episode: {}, score: {}, global_step: {}, avg loss: {}, step: {}, memory length: {}"
                        .format(current_position, epsilon, episode_number, score, global_step, loss / float(step), step,
                                len(memory)))

                # Save model every 100 episodes and final episode
                if episode_number % 100 == 0 or (episode_number + 1) == params.get("n_episodes"):
                    file_name = "pong_model_{}.h5".format(episode_number)
                    model_path = os.path.join(params.get("train_dir"), file_name)
                    model.save(model_path)

                # Add loss and score  data to TensorBoard
                # loss_summary = tf.Summary(
                #     value=[tf.Summary.Value(tag="loss", simple_value=loss / float(step))])
                # file_writer.add_summary(loss_summary, global_step=episode_number)
                #
                # score_summary = tf.Summary(
                #     value=[tf.Summary.Value(tag="score", simple_value=score)])
                # file_writer.add_summary(score_summary, global_step=episode_number)

                with file_writer.as_default():
                    tf.summary.scalar("loss", loss / float(step), episode_number)
                    tf.summary.scalar("score", score, episode_number)

                # Increment episode number
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

