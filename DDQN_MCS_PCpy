# DDQN - Pandemic Control - 1
      # MontecarloSimulation based Pandemic curing Artificial Intellegent Bot Designed using Double Deep Q Networks
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.models import model_from_json
import tensorflow as tf
from collections import deque
import random
from tqdm import tqdm
import os
from matplotlib import style
import time
import matplotlib.pyplot as plt

style.use("ggplot")

import pickle

DISCOUNT = 0.95
REPLAY_MEMORY_SIZE = 200000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 5000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 200  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = 33_50_000  # For model save
MEMORY_FRACTION = 0.20
render_every = 5 * 10 ** 34
# Environment settings
EPISODES = 1
discrete = False
save_model = False
model_load = True  # after 1 run
randomness = False  # After 1-2 runs
# Exploration settings
model_weights_save = "models/lunar1_google.h5"
model_name_save = "models/lunar1_model.json"
model_name_load = "models/lunar1_model(3).json"
model_weights_load = "models/lunar1_google(3
).h5"
# load_model =  models.load_model('C:\\Users\dell-3568\PycharmProjects\Life\models\like.model',compile=True)
#  Stats settings
AGGREGATE_STATS_EVERY = 3  # episodes
SHOW_PREVIEW = False

try:
    file_pickle = open("ep_rewards".format(), "rb")
    load = pickle.load(file_pickle)
    file_pickle.close()
    ep_rewards = load[:-4]
except:
    ep_rewards = []
# bservation = (env.observation_space.shape[0], env.observation_space.shape[1],env.observation_space.shape[2])
# print(observation)
#env = PandemicOutbreak()


def DeepNetworkAgent(OBSERVATION_SPACE_VALUES, ACTION_SPACE_SIZE,env):
    ################
    epsilon = 1  # not a constant, going to be decayed
    EPSILON_DECAY = 0.999
    MIN_EPSILON = 0.01

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    # Memory fraction, used mostly when trai8ning multiple agents
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    # backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    # Own Tensorboard class
    class ModifiedTensorBoard(TensorBoard):

        # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.step = 1
            self.writer = tf.summary.FileWriter(self.log_dir)

        # Overriding this method to stop creating default log writer
        def set_model(self, model):
            pass

        # Overrided, saves logs with our step number
        # (otherwise every .fit() will start writing from 0th step)
        def on_epoch_end(self, epoch, logs=None):
            self.update_stats(**logs)

        # Overrided
        # We train for one batch only, no need to save anything at epoch end
        def on_batch_end(self, batch, logs=None):
            pass

        # Overrided, so won't close writer
        def on_train_end(self, _):
            pass

        # Custom method for saving own metrics
        # Creates writer, writes custom metrics and closes writer
        def update_stats(self, **stats):
            self._write_logs(stats, self.step)

    # Agent class
    class DQNAgent:
        def __init__(self, OBSERVATION_SPACE_VALUES, ACTION_SPACE_SIZE):

            # Main model
            self.OBSERVATION_SPACE_VALUES = OBSERVATION_SPACE_VALUES
            self.ACTION_SPACE_SIZE = ACTION_SPACE_SIZE
            # self.model = load_model
            self.model = self.create_model3()

            # Target network
            self.target_model = self.create_model3()
            self.target_model.set_weights(self.model.get_weights())

            # An array with last n steps for training
            self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

            # Custom tensorboard object
            self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

            # Used to count when to update target network with main network's weights
            self.target_update_counter = 0


        def create_model3(self):
            # Neural Net for Deep-Q learning Model
            model = Sequential()
            model.add(Dense(32, input_dim=self.OBSERVATION_SPACE_VALUES, activation='relu'))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(16))
            model.add(Dense(32))
            model.add(Dense(32))
            model.add(Dense(64))
            model.add(Dense(self.ACTION_SPACE_SIZE, activation='linear'))
            model.compile(loss='mse',
                          optimizer=Adam(lr=0.004))
            # model.load_model('2x256___500.00max__197.32avg___25.00min__1594509056.model',compile = False)

            if model_load == True:
                json_file = open(model_name_load, "r")
                loaded_model_json = json_file.read()
                json_file.close()
                model_ = model_from_json(loaded_model_json)

                model_.load_weights(model_weights_load)
                model = model_
                print("loaded model...###")
                '''model = Sequential()
                for layer in model_.layers[:-2]:
                   model.add(layer)
                model.add(Dense(32))
                model.add(Dense(self.ACTION_SPACE_SIZE, activation='linear'))'''
                model.layers[0].trainable = True
                model.layers[1].trainable = True
                model.layers[2].trainable = True
                model.layers[3].trainable = True

                # model.layers[4].trainable = True
                model.compile(loss='mse', optimizer=Adam(lr=0.008))
                print(model.summary())

            return model

        def create_model2(self):

            if model_load == True:
                json_file = open(model_name_load, "r")
                loaded_model_json = json_file.read()
                json_file.close()
                model = model_from_json(loaded_model_json)

                model.load_weights(model_weights_load)
                print("loaded model...###")

                model.compile(loss='mse', optimizer=Adam(lr=0.9), metrics=['accuracy'])
                print(model.summary())
            return model

        def create_model(self):
            model = Sequential()

            model.add(Conv2D(256, (3, 3),
                             input_shape=OBSERVATION_SPACE_VALUES))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(256, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
            model.add(Dense(64))

            model.add(Dense(ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
            model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

            if model_load == True:
                json_file = open(model_name_load, "r")
                loaded_model_json = json_file.read()
                json_file.close()
                model = model_from_json(loaded_model_json)
                model.load_weights(model_weights_load)
                print("loaded model...###")
                model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])
            return model

        # Adds step's data to a memory replay array
        # (observation space, action, reward, new observation space, done)
        def update_replay_memory(self, transition):
            self.replay_memory.append(transition)

        # Trains main network every step during episode
        def train(self, terminal_state, step):

            # Start training only if certain number of samples is already saved
            if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
                return

            # Get a minibatch of random samples from memory replay table
            minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

            # Get current states from minibatch, then query NN model for Q values
            current_states = np.array([transition[0] for transition in minibatch]) / 255
            current_qs_list = self.model.predict(current_states)

            # Get future states from minibatch, then query NN model for Q values
            # When using target network, query it, otherwise main network should be queried
            new_current_states = np.array([transition[3] for transition in minibatch]) / 255
            future_qs_list = self.target_model.predict(new_current_states)

            X = []
            y = []

            # Now we need to enumerate our batches
            for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

                # If not a terminal state, get new q from future states, otherwise set it to 0
                # almost like with Q Learning, but we use just part of equation here
                if not done:
                    max_future_q = np.max(future_qs_list[index])
                    new_q = reward + DISCOUNT * max_future_q
                else:
                    new_q = reward

                # Update Q value for given state
                current_qs = current_qs_list[index]
                current_qs[action] = new_q

                # And append to our training data
                X.append(current_state)
                y.append(current_qs)

            # Fit on all samples as one batch, log only on terminal state

            self.model.fit(np.array(X) / 255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False,
                           callbacks=[self.tensorboard] if terminal_state else None)

            # Update target network counter every episode
            if terminal_state:
                self.target_update_counter += 1

            # If counter reaches set value, update target network with weights of main network
            if self.target_update_counter > UPDATE_TARGET_EVERY:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0

        # Queries main network for Q values given current observation space (environment state)
        def get_qs(self, state):
            # print(state)
            return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

        # --------------------------------------------------------------------------------------------------------------

    agent = DQNAgent(OBSERVATION_SPACE_VALUES=OBSERVATION_SPACE_VALUES,
                     ACTION_SPACE_SIZE=ACTION_SPACE_SIZE)

    # Iterate over episodes

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        while not done:

            # This part stays mostly the same, the change is to query a model for Q values
            # action = np.argmax(agent.get_qs(current_state))
            if randomness == True:
                # print("randomnessON")
                if np.random.random() > epsilon:
                    # Get action from Q table
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, ACTION_SPACE_SIZE)
            else:
                action = np.argmax(agent.get_qs(current_state))

            new_state, reward, done, info = env.step(action)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward
            if episode % render_every == 0:
                env.render()

            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)

            current_state = new_state
            step += 1

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD and save_model == True:
                # agent.model.save(
                #  f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

                model_json = agent.model.to_json()
                with open(model_name_save, "w") as json_file:
                    json_file.write(model_json)
                agent.model.save_weights(model_weights_save)
                print("----------------------------------model SAVED-----------------------------------------")
        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
        if episode % AGGREGATE_STATS_EVERY == 0 and episode % 3 == 0:
            print(min_reward, epsilon)
            if randomness:
                print(epsilon)
        live_ep_rewards = ep_rewards[(len(ep_rewards) - 100):]
        open_file = open("ep_rewards", "wb")
        pickle.dump(ep_rewards, open_file)
        open_file.close()
        if episode % 10 == 0:
            plt.plot([i for i in range(len(live_ep_rewards))], live_ep_rewards)
            plt.ylabel(f"Reward - last 1k episodes")
            plt.xlabel("episode ")
            plt.show()

    # moving_avg = np.convolve(ep_rewards, np.ones((render_every,)) / render_every, mode='valid')

    plt.plot([i for i in range(len(ep_rewards))], ep_rewards)
    plt.ylabel(f"Reward {render_every}ma")
    plt.xlabel("episode ")
    plt.show()

'''
print("-=-=-", env.observation_space.shape, env.action_space)
print(env.observation_space.shape[0])
if discrete == False:
    state = 1
    act_state = 1
    for i in range(len(env.observation_space.shape)):
        state *= env.observation_space.shape[i]
    for i in range(len(env.action_space.shape)):
        act_state *= env.action_space.shape[i]
    print(state, act_state)
    DeepNetworkAgent(state, env.action_space.n)
else:
    DeepNetworkAgent(env.observation_space.shape, env.action_space.n)'''
