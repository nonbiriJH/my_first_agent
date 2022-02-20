from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from mario_package.mario_value_function import value_function
from mario_package.mario_agent import agent
from tensorflow.keras.optimizers import Adam
import numpy as np

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

dim_action = env.action_space.n
dim_observation = env.observation_space.shape
memory_recent_length=1

# build ann to model action value
# Input observation output action value for each in action space
val_func = value_function(dim_observation, dim_action, memory_recent_length)
model = val_func.build_core_ann()
model.summary()

# An agent defines policy and update rule, but allowed customised value function
# also defines batch size (default 32) for mini batch
# memory_recent_length=2 needs input shape to be (2,n_obs)
agt = agent(model, dim_action, target_model_update= 1e-4, memory_sample_limit = 1000, memory_recent_length= memory_recent_length) # recent
dqn = agt.build_agent()
# compiling defines loss function to minimise
# Adam is an optimised back propagation
dqn.compile(optimizer=Adam(learning_rate=1e-3),metrics=['mae'])  # mean absolute error loss

dqn.load_weights('weight')
score = dqn.test(env, nb_episodes=3, visualize=True, nb_max_episode_steps= 2000)
np.mean(score.history['episode_reward'])

