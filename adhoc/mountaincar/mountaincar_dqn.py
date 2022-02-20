import gym
import numpy as np
from mountaincar_package.mountaincar_value_function import value_function
from mountaincar_package.mountaincar_agent import agent
from tensorflow.keras.optimizers import Adam

env = gym.make('MountainCar-v0')
dim_action = env.action_space.n
dim_observation = env.observation_space.shape[0]
# build ann to model action value
# Input observation output action value for each in action space
val_func = value_function(dim_observation, dim_action)
model = val_func.build_core_ann()
model.summary()

# An agent defines policy and update rule, but allowed customised value function
# also defines batch size (default 32) for mini batch
# memory_recent_length=2 needs input shape to be (2,n_obs)
agt = agent(model, dim_action, nb_steps_warmup=1000, target_model_update= 1e-4, memory_sample_limit = 10000, memory_recent_length=2) # recent
dqn = agt.build_agent()
# compiling defines loss function to minimise
# Adam is an optimised back propagation
dqn.compile(optimizer=Adam(learning_rate=1e-3),
            metrics=['mae'])  # mean absolute error loss

# train by train_on_batch: sample from random samles of the past experiences of agent's batch size
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
dqn.save_weights('weight', overwrite=True)

#remove processor when testing
#dqn.processor=None
score = dqn.test(env, nb_episodes=5, visualize=False)
np.mean(score.history['episode_reward'])

