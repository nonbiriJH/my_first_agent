from email import policy
from termios import N_MOUSE
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.optimizers import Adam

def build_core_ann (state, action):
    model = Sequential()
    #n_state input, n_action output, 2 hiddent 24-unit layers with relu activation
    model.add(Flatten(input_shape=(1,state))) #input need to be flat
    model.add(Dense(24,activation='relu'))
    model.add(Dense(24,activation='relu'))
    model.add(Dense(action,activation='linear'))
    return model

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
def build_agent(model, action):
    return DQNAgent( #update action value function as per TD target r_t + gamma * max_a Q(s_t+1, a)
        model=model
        ,policy=BoltzmannQPolicy() #choose action stochastically on softmax of q value
        ,memory=SequentialMemory(limit=50000,window_length=1) #required for mini-batch
        ,nb_actions=action
        ,nb_steps_warmup=10
        ,target_model_update=1e-2)

import gym
env = gym.make('CartPole-v0')

#build ann to model action value
#Input observation output action value for each in action space
model = build_core_ann(4,2)
model.summary()

#An agent defines policy and update rule, but allowed customised value function
#also defines batch size (default 32) for mini batch 
dqn = build_agent(model,2)
#compiling defines loss function to minimise
#Adam is an optimised back propagation
dqn.compile(optimizer=Adam(learning_rate=1e-3),metrics=['mae']) #mean absolute error loss
#train by train_on_batch: sample from random samles of the past experiences of agent's batch size
dqn.fit(env,nb_steps=500,visualize=False,verbose=1,log_interval=200)
score=dqn.test(env,nb_episodes=100,visualize=False)
np.mean(score.history['episode_reward'])
