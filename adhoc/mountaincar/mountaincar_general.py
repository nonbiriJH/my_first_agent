import gym
from mountaincar_package.mountaincar_value_function import value_function
from tensorflow.keras.optimizers import Adam
from rl.policy import BoltzmannQPolicy

env = gym.make('MountainCar-v0')
dim_action = env.action_space.n
dim_observation = env.observation_space.shape[0]
action_selector = BoltzmannQPolicy()
# build ann to model action value
# Input observation output action value for each in action space
val_func = value_function(dim_observation, dim_action)
model = val_func.build_core_ann()
model.summary()


import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, Layer, Dense
import numpy as np
from rl.util import *
def clipped_masked_error(args):
        y_true, y_pred, mask = args
        loss = huber_loss(y_true, y_pred, np.inf)
        loss *= mask  # apply element-wise mask
        return K.sum(loss, axis=-1)

y_pred = model.output
y_true = Input(name='y_true', shape=(dim_action,))
mask = Input(name='mask', shape=(dim_action,))
loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_true, y_pred, mask])

ins = [model.input] if type(model.input) is not list else model.input
trainable_model = Model(inputs=ins + [y_true, mask], outputs=[loss_out, y_pred])
trainable_model.summary()
assert len(trainable_model.output_names) == 2
combined_metrics = {trainable_model.output_names[1]: 'mse'}
losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
trainable_model.compile(optimizer=Adam(learning_rate=1e-3), loss=losses, metrics=combined_metrics)

layer = model.layers[-2]
y = Dense(3 + 1, activation='linear')(layer.output)
outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], axis=1, keepdims=True), output_shape=(3,))(y)
newmodel = model = Model(inputs=model.input, outputs=outputlayer)
newmodel.summary()
model.summary()

nepisode = 10
ncomplete = 0
ret_list = []
for i_episode in range(nepisode):
    observation = env.reset()
    ret = 0
    for t in range(200):
        #take action
        q_vals = model.predict(observation)
        action = action_selector.select_action(q_vals)
        #store episode trajectory
        observation, reward, done, info = env.step(action)
        ret += reward
        if done:
            ret_list.append(ret)
            break
env.close()


print(sum(ret_list)/ncomplete)
