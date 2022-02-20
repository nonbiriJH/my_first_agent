from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten

class value_function:
    def __init__(self, n_state, n_action):
        self.n_state = n_state
        self.n_action = n_action

    def build_core_ann (self):
        model = Sequential()
        #n_state input, n_action output, 2 hiddent 24-unit layers with relu activation
        model.add(Flatten(input_shape=(2,self.n_state))) #input need to be flat
        model.add(Dense(25,activation='relu'))
        model.add(Dense(self.n_action,activation='linear'))
        return model