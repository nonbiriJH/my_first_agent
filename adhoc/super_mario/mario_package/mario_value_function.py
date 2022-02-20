from keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv3D,MaxPooling3D,Lambda

class value_function:
    def __init__(self, state_shape, n_action, memory_recent_length):
        self.n_action = n_action
        self.memory_recent_length = memory_recent_length

        self.state_shape = list(state_shape)
        self.state_shape.insert(0,self.memory_recent_length)

    def build_core_ann (self):
        model = Sequential()
        input_layer = Input(self.state_shape)
        model.add(input_layer)
        model.add(Lambda(lambda x: (0.21 * x[:,:,:,:,:1]) + (0.72 * x[:,:,:,:,1:2]) + (0.07 * x[:,:,:,:,-1:])))
        model.add(Conv3D(filters=6,kernel_size=8,strides=4,activation='relu',padding="same", data_format="channels_last")) #out(60,64,32)
        model.add(MaxPooling3D(pool_size=(self.memory_recent_length,2,2)))#out(30,32,32)
        model.add(Conv3D(filters=12,kernel_size=4,strides=2,padding="same",activation='relu'))#out(15,16,64)
        model.add(MaxPooling3D(pool_size=(self.memory_recent_length,2,2)))#out(7,8,64)
        model.add(Flatten())
        model.add(Dense(self.n_action*2,activation='relu'))
        model.add(Dense(self.n_action,activation='linear'))
        return model