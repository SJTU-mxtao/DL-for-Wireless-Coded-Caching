import keras
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, LocallyConnected1D, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
import scipy.io as scio
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras import initializers
from keras.layers import Dense, Flatten, Input, merge, Lambda
from keras import backend as K

def Relu(x):
    return K.relu(x, max_value=1)
# #####################load data #####################

F = 50  # file
U = 20  # uers
N = 7  # bs stations
M = 5.
T = 660
state_size = 400
action_size = 350
dataFile = 'E:\\coded_cache\\DDPG\\final\\train_set_Youtube_pre_gam2_M5_hope.mat'
opt = 'E:\\coded_cache\\DDPG\\final\\state_action_gam2_M5_hope.mat'  # 预测的action存储路径
weigths_path = 'E:\\coded_cache\\DDPG\\final\\state_action_gam2_M5_hope.h5'
data_file = scio.loadmat(dataFile)
state = data_file['state']
action = data_file['action']
#action = action/5.0
pro_view = state[:, 0:F]
pro_lambda = state[:, F:F+F*N]
# ##################### creat model #####################
#PV = Input(shape=[F])
#PL = Input(shape=[F*N])
# BN_S = BatchNormalization()(S)
# BN = BatchNormalization()(S)
#v1 = Dense(100, activation='relu')(PV)
#l1 = Dense(700, activation='relu')(PL)
#h1 = keras.layers.concatenate([v1, l1])
model = Sequential()
model.add(Dense(units=800, activation='relu', input_dim=400))
model.add(Dense(units=400, activation='relu'))
#model.add(Dense(units=350, activation='sigmoid'Relu))
model.add(Dense(units=350, activation='sigmoid'))
#model.add(Activation(Relu))

# S = Input(shape=[F+F*N])
# BN = BatchNormalization()(S)
# h0 = Dense(800, activation='relu')(BN)
# h1 = Dense(400, activation='relu')(h0)
# h2 = Dense(350)(h1)
# V = keras.activations.relu(h2, max_value=1)
# n1 = Dense(F, activation='softmax')(h1)
# n2 = Dense(F, activation='softmax')(h1)
# n3 = Dense(F, activation='softmax')(h1)
# n4 = Dense(F, activation='softmax')(h1)
# n5 = Dense(F, activation='softmax')(h1)
# n6 = Dense(F, activation='softmax')(h1)
# n7 = Dense(F, activation='softmax')(h1)
# op1 = keras.layers.concatenate([n1, n2], axis=-1)
# op2 = keras.layers.concatenate([op1, n3], axis=-1)
# op3 = keras.layers.concatenate([op2, n4], axis=-1)
# op4 = keras.layers.concatenate([op3, n5], axis=-1)
# op5 = keras.layers.concatenate([op4, n6], axis=-1)
# V = keras.layers.concatenate([n1, n2, n3, n4, n5, n6, n7], axis=-1)

# Steering = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
# Acceleration = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
# Brake = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
# V = merge([Steering,Acceleration,Brake],mode='concat')
#        V = Dense(action_dim, activation='sigmoid', kernel_initializer=keras.initializers.RandomNormal(
#            mean=0.0, stddev=0.9, seed=None))(h1)
#         keras.layers.merge
#         V = Dense(action_dim, activation='sigmoid')(h1)
# model = Model(input=S, output=V)
lr = 0.000025

adam = Adam(lr=lr)
model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
# model.fit([pro_view[0:600], pro_lambda[0:600]], action[0:600], epochs=130, batch_size=100, validation_split=0.05,
#           callbacks=[TensorBoard(log_dir='./tmp/log')])
model.fit(state[0:600], action[0:600], epochs=100, batch_size=32, validation_split=0.05,
          callbacks=[TensorBoard(log_dir='./tmp/log')])
#model.load_weights(weigths_path)
# pre_action = model.predict([pro_view[600:660], pro_lambda[600:660]])
pre_action = model.predict(state)

scio.savemat(opt, {'opt_action': action, 'pre_action': pre_action, 'lr': lr})
model.save_weights(weigths_path)