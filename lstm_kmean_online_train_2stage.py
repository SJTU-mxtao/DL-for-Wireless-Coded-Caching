import numpy
# import pandas
from collections import deque
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, LocallyConnected1D, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
import scipy.io as scio
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import to_categorical
#from sklearn.preprocessing import LabelEncoder
from keras import initializers
import random
###################################define buffer#####################################
class PreBuffer(object):           #从object（python默认的类）继承

    def __init__(self, buffer_size):    # 定义ReplayBuffer类自己的变量
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()         ## 双向队列

    def getBatch(self, batch_size):    # 定义ReplayBuffer类自己的函数
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, ip, op):
        experience = (ip, op)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()   #最左端元素出列，即去除最早进入的状态
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0
############################begin###################################

sequence_length = 11
time_step = sequence_length-1
buf_size = 1000
PreBuf = PreBuffer(buf_size)
BATCH_SIZE = 32
# 读入数据
save_name = 'Lstm_kmeans_pre_C6W10_online_t660_hope_two_stage.mat'
dataFile = 'D:\\dataset\\music\\lstm_eachf\\Lstm_kmeans_C6W10T660_online_train_hope.mat'
data_file = scio.loadmat(dataFile)
data_ori = data_file['kmeans_sample_norm']
num_sample = data_file['samp_each_t_c']
num_cluster = int(data_file['num_cluster'])
T = 660-time_step
explore = 100
ori_view = np.zeros([T, 50, explore, num_cluster])
pre_view = np.zeros([T, 50, explore, num_cluster])
total_loss_all=np.zeros([explore, num_cluster])
np.random.seed(10)
for i in range(num_cluster):
    PreBuf.erase()  # 清空buffer空间
    data = data_ori[:, :, :, i]
    #data = data_ori
    print(data.shape)
    # ############################################ train
    dropout_value = 0.5
    window_size = time_step
    model = Sequential()
    model.add(LSTM(time_step * 2, input_shape=(time_step, 1), return_sequences=True))
    model.add(Dropout(dropout_value))

    # Second recurrent layer with dropout
    model.add(LSTM(time_step * 2, return_sequences=True))
    model.add(Dropout(dropout_value))

    # Third recurrent layer
    model.add(LSTM(time_step, return_sequences=False))
    # Output layer (returns the predicted value)
    model.add(Dense(units=1))

    # Set activation function
    model.add(Activation('linear'))

    adam = Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])

    Pre_Pf_opt = np.zeros([T, 50])

    total_opt_loss = 10000000
    Opt_Pf = np.zeros([T, 50])
    # training with previous 500 time slots
    for k in range(explore):
        Pre_Pf = np.zeros([T, 50])
        total_loss = 0
        for t in range(500):
            if num_sample[t, i]>0:
                if num_sample[t, i]<2:
                    train_X = data[t, 0:int(num_sample[t, i]), 0:time_step]  # 第一维是样本，第二维是天，第三维才是每天的数据。到最后一天的前一天的所有数据 包括比特币价格 期货量等因素
                    train_Y = data[t, 0:int(num_sample[t, i]), time_step]  # 最后一行也就是最后一天49的所有数据
                    pre_temp = model.predict(np.expand_dims(train_X, -1))

                    # score = model.evaluate(np.expand_dims(train_X, -1), train_Y, batch_size=1)
                    # print(score[0])
                    # print(1)

                    Pre_Pf[t, 0] = pre_temp[0]
                    Opt_Pf[t, 0] = train_Y
                    PreBuf.add(train_X[0], train_Y)
                else:
                    train_X = data[t, 0:int(num_sample[t, i]), 0:time_step]  # 第一维是样本，第二维是天，第三维才是每天的数据。到最后一天的前一天的所有数据 包括比特币价格 期货量等因素
                    train_Y = data[t, 0:int(num_sample[t, i]), time_step]  # 最后一行也就是最后一天49的所有数据
                    pre_temp = model.predict(np.expand_dims(train_X, -1))

                    # score = model.evaluate(np.expand_dims(train_X, -1), train_Y, batch_size=1)
                    # print(score[0])
                    # print(2)

                    for dk in range(len(pre_temp)):
                        Pre_Pf[t, dk] = pre_temp[dk]
                        Opt_Pf[t, dk] = train_Y[dk]
                    # train
                    if k == 0:
                        for ak in range(len(train_X)):
                            PreBuf.add(train_X[ak], train_Y[ak])
            if PreBuf.num_experiences>2:
                batch = PreBuf.getBatch(BATCH_SIZE)
                ip = np.asarray([e[0] for e in batch])
                op = np.asarray([e[1] for e in batch])
                # print(ip.shape)
                # print(t)
                loss = model.train_on_batch(np.expand_dims(ip, -1), op)
                total_loss = total_loss + loss[0]
        print("train"+str(k) + "explore error" + str(total_loss))
        total_loss_all[k, i] = total_loss
        # for kkk in range(T):
        #     pre_view[kkk, :, k, i] = Pre_Pf[kkk]
        #     ori_view[kkk, :, k, i] = Opt_Pf[kkk]

    # #########################################################################
    # test with 501-600 time slots ############################################

    for k in range(1):
        Pre_Pf = np.zeros([T, 50])
        total_loss = 0
        for t in range(500, 600):
            if num_sample[t, i]>0:
                if num_sample[t, i]<2:
                    train_X = data[t, 0:int(num_sample[t, i]), 0:time_step]  # 第一维是样本，第二维是天，第三维才是每天的数据。到最后一天的前一天的所有数据 包括比特币价格 期货量等因素
                    train_Y = data[t, 0:int(num_sample[t, i]), time_step]  # 最后一行也就是最后一天49的所有数据
                    pre_temp = model.predict(np.expand_dims(train_X, -1))

                    # score = model.evaluate(np.expand_dims(train_X, -1), train_Y, batch_size=1)
                    # print(score[0])
                    # print(1)

                    Pre_Pf[t, 0] = pre_temp[0]
                    Opt_Pf[t, 0] = train_Y
                    PreBuf.add(train_X[0], train_Y)
                else:
                    train_X = data[t, 0:int(num_sample[t, i]), 0:time_step]  # 第一维是样本，第二维是天，第三维才是每天的数据。到最后一天的前一天的所有数据 包括比特币价格 期货量等因素
                    train_Y = data[t, 0:int(num_sample[t, i]), time_step]  # 最后一行也就是最后一天49的所有数据
                    pre_temp = model.predict(np.expand_dims(train_X, -1))

                    # score = model.evaluate(np.expand_dims(train_X, -1), train_Y, batch_size=1)
                    # print(score[0])
                    # print(2)

                    for dk in range(len(pre_temp)):
                        Pre_Pf[t, dk] = pre_temp[dk]
                        Opt_Pf[t, dk] = train_Y[dk]
                    # train
                    for ak in range(len(train_X)):
                        PreBuf.add(train_X[ak], train_Y[ak])
            if PreBuf.num_experiences>2:
                batch = PreBuf.getBatch(BATCH_SIZE)
                ip = np.asarray([e[0] for e in batch])
                op = np.asarray([e[1] for e in batch])
                # print(ip.shape)
                # print(t)
                loss = model.train_on_batch(np.expand_dims(ip, -1), op)
                total_loss = total_loss + loss[0]
        print("test"+str(k) + "explore error" + str(total_loss))
        total_loss_all[k, i] = total_loss
        for kkk in range(T):
            pre_view[kkk, :, k, i] = Pre_Pf[kkk]
            ori_view[kkk, :, k, i] = Opt_Pf[kkk]
    print(i)
    scio.savemat(save_name, {'pre': pre_view, 'opt': ori_view, 'loss': total_loss_all, 'buf_size': buf_size,
                             'BATCH_SIZE': BATCH_SIZE})
# Pre_Pf = Pre_Pf.reshape(len(train_data)-11500, 50))
# pre = 'pre_pf.mat'
# opt = 'opt_pf.mat'
# scio.savemat(pre, {'pre_pf': Pre_Pf})
# scio.savemat(opt, {'opt_pf': Opt_Pf})
learn_rate = 0.0005
scio.savemat(save_name, {'pre': pre_view, 'opt': ori_view, 'loss': total_loss_all, 'buf_size': buf_size, 'BATCH_SIZE': BATCH_SIZE})


