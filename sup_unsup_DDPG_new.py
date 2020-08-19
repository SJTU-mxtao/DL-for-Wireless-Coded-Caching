import scipy.io as scio
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
# from keras.engine.training import collect_trainable_weights
import json
from keras import initializers
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from Ou import OU
import timeit

F = 50  # file
U = 20  # uers
N = 7  # bs stations
OU = OU()  # Ornstein-Uhlenbeck Process
M = 5.
T = 500

# req = np.random.randint(F, size=(T, U, F))
# pre_req = np.random.randint(20, size=(T, F))
index = 1  # 1是class预测的结果 2是线性预测的结果
dataFile = 'E:\\coded_cache\\DDPG\\final\\Youtube_jurnal.mat'
data_file = scio.loadmat(dataFile)
req = data_file['view_count_norm']
if index == 1:
    pre_req = req
    pre = 'E:\\coded_cache\\DDPG\\final\\cost_nn_pre_gam015_M5_hope.mat'               # 结果保存路径
else:
    pre_req = data_file['view_count_lin_r']
    pre = 'E:\\coded_cache\\DDPG\\final\\cost_lin_pre_gam1_M5_r34.mat'              # 结果保存路径
# Generate dummy daa

dataFile = 'E:\\coded_cache\\DDPG\\Ng_delay.mat'
data_file = scio.loadmat(dataFile)
Ng = data_file['Ng']
delay = data_file['delay']
weigths_path = 'E:\\coded_cache\\DDPG\\final\\state_action_gam015_M5_hope.h5'
# 读入s_t的初始状态
dataFile = 'E:\\coded_cache\\DDPG\\final\\train_set_Youtube_pre_gam015_M5_hope.mat'
data_file = scio.loadmat(dataFile)
state_ini = data_file['state']
def func_cost(cache, req1, cache_last):
    R = np.zeros([N + 1, F, U])
    com_lambda = np.zeros([N + 1, F, U])
    cost = 0.
    for f in range(F):
        for u in range(U):
            for k in range(N + 1):
                if k == 0:
                    R[k][f][u] = R[k][f][u] + delay[u][Ng[u][k] - 1]
                elif Ng[u][k] > 0:
                    for i in range(k):
                        R[k][f][u] = R[k][f][u] + cache[f, Ng[u][i] - 1] * delay[u][Ng[u][i] - 1]
                        com_lambda[k][f][u] = com_lambda[k][f][u] + cache[f, Ng[u][i] - 1]
                    R[k][f][u] = R[k][f][u] + (1 - com_lambda[k][f][u]) * delay[u][Ng[u][k] - 1]

    for f in range(F):

        for u in range(U):
            cost = cost + max(R[:, f, u]) * req1[f]
    cost = cost*0.25789
    for f in range(F):
        for n in range(N):
            cost = cost + 1.5*max(cache[f][n] - cache_last[f][n], 0)  # 此时算出的结果大概是30左右
    cost=cost*0.03333
    # for n in range(N):    #惩罚总的cache超过大小
    #     if sum(cache[:,n])>M:
    #         cost = cost +10000
    return cost


def playGame(train_indicator=1):  # 1 means Train, 0 means simply Run

    BUFFER_SIZE = 1000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.0005  # Target Network HyperParameters 每次更新target网络的0.5%
    LRA = 0.00001  # Learning rate for Actor
    LRC = 0.0005 # Learning rate for Critic

    action_dim = 350  # 七个基站的存储状态 F*N
    state_dim = 400  # F*N+F 七个基站的存储状态+五十个文件的流行度

    np.random.seed(1337)
    vision = False

    max_steps = 1600
    reward = 0
    done = False
    step = 0
    epsilon = 1.
    indicator = 0

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    # ########################################### create actor and critic network #####################################
    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer
    # ########################################### load actor parameter ################################################

    print("Experiment Start. Let's go!")

    actor.model.load_weights(weigths_path)
    actor.target_model.load_weights(weigths_path)
    s_t = np.zeros(F * N + F)  # 每个基站对每个文件的存储状态和预测的F个文件的流行度 前F个为流行度，后面是cache
    s_t_next = np.random.rand(F * N + F)
    result_class = np.zeros(T)
    a_t_end = []
    a_t_ori = []
    a_t_noise = []
    a_t_opt = np.zeros([T, F * N])
    a_t_temp = np.zeros([T, F * N])
    opt_cost = np.zeros(T)
    opt_cost_temp = np.zeros(T)
    opt_reward = 10000000
    EXPLORE = 32950.
    episode_count = 140
    total_reward_temp = np.zeros(episode_count)
    mse_temp = []
    for i in range(episode_count):  # 跑多少圈，也就是把这T天的数据看多少遍
        # initialize #
        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
        total_reward = 0.
        # s_t[[70, 79, 80,  100,   101,   102,   150,   152,   162,   200,   201,   202,   250,   251,   252,
        # 300, 302, 312, 350,   351,   352]] = 1.
        # s_t[0:F] = pre_req[0, :]
        s_t = state_ini[0, :]
        for t in range(T - 1):
            loss = 0
            epsilon -= 1.0 / EXPLORE  # 探索的力度
            a_t = np.zeros(action_dim)
            a_t_scale = np.zeros(action_dim)
            noise_t = np.zeros(action_dim)
            a_t_original = actor.model.predict(np.expand_dims(s_t, axis=0))

            #########################################################是不是应该*M#############？？？？？？？？？？？？？？？
            # record a_t orignal
            a_t_ori.append(a_t_original[0])
            # #################### design the noise of exploration ##################################
            for n in range(N):
                for f in range(F):
                    noise_t[n * 50 + f] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][n * 50 + f], 0
                                                                                          , 0, 0.3)
            for m in range(action_dim):
                a_t[m] = a_t_original[0][m] + noise_t[m]
            # record a_t with noise
            a_t_noise.append(a_t)
            # ################## scaling #####################################
            for kk in range(F * N):
                if a_t[kk] > 1:
                    a_t[kk] = 1
                if a_t[kk] < 0:
                    a_t[kk] = 0
            for n in range(N):
                if sum(a_t[n * F:(n + 1) * F]) > 0:
                    a_t_scale[n * F:(n + 1) * F] = M * a_t[n * F:(n + 1) * F] / sum(a_t[n * F:(n + 1) * F])
            for kk in range(F * N):
                if a_t_scale[kk] > 1:
                    a_t_scale[kk] = 1
                if a_t_scale[kk] < 0:
                    a_t_scale[kk] = 0
            a_t = a_t_scale
            for n in range(N):
                if sum(a_t[n * F:(n + 1) * F]) > 0:
                    a_t_scale[n * F:(n + 1) * F] = M * a_t[n * F:(n + 1) * F] / sum(a_t[n * F:(n + 1) * F])
            for kk in range(F * N):
                if a_t_scale[kk] > 1:
                    a_t_scale[kk] = 1
                if a_t_scale[kk] < 0:
                    a_t_scale[kk] = 0
            a_t = a_t_scale
            # #################### calculate reward and update state and add buffer ##########
            cache = a_t.reshape(N, F)
            cache = np.transpose(cache)
            cache_last = s_t[F:F * N + F].reshape(N, F)
            cache_last = np.transpose(cache_last)
            r_t = func_cost(cache, req[t, :], cache_last)
            s_t_next[0:F] = pre_req[t + 1, :]
            s_t_next[F:F * N + F] = a_t
            done = True
            # #################### Add replay buffer and update state ###################
            buff.add(s_t, a_t, r_t, s_t_next, done)
            s_t = s_t_next
            # Do the batch update 计算y_t 即要更新critic network的Q值
            if t >= 2:
                batch = buff.getBatch(BATCH_SIZE)
                states = np.asarray([e[0] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_states = np.asarray([e[3] for e in batch])
                dones = np.asarray([e[4] for e in batch])
                y_t = np.asarray([e[2] for e in batch])
                target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])
                for k in range(len(batch)):
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]

                if (train_indicator):
                    loss = critic.model.train_on_batch([states, actions], y_t)
                    mse_temp.append(loss)
                    a_for_grad = actor.model.predict(states)
                    grads =  critic.gradients(states, a_for_grad) # 这个梯度是否绝对值太大 做归一化

                    if i >= 80:
                        actor.train(states, grads)
                        actor.target_train()
                    critic.target_train()
                    total_reward += r_t
            # s_t = s_t1

            # print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)

            step += 1
            '''if done:
                break'''
            # print(step)
            if i == episode_count - 1:
                result_class[t] = r_t
                a_t_end.append(a_t)  # 记录所有的a_t
            a_t_temp[t, :] = a_t
            opt_cost_temp[t] = r_t
        if opt_reward > total_reward:
            for kk in range(T):
                for bb in range(350):
                  a_t_opt[kk][bb] = a_t_temp[kk][bb]
            opt_reward = total_reward
            for cc in range(T):
                opt_cost[cc] = opt_cost_temp[cc]
            if opt_reward < 7500:
                if (train_indicator):
                    print("Now we save model")
                    actor.model.save_weights("actormodelgam2_r.h5", overwrite=True)
                    with open("actormodelgam2_r.json", "w") as outfile:
                        json.dump(actor.model.to_json(), outfile)

                    critic.model.save_weights("criticmodelgam2_r.h5", overwrite=True)
                    with open("criticmodelgam2_r.json", "w") as outfile:
                        json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) + "-th Episode  : Cost " + str(total_reward))
        print("Total Step: " + str(step))
        print("")
        total_reward_temp[i] = total_reward

    # env.end()  # This is for shutting down TORCS
    if index == 1:

        scio.savemat(pre, {'cost_nn_pre': result_class, 'a_t_end': a_t_end, 'a_t_ori': a_t_ori, 'a_t_noise': a_t_noise,
                           'a_t_opt': a_t_opt, 'opt_cost': opt_cost, 'total_reward_temp': total_reward_temp, 'mse_temp':
                               mse_temp})
    else:

        scio.savemat(pre,
                     {'cost_lin_pre': result_class, 'a_t_end': a_t_end, 'a_t_ori': a_t_ori, 'a_t_noise': a_t_noise,
                      'a_t_opt': a_t_opt, 'opt_cost': opt_cost, 'total_reward_temp': total_reward_temp, 'mse_temp':
                               mse_temp})

    print("Finish.")


if __name__ == "__main__":
    playGame()
