import numpy as np
import math
import keras
import keras.initializers
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
# from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from keras.layers.normalization import BatchNormalization

HIDDEN1_UNITS = 800
HIDDEN2_UNITS = 400
F = 50

class ActorNetwork(object):

    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        # Now create the model
        self.model, self.weights, self.state = self.create_actor_network(
            state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(
            state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
        self.params_grad = tf.gradients(
            self.model.output, self.weights, -1.00*self.action_gradient)

        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(
            LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[
                i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size, action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])
        #BN = BatchNormalization()(S)
        h0 = Dense(800, activation='relu')(S)
        h1 = Dense(400, activation='relu')(h0)
        #
        # n1 = Dense(F, activation='softmax')(h1)
        # n2 = Dense(F, activation='softmax')(h1)
        # n3 = Dense(F, activation='softmax')(h1)
        # n4 = Dense(F, activation='softmax')(h1)
        # n5 = Dense(F, activation='softmax')(h1)
        # n6 = Dense(F, activation='softmax')(h1)
        # n7 = Dense(F, activation='softmax')(h1)
        # V = keras.layers.concatenate([n1, n2, n3, n4, n5, n6, n7], axis=-1)
        V = Dense(350, activation='sigmoid')(h1)
        # Steering = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        # Acceleration = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        # Brake = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        # V = merge([Steering,Acceleration,Brake],mode='concat')
#        V = Dense(action_dim, activation='sigmoid', kernel_initializer=keras.initializers.RandomNormal(
#            mean=0.0, stddev=0.9, seed=None))(h1)
#         keras.layers.merge
#         V = Dense(action_dim, activation='sigmoid')(h1)
        model = Model(input=S, output=V)
        return model, model.trainable_weights, S
