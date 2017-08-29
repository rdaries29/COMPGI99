# Course: COMPGI99 - Thesis
# Institution: University College London
# Developer: Russel Daries (16079408)

import tensorflow as tf
import numpy as np
import pandas as pd
from game_state import game_state
from common_methods import *
from memory_profiler import profile

class base_ac_network(object):

    def __init__(self,action_space,observation_space,thread_index,device = "/cpu:0"):

        self.num_actions = action_space
        self.thread_index = thread_index
        self._device = device
        self._observation_space = observation_space

    def prepare_loss(self, entropy_beta):
        with tf.device(self._device):
            # taken action (input for policy)
            self.a = tf.placeholder(dtype=tf.float32, shape = [None, self.num_actions])

            # temporary difference (R-V) (input for policy)
            self.td = tf.placeholder(dtype=tf.float32, shape = [None,])

            # avoid NaN with clipping when value in pi becomes zero
            log_pi = tf.log(tf.clip_by_value(self.policy, 1e-20, 1.0))

            # policy entropy
            entropy = -tf.reduce_sum(self.policy * log_pi, reduction_indices=1)

            # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
            policy_loss = - tf.reduce_sum(tf.reduce_sum(tf.multiply(log_pi, self.a), reduction_indices=1) * self.td + entropy * entropy_beta)

            # R (input for value)
            self.r = tf.placeholder(dtype=tf.float32, shape = [None,])

            # value loss (output)
            # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
            value_loss = 0.5 * tf.nn.l2_loss(self.r - self.value)

            # gradienet of policy and value are summed up
            self.total_loss = policy_loss + value_loss

    def predict_value(self,x):
        raise NotImplementedError()

    def predict_policy(self,x):
        raise NotImplementedError()

    def predict_value_and_policy(self,x):
        raise NotImplementedError()

    def get_vars(self):
        raise NotImplementedError()

    def sync_from(self,src_network, name = None):

        src_vars = src_network.get_vars()
        dst_vars = self.get_vars()

        sync_ops = []

        with tf.device(self._device):
            with tf.name_scope(name,"base_ac_network",[]) as name:
                for(src_vars,dst_vars) in zip(src_vars,dst_vars):

                    sync_op = tf.assign(dst_vars,src_vars)
                    sync_ops.append(sync_op)

                return tf.group(*sync_ops,name=name)

    def _fc_variable(self, weight_shape):
        input_channels = weight_shape[0]
        output_channels = weight_shape[1]
        d = 1.0 / np.sqrt(input_channels)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d))
        return weight, bias

    def _conv_variable(self, weight_shape):
        w = weight_shape[0]
        h = weight_shape[1]
        input_channels = weight_shape[2]
        output_channels = weight_shape[3]
        d = 1.0 / np.sqrt(input_channels * w * h)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d))
        return weight, bias

    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")


class lstm_ac_network(base_ac_network):

    def __init__(self,action_space,observation_space,thread_index,all_paths,device = "/cpu:0"):

        base_ac_network.__init__(self,action_space,observation_space,thread_index,device)

        scope_name = "network_" + str(self.thread_index) + "_thread"

        self.frame_stack_size = 4
        self.state_dims = self._observation_space
        self.lstm_cell_size = 256
        self.scope_name = scope_name
        self.table_path = all_paths[3]

        with tf.device(self._device), tf.variable_scope(scope_name) as scope:

            self.state_arrays = tf.placeholder(shape=[None, self.state_dims * self.frame_stack_size], dtype=tf.float32)
            input_layer = tf.reshape(self.state_arrays, [-1, self.state_dims, self.frame_stack_size, 1])

            # 3 Convolutional Layers as specified in Mnih DQN paper
            # 32 20x20 feature map
            conv_layer_1 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=input_layer, kernel_size=[4, 2], padding='valid', filters=32, strides=(1, 1),activation=tf.nn.relu))

            # 64 9x9 feature map
            conv_layer_2 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=conv_layer_1, kernel_size=[3, 2], padding='valid', filters=64, strides=(1, 1),activation=tf.nn.relu))

            # 64 7x7 feature map
            conv_layer_3 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=conv_layer_2, kernel_size=2, padding='valid', filters=64, strides=(1, 1),activation=tf.nn.relu))

            conv_2d_flatten = tf.contrib.layers.flatten(conv_layer_3)

            fully_connected = tf.layers.batch_normalization(tf.layers.dense(inputs=conv_2d_flatten, units=self.lstm_cell_size, activation=tf.nn.relu))

            fully_connected = tf.reshape(fully_connected,[1,-1,self.lstm_cell_size])

            # place holder for LSTM unrolling time step size.
            self.step_size = tf.placeholder(tf.float32, [1])

            self.lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_cell_size, state_is_tuple=True)

            self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, 256])
            self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, 256])
            self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state0,
                                                                    self.initial_lstm_state1)

            # Unrolling LSTM up to LOCAL_T_MAX time steps. (= 5time steps.)
            # When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
            # Unrolling step size is applied via self.step_size placeholder.
            # When forward propagating, step_size is 1.
            # (time_major = False, so output shape is [batch_size, max_time, cell.output_size])
            lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,fully_connected,initial_state=self.initial_lstm_state,sequence_length=self.step_size,time_major=False,scope=scope)

            # lstm_outputs: (1,5,256) for back prop, (1,1,256) for forward prop.

            lstm_outputs = tf.reshape(lstm_outputs, [-1, self.lstm_cell_size])

            # policy (output)
            self.policy = tf.nn.softmax(tf.layers.dense(inputs=lstm_outputs, units=self.num_actions, activation=None))

            # value (output)
            v_ = tf.layers.dense(inputs=lstm_outputs, units=1, activation=None)
            self.value = tf.reshape(v_, [-1])

            self.reset_state()

    def reset_state(self):
        self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1,self.lstm_cell_size]),np.zeros([1,self.lstm_cell_size]))


    # prediction for value network

    def predict_value(self,sess,x):

        prev_lstm_state_out = self.lstm_state_out

        prediction = sess.run(self.value, feed_dict = {self.state_arrays: x,
                                                                   self.initial_lstm_state0: self.lstm_state_out[0],
                                                                   self.initial_lstm_state1: self.lstm_state_out[1],
                                                                   self.step_size:[1]})
        self.lstm_state_out = prev_lstm_state_out
        return prediction[0]

    # prediction for policy network
    def predict_policy(self,sess,x):
        prediction,self.lstm_state_out = sess.run([self.policy,self.lstm_state],
                                                       feed_dict = {self.state_arrays: x,
                                                                    self.initial_lstm_state0: self.lstm_state_out[0],
                                                                     self.initial_lstm_state1: self.lstm_state_out[1],
                                                                     self.step_size:[1]})
        return prediction[0]

    # function for single prediction
    def predict_single(self, x):
        return self.predict_p(x[None, :])[0]

    # prediction for p & v network
    def predict_policy_and_value(self,sess,x):

        policy_out,value_out = sess.run([self.policy,self.value],
                             feed_dict = {self.state_arrays: x,
                                          self.initial_lstm_state0: self.lstm_state_out[0],
                                          self.initial_lstm_state1: self.lstm_state_out[1],
                                          self.step_size:[1]})


        return policy_out[0],value_out[0]


    def get_vars(self):

        tempt_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)

        return tempt_var

    def choose_action_test(self, pi_values):
        return np.random.choice(range(len(pi_values)), p=pi_values)

    @profile
    def process_test(self, sess,experiments,game_name,rand_seeding,construct_agent,all_paths,training_mode,count_rendering):

        thread_index = -1
        done = False

        all_rewards = []
        episode_steps = []
        all_scores = []
        max_time_steps = 1000
        discount = 0.99
        episode_reward_undiscounted = 0
        episode_reward_discounted = 0

        game = game_state(game_name,rand_seeding,construct_agent,thread_index,all_paths,training_mode,count_rendering)

        for episode in range(experiments):

            # t_max times loop
            for i in range(max_time_steps):
                temp_state = np.reshape(game.s_t,(1,-1))
                pi_, value_ = self.predict_policy_and_value(sess, temp_state)
                action = self.choose_action_test(pi_)
                action_vector = np.zeros([self.num_actions])
                action_vector[action] = 1

                # process game
                next_state,reward,done,_ = game.env.step(action_vector)

                next_frame = game.frame_preprocess(next_state)

                game.add_frame_train(next_frame)

                game.s_t1 = game.compile_frames_train()

                limited_reward = limit_return(reward)

                episode_reward_undiscounted += limited_reward
                episode_reward_discounted += (limited_reward * discount ** i)

                # s_t1 -> s_t
                game.update()

                if done:
                    terminal_end = True
                    all_rewards.append(episode_reward_discounted)
                    all_scores.append(episode_reward_undiscounted)
                    episode_steps.append(i+1)
                    episode_reward_undiscounted = 0
                    episode_reward_discounted = 0
                    game.env.reset()
                    self.reset_state()
                    break

        return np.mean(all_rewards), np.std(all_rewards), np.mean(episode_steps), np.std(episode_steps), np.mean(all_scores), np.std(all_scores)

    def count_parameters(self, sess):
        """Returns the number of parameters of a computational graph."""
        parameter_trace_network = dict()
        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        n_params = 0

        for k, v in zip(variables_names, values):
            print('-----------------------------------------')
            print('Variable:\t{:20s} \t{:20} \tShape: {:}\t{:20} parameters'.format(k, str(v.dtype), v.shape, v.size))
            parameter_trace_network[k] = [str(v.dtype), v.shape, v.size]

            n_params += v.size

        parameter_trace_network['number_params'] = n_params

        print('\n\n-----------------------------\nTotal # parameters:\t{}'.format(n_params))

        df = pd.DataFrame(parameter_trace_network)
        writer = pd.ExcelWriter(self.table_path + 'parameter_trace_network.xlsx', engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Sheet1')
        writer.save()
        return n_params