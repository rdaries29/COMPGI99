# Course: COMPGI99 - Thesis
# Institution: University College London
# Developer: Russel Daries (16079408)

# Define class for Agent
from collections import deque
import numpy as np
import tensorflow as tf
import random
import pandas as pd
import pdb

from common_methods import *


def discretize_actions(output_nodes_vec):

    maximum_indices = tf.argmax(output_nodes_vec,axis=2)
    action_matrix= quantize(maximum_indices)

    maximum_values = tf.reduce_max(output_nodes_vec,axis=2)
    maximum_values = tf.reduce_sum(maximum_values,axis=1)

    return action_matrix,maximum_values

def quantize(maximum_index_vector):

    dict_mapping = {0:-1, 1:-0.5, 2:0, 3:0.5, 4:1}
    dict_keys = tf.Variable(initial_value=[0,1,2,3,4])
    dict_values = tf.Variable(initial_value=[-1,-0.5,0,0.5,1])
    # pdb.set_trace()
    table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(dict_keys,dict_values), -1)
    out_tensor = table.lookup(tf.cast(maximum_index_vector,dtype=tf.int32))

    # action_matrix = tf.map_fn(fn = mapping,elems=maximum_index_vector)

    return out_tensor

def mapping(input_vec):

    dict_mapping = {0:-1, 1:-0.5, 2:0, 3:0.5, 4:1}

    output_vec = [dict_mapping[k] for k in input_vec]

    return output_vec


class Agent:

    def __init__(self, enviroment, learning_rate):

        self.frame_stack_size = 4
        self.experience_buffer_size = 500
        self.discount = 0.99
        self.max_episodes = 300
        self.epsilon = 0.05
        self.select = 'RMS'
        self.result_display = 10
        self.max_time_steps = 300
        self.image_size = 84

        self.experience_buffer_episodes = deque(maxlen=self.experience_buffer_size)
        self.episode_lens = np.array([])

        self.frame_buffer_train = deque(maxlen = self.frame_stack_size)
        self.frame_buffer_test = deque(maxlen= self.frame_stack_size)

        self.save_best_model = False
        self.best = 120
        self.seeding = 200
        self.learning_rate = learning_rate

        self.env = enviroment
        self.state_dims = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.steps = 0
        self.alpha = 0.004
        self.discrete_levels = 5
        self.build_dqn()

    def action(self,sess,experiments,construct_agent):

        rewards = []
        experiment_lengths = []
        scores = []

        for episode in range(experiments):

            state = self.env.reset()

            pre_proc_state = self.frame_preprocess(state)

            self.add_frame_test(pre_proc_state)

            state = self.compile_frame_test()

            episode_reward = 0
            score = 0

            for time_step in range(self.max_time_steps):

                # if(construct_agent):
                #
                #     self.env.render()
                self.env.render()

                # Re-shape the state
                temp_state = np.reshape(state,(1,-1))

                action,_ = self.q_prediction(sess,temp_state)

                # Step in enviroment
                next_state_test,reward,done,_ = self.env.step(action)

                limited_reward = limit_return(reward)

                next_frame_test = self.frame_preprocess(next_state_test)

                # Add next frame to the stack

                self.add_frame_test(next_frame_test)

                state = self.compile_frame_test()

                # Check this rewarding structure
                episode_reward += (limited_reward * self.discount ** time_step)
                score +=reward

                if(done):

                    rewards.append(episode_reward)
                    experiment_lengths.append([time_step+1])
                    scores.append(score)
                    break
                #
                # if(done):
                #
                #     rewards.append(-1*self.discount**episode)

        return rewards,experiment_lengths,scores


    def create_experience_replay_buffer(self,sess):

        for iteration in range(self.experience_buffer_size):

            state = self.env.reset()

            pre_proc_state = self.frame_preprocess(state)

            self.add_frame_train(pre_proc_state)

            state = self.compile_frame_train()

            for time_step in range(self.max_time_steps):

                # Check state dimensions for this input
                temp_state = np.reshape(state,(1,-1))

                action, _ = self.q_prediction(sess,temp_state)

                next_state,reward,done,_ = self.env.step(action)

                limited_reward = limit_return(reward)

                pre_proc_frame = self.frame_preprocess(next_state)

                self.add_frame_train(pre_proc_frame)

                next_state = self.compile_frames_train()

                if(done):

                    next_state = None

                self.add_episode((state,action,limited_reward,next_state))

                state = next_state

                buffer_length = len(self.experience_buffer_episodes)

                if(self.experience_buffer_size == buffer_length):
                    return

                if(done):
                    break


    def build_dqn(self):

        tf.reset_default_graph()
        tf.set_random_seed(self.seeding)

        # Place holders for 4 screen stacked inputs to NN
        self.state_arrays = tf.placeholder(shape=[None, self.state_dims * self.frame_stack_size], dtype=tf.float32)
        self.next_state_arrays = tf.placeholder(shape=[None, self.state_dims * self.frame_stack_size], dtype=tf.float32)
        # self.current_actions_holder = tf.placeholder(shape=[None,self.num_actions], dtype=tf.float32)
        # self.next_actions_holder  = tf.placeholder(shape=[None,self.num_actions], dtype=tf.float32)
        self.rewards_holder = tf.placeholder(shape=[None,], dtype=tf.float32)
        self.done_holder = tf.placeholder(shape=[None,], dtype=tf.float32)

        # Create convolutional networks
        with tf.name_scope("predict"):
            with tf.name_scope("Q_net"):
                self.q_network_current = self.create_q_network(self.num_actions,self.state_arrays)

        with tf.name_scope("estimate"):
            with tf.name_scope("T_net"):
                self.t_network_current = self.create_q_network(self.num_actions,self.next_state_arrays)

        q_prediction_matrix = tf.reshape(self.q_network_current,shape=[-1,3,5])

        q_prediction_actions,q_prediction_value = discretize_actions(q_prediction_matrix)

        self.q_predict = q_prediction_actions

        self.current_av = q_prediction_value

        t_prediction_matrix = tf.reshape(self.q_network_current,shape=[-1,3,5])

        t_prediction_actions,t_prediction_value = discretize_actions(t_prediction_matrix)

        self.t_predict = t_prediction_actions

        self.max_q_new = t_prediction_value * self.done_holder

        self.new_av = self.rewards_holder + (self.discount * self.max_q_new)

        #temporal difference
        self.td = tf.clip_by_value(self.new_av-self.current_av,clip_value_min=-1,clip_value_max=1)

        # Be sure to clip the gradients so they don't vanish

        self.loss = tf.reduce_mean(tf.square(self.td))

        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.99, staircase=True)
        # pdb.set_trace()

        if(self.select=='RMS'):
            self.optimize = tf.train.RMSPropOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(self.loss)
        elif(self.select=='ADAM'):
            self.optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        else:
            pass

        # Record loss in tensorflow plots to be used for tensorboard
        tf.summary.scalar("loss", self.loss)
        self.summary_merged = tf.summary.merge_all()

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        with tf.name_scope("update_target_net"):
            self.target_network_up = []

            # Updating the target network with the Q Network parameters
            q_variables = tf.get_collections(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_net")
            t_variables = tf.get_collections(tf.GraphKeys.TRAINABLE_VARIABLES, scope="t_net")

            for v_source, v_target in zip(q_variables, t_variables):
                update_op = v_target.assign_sub(self.alpha* (v_target - v_source))
                self.target_network_up.append(update_op)

            self.target_network_up = tf.group(*self.target_network_up)



    # Q Network function creation
    def create_q_network(self,num_actions,state_dims):

        # input_layer = tf.reshape(state_dims,[-1,self.image_size,self.image_size,self.frame_stack])
        input_layer = tf.reshape(state_dims,[-1,self.state_dims,self.frame_stack_size,1])
        # input_layer = tf.expand_dims(input_layer,dim=3)

        # 3 Convolutional Layers as specified in Mnih DQN paper
        # 32 20x20 feature map
        conv_layer_1 = tf.layers.conv2d(inputs=input_layer,kernel_size=[4,2],padding='valid',filters=32,strides=(1,1),activation=tf.nn.relu)

        # 64 9x9 feature map
        # pdb.set_trace()

        conv_layer_2 = tf.layers.conv2d(inputs=conv_layer_1,kernel_size=[3,2],padding='valid',filters=64,strides=(1,1),activation=tf.nn.relu)

        # 64 7x7 feature map
        conv_layer_3 = tf.layers.conv2d(inputs=conv_layer_2, kernel_size=2, padding='valid', filters=64, strides=(1,1),activation=tf.nn.relu)

        # conv_2d_flatten = tf.reshape(conv_layer_3,[-1,9 * 1 * 64])
        conv_2d_flatten = tf.contrib.layers.flatten(conv_layer_2)

        fully_connected = tf.layers.dense(inputs=conv_2d_flatten,units=512,activation=tf.nn.relu)

        Q = tf.layers.dense(inputs=fully_connected,units=num_actions*self.discrete_levels)

        return Q

    # Function to train network
    def train(self,sess,current_states,next_states,rewards,done_flag):

        return sess.run([self.optimize,self.loss,self.summary_merged,self.q_network_current],
                        feed_dict = {self.state_arrays: current_states,
                                     self.next_state_arrays: next_states,
                                     self.rewards_holder: rewards,
                                     self.done_holder: done_flag})

        # return sess.run([self.optimize, self.loss, self.summary_merged, self.q_network_current],
        #             feed_dict={self.state_arrays: current_states,
        #                        self.next_state_arrays: next_states,
        #                        self.current_actions_holder: current_actions,
        #                        self.next_actions_holder: next_actions,
        #                        self.rewards_holder: rewards,
        #                        self.done_holder: done_flag})

    # Function call to predict next Q (return)
    def q_prediction(self,sess,x_input):

        return sess.run([self.q_predict,self.q_network_current], feed_dict = {self.state_arrays:x_input})

    def q_prediction_target(self,sess,x_input):

        return sess.run([self.t_predict,self.t_network_current], feed_dict = {self.next_state_arrays:x_input})


    def add_episode(self,sample):

        self.experience_buffer_episodes.append(sample)

    def sample_episodes(self,batch):

        batch = min(batch,len(self.experience_buffer_episodes))

        return random.sample(tuple(self.experience_buffer_episodes),batch)

    def add_frame_train(self,frame,repeat=1):

        for count in range(repeat):
            self.frame_buffer_train.append(frame)

    def add_frame_test(self,frame,repeat=1):

        for count in range(repeat):
            self.frame_buffer_test.append(frame)

    def compile_frames_train(self):

        compiled_frames_train = np.array(list(self.frame_buffer_train))

        return compiled_frames_train

    def compile_frame_test(self):

        compiled_frames_test = np.array(list(self.frame_buffer_test))

        return compiled_frames_test

    def replay(self,num_episodes,batch_size,training_mode=False):

        with tf.Session() as sess:

            sess.run(self.init)

            #writer.add_graph(sess.graph)

            all_mean_rewards = []
            all_std_rewards = []
            all_std_performances = []
            all_std_performances.append(0)
            performance = []
            performance.append(0)
            loss = [0.99]

            self.create_experience_replay_buffer(sess)

            if(training_mode):

                print('------ Training Mode underway-----')

                for i in range(num_episodes):

                    state = self.env.reset()

                    pre_proc_state = self.frame_preprocess(state)

                    self.add_frame_train(pre_proc_state)

                    state = self.compile_frames_train()


                    for j in range(self.max_time_steps):

                        # Check state expanding
                        temp_state = np.reshape(state,(1,-1))

                        action, _ = self.q_prediction(sess,temp_state)

                        rand_number = np.random.rand(1)

                        # Epsilon Greedy strategy exploration (Exploitation vs exploration)
                        if(self.epsilon>rand_number):
                            action = self.env.action_space.sample()

                        next_state,reward,done,_ = self.env.step(action)

                        limited_reward = limit_return(reward)

                        pre_proc_state = self.frame_preprocess(next_state)

                        self.add_frame_train(pre_proc_state)

                        next_state = self.compile_frames_train()

                        if(done):
                            next_state = None
                        else:
                            pass

                        self.add_episode((state,action,limited_reward,next_state))

                        batch_data = self.sample_episodes(batch_size)

                        batch_states = list(map(lambda x:x[0], batch_data))
                        batch_actions = list(map(lambda x:x[1], batch_data))
                        batch_rewards = list(map(lambda x:x[2], batch_data))
                        next_states = list(map(lambda x:x[3], batch_data))

                        #current_action,Q_current = self.q_prediction(sess,batch_states)

                        next_actions, _ = self.q_prediction_target(sess,next_states)

                        # current_actions_hot = one_hot_convert(batch_actions, self.num_actions)
                        # next_actions_hot = one_hot_convert(next_actions, self.num_actions)

                        done_flags = done_state_check(next_states,batch_size)

                        _, agent_loss,_, q_vals = self.train(sess,batch_states,
                                                             next_states,
                                                             batch_rewards,
                                                             done_flags)

                        #writer.add_summary(summary, i)

                        state = next_state

                        if(done):
                            break


                    if (i % self.result_display == 0):

                        rewards, experiment_lengths, scores = self.action(sess, 100)

                        mean_rewards,std_rewards = results(rewards)
                        mean_experiment_lengths,std_experiment_lengths = results(experiment_lengths)

                        print('Episode: ' + str(int(i)) + ', Mean Reward:' + str(mean_rewards) + ', Mean Epi Length:' + str(mean_experiment_lengths) + ', Loss:' + str(agent_loss))

    # # Function for preprocessing input screen for games
    def frame_preprocess(self,input):

        #Edit screen size/re-size

        #Edit frame colour


        # Edit data type


        return input

