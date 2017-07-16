# Course: COMPGI99 - Thesis
# Institution: University College London
# Developer: Russel Daries (16079408)

# Add additional directories
import sys
# Directory for common function files
sys.path.insert(0, '../common')

import numpy as np
import pandas as pd
import tensorflow as tf
import random

from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque

# Defining weight variable function
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

# Defining bias variable function
def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

# Define class for Agent

class Agent:

    def __init__(self, enviroment, learning_rate):

        self.frame_stack_size = 4
        self.experience_buffer_size = 500
        self.discount = 0.99
        self.max_episodes = 300
        self.epsilon = 0.05
        self.select = 'RMS'
        self.result_display = 10

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
        self.num_actions = self.env.action_space.n
        self.steps = 0


    def limit_reward(self, reward):
        if reward > 0:
            return 1
        elif reward < 0:
            return -1
        else:
            return 0

    def action(self,sess,experiments,construct_agent):

        rewards = []
        experiment_lengths = []
        scores = []

        for experiment in range(experiments):

            state = self.env.reset()

            pre_proc_frame = self.frame_preprocess(state)

            self.add_frame_test(pre_proc_frame)

            state = self.compile_frame_test()

            # Check frame stacking functions

            episode_reward = 0
            score = 0

            for episode_count in range(self.max_episodes):

                if(construct_agent):

                    self.env.render()

                # Re-shape the state
                temp_state = np.expand_dims(state,axis=0)

                action,_ = self.q_prediction(sess,temp_state)

                # Step in enviroment
                next_state_test,reward,done,_ = self.env.step(action[0])

                limited_reward = self.limit_reward(reward)

                next_frame_test = self.frame_preprocess(next_state_test)

                # Add next frame to the stack

                self.add_frame_test(next_frame_test)

                state = self.compile_frame_test()

                episode_reward += (reward * self.discount ** episode_count)
                score +=reward

                if(done):

                    rewards.append(episode_reward)
                    experiment_lengths.append([i+1])
                    scores.append(score)
                    break
                #
                # if(done):
                #
                #     rewards.append(-1*self.discount**episode)

        return np.mean(rewards),np.mean(experiment_lengths),np.mean(scores),np.std(rewards),np.std(experiment_lengths),np.std(scores)


    def create_experience_replay_buffer(self,sess):

        for iteration in range(self.experience_buffer_size):

            state = self.env.reset()

            input_frame = self.frame_preprocess(state)

            self.add_frame_train(input_frame)

            state = self.compile_frame_train()

            for episode in range(self.max_episodes):

                # Check state dimensions for this input
                temp_state = np.expand_dims(state,axis=0)

                action, _ = self.q_prediction(sess,temp_state)

                next_state,reward,done,_ = self.env.step(action[0])

                reward = self.limit_reward(reward)

                pre_proc_frame = self.frame_preprocess(next_state)

                self.add_frame_train(pre_proc_frame)

                next_state = self.compile_frames_train()

                self.add_episode((state,action[0],reward,next_state))

                state = next_state

                buffer_length = len(self.experience_buffer_episodes)


                if (self.experience_buffer_size == buffer_length):
                    return

                if (done):
                    break


    def build_dqn(self):

        tf.reset_default_graph()
        tf.set_random_seed(self.seeding)

        self.state_arrays = tf.placeholder(shape=[None, self.state_dims], dtype=np.float32)

        # Convolutional Layer stacks





        self.predict = tf.argmax(self.Q, 1)

        self.Q_next = tf.placeholder(shape=[None, self.num_actions], dtype=tf.float32)

        # Be sure to clip the gradients so they don't vanish

        self.loss = tf.reduce_mean(tf.square(self.Q_next - self.Q))

        if(self.select=='RMS'):
            self.optimize = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,momentum=0.9).minimize(self.loss)
        elif(self.select=='ADAM'):
            self.optimize = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        else:
            pass

        # Record loss in tensorflow plots to be used for tensorboard
        tf.summary.scalar("loss", self.loss)
        self.summary_merged = tf.summary.merge_all()

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    # Function to train network
    def train(self,sess,x_input,y_output):

        return sess.run([self.optimize,self.loss,self.summary_merged],
                        feed_dict = {self.state_arrays:x_input, self.Q_next:y_output})

    # Function call to predict next Q (return)
    def q_prediction(self,sess,x_input):

        return sess.run([self.predict,self.Q], feed_dict = {self.state_arrays:x_input})

    def memory_replay(self):

        def __init__(self):



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

    def replay(self,learning_rate,num_episodes,batch,training_mode=False):

        with tf.Session() as sess:

            sess.run(self.init)

            writer.add_graph(sess.graph)

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

                    pre_proc_frame = self.frame_preprocess(state)

                    self.add_frame_train(pre_proc_frame)

                    state = self.compile_frames_train()


                    for j in range(self.max_episodes):

                        # Check state expanding
                        temp_state = np.expand_dims(state,axis=0)

                        action,Q = self.q_prediction(sess,temp_state)

                        rand_number = np.random.rand(1)

                        if(self.epsilon>rand_number):
                            action[0] = self.env.action_space.sample()

                        next_state,reward,done,_ = self.env.step(action[0])

                        reward = self.limit_reward(reward)



                        if(done):

                        else:


                        self.memory.add_episode((state,action[0],reward,next_state))

                        batch_data = self.memory.sample_episodes(batch)

                        batch_rewards = list(map(lambda x:x[2],batch_data))
                        batch_actions = list(map(lambda x:x[1],batch_data))
                        batch_states = list(map(lambda x:x[0],batch_data))

                        Q_max,Q = self.q_prediction(sess,batch_states)

                        resultant_states  = list(map(lambda x:x[3],batch_data))
                        Q_max_next,Q_next = self.q_prediction(sess,resultant_states)

                        Q_target = np.copy(Q)

                        Q_update = batch_rewards + (self.discount* np.amax(Q_next,1))

                        Q_target_add = list(map(lambda a,q_old,q_new: np.array(
                                [q_new if a == 0 else q_old[0], q_new if a == 1 else q_old[1]]),
                                            batch_actions, Q_target, Q_update))


                        _ , agent_loss, summary = self.agent_brain.train(sess, x_input=resultant_states,y_output=Q_target_add)

                        writer.add_summary(summary, i)

                        state = next_state

                        if(done):
                            break


                    if (i % self.result_display == 0):
                        agent_rewards, agent_performances, std_rewards, std_performances = self.action(sess, 100)

                        print('Episode: ' + str(int(i)) + ', Performance:' + str(agent_performances) + ', Loss:' + str(agent_loss))

    # Function for preprocessing input screen for games
    def frame_preprocess(self,input):

        #Edit screen size/re-size

        #Edit frame colour


        # Edit data type



        return frame_conversion






    def











