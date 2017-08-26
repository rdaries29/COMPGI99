# Course: COMPGI99 - Thesis
# Institution: University College London
# Developer: Russel Daries (16079408)

# Define class for Agent
from collections import deque
import numpy as np
import tensorflow as tf
import random
import pandas as pd
import gym

from gym import wrappers
from memory_profiler import profile
from ddqn_common_methods import *

class Agent:

    def __init__(self, enviroment, learning_rate,buffer_size,discount,all_paths,algorithm,training_mode,game_name):

        self.algorithm_name = algorithm
        self.game_name = game_name
        self.frame_stack_size = 4
        self.experience_buffer_size = buffer_size
        self.discount = discount
        self.final_epsilon = 0.1
        self.initial_epsilon = 1.0
        self.epsilon = self.initial_epsilon
        self.select = 'RMS'
        self.result_display = 2000
        self.reuse_flag = False

        self.model_path = all_paths[0]
        self.variable_path = all_paths[1]
        self.plot_path = all_paths[2]
        self.table_path = all_paths[3]
        self.video_path = all_paths[4]

        self.experience_buffer_episodes = deque(maxlen=self.experience_buffer_size)
        self.episode_lens = np.array([])
        self.target_network_up_count = 1900

        self.frame_buffer_train = deque(maxlen = self.frame_stack_size)
        self.frame_buffer_test = deque(maxlen= self.frame_stack_size)

        self.save_best_model = False
        self.best = 120
        self.seeding = 200
        self.learning_rate = learning_rate
        self.save_model_step = 1700

        self.env = enviroment
        self.max_time_steps = enviroment._max_episode_steps
        self.state_dims = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.steps = 0
        self.alpha = 0.06
        self.discrete_levels = 5
        self.build_dqn()
        self.training_mode = training_mode

    def action_during_training(self,sess,experiments,construct_agent,count_rendering):

        all_rewards = []
        episode_steps = []
        all_scores = []

        if(count_rendering==0):
            pass
            # self.env = wrappers.Monitor(self.env, self.video_path, video_callable=None,force=True)
        else:
            pass

        for episode in range(experiments):

            state = self.env.reset()

            pre_proc_state = self.frame_preprocess(state)

            self.add_frame_test(pre_proc_state,4)

            state = self.compile_frame_test()

            episode_reward = 0
            score = 0

            for time_step in range(self.max_time_steps):

                if(construct_agent):

                    self.env.render()

                # Re-shape the state
                temp_state = np.reshape(state,(1,-1))

                action,_ = self.q_prediction(sess,temp_state)
                action = np.squeeze(action)
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

                    all_rewards.append(episode_reward)
                    episode_steps.append([time_step+1])
                    all_scores.append(score)
                    break

        return np.mean(all_rewards),np.std(all_rewards),np.mean(episode_steps),np.std(episode_steps),np.mean(all_scores),np.std(all_scores)

    @profile
    def action(self,sess,experiments,construct_agent,record_videos):

        all_rewards = []
        episode_steps = []
        all_scores = []

        if(record_videos):
            self.env = wrappers.Monitor(self.env, self.video_path, video_callable=None,force=True)
        else:
            pass

        for episode in range(experiments):

            state = self.env.reset()

            pre_proc_state = self.frame_preprocess(state)

            self.add_frame_test(pre_proc_state,4)

            state = self.compile_frame_test()

            episode_reward = 0
            score = 0

            for time_step in range(self.max_time_steps):

                if(construct_agent):

                    self.env.render()

                # Re-shape the state
                temp_state = np.reshape(state,(1,-1))

                action,_ = self.q_prediction(sess,temp_state)
                action = np.squeeze(action)
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

                    all_rewards.append(episode_reward)
                    episode_steps.append([time_step+1])
                    all_scores.append(score)
                    break

        return np.mean(all_rewards),np.std(all_rewards),np.mean(episode_steps),np.std(episode_steps),np.mean(all_scores),np.std(all_scores)



    def create_experience_replay_buffer(self,sess):

        for iteration in range(self.experience_buffer_size):

            state = self.env.reset()

            pre_proc_state = self.frame_preprocess(state)

            self.add_frame_train(pre_proc_state,4)

            state = self.compile_frames_train()

            for time_step in range(self.max_time_steps):

                # Check state dimensions for this input
                temp_state = state
                temp_state = np.reshape(temp_state,(1,-1))

                action, _ = self.q_prediction(sess,temp_state)
                action = np.squeeze(action)

                next_state,reward,done,_ = self.env.step(action)

                limited_reward = limit_return(reward)

                pre_proc_frame = self.frame_preprocess(next_state)

                self.add_frame_train(pre_proc_frame)

                next_state = self.compile_frames_train()

                if(done):
                    next_state = np.zeros((self.frame_stack_size,self.state_dims))
                else:
                    pass

                self.add_episode((state,action,limited_reward,next_state,done))

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
        with tf.variable_scope('Q_net'):
            self.q_network_current = self.create_q_network(self.num_actions,self.state_arrays)
            self.q_network_current_next = self.create_q_network(self.num_actions,self.next_state_arrays)

        with tf.variable_scope('T_net'):
            self.t_network_current = tf.stop_gradient(self.create_t_network(self.num_actions,self.next_state_arrays))

        q_prediction_matrix = tf.reshape(self.q_network_current,shape=[-1,self.num_actions,self.discrete_levels])

        q_prediction_matrix_next = tf.reshape(self.q_network_current_next,shape=[-1,self.num_actions,self.discrete_levels])

        t_prediction_matrix = tf.reshape(self.t_network_current,shape=[-1,self.num_actions,self.discrete_levels])

        q_prediction_actions_next,t_prediction_value = action_value_selection(q_prediction_matrix_next,t_prediction_matrix,ddqn_prediction=True,discrete_level=self.discrete_levels)

        q_prediction_actions,q_prediction_value = action_value_selection(q_prediction_matrix,q_prediction_matrix,ddqn_prediction=False,discrete_level=self.discrete_levels)

        self.q_predict = q_prediction_actions

        self.current_av = q_prediction_value

        t_prediction_actions = q_prediction_actions_next

        self.t_predict = t_prediction_actions

        self.max_q_new = t_prediction_value * self.done_holder

        self.new_av = self.rewards_holder + (self.discount * self.max_q_new)

        #temporal difference
        self.td = (self.new_av-self.current_av)

        # Be sure to clip the gradients so they don't vanish

        self.loss = tf.reduce_mean(tf.square(self.td))

        global_step = tf.Variable(0, trainable=False)
        clip_gradients = 2.0
        gradient_noise_scale = None
        learning_rate_decay = tf.train.exponential_decay(self.learning_rate, global_step, 10000, 0.99, staircase=True)

        if(self.select=='RMS'):
            self.optimize = tf.contrib.layers.optimize_loss(
                loss=self.loss,
                global_step=global_step,
                learning_rate=learning_rate_decay,
                optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate_decay),
                clip_gradients=clip_gradients,
                gradient_noise_scale=gradient_noise_scale
            )
        elif(self.select=='ADAM'):
            self.optimize = tf.train.AdamOptimizer(learning_rate=learning_rate_decay).minimize(self.loss)
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
            q_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Q_net')
            t_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='T_net')

            for v_source, v_target in zip(q_variables, t_variables):
                update_op = v_target.assign_sub(self.alpha* (v_target - v_source))
                self.target_network_up.append(update_op)

            self.target_network_up = tf.group(*self.target_network_up)

    def update_target_network(self,sess):

        return sess.run([self.target_network_up])

    # Q Network function creation
    def create_q_network(self,num_actions,state_dims):

        if self.reuse_flag==False:
            reuse_check = None
            self.reuse_flag=True
        else:
            reuse_check = True

        input_layer = tf.reshape(state_dims,[-1,self.state_dims,self.frame_stack_size,1])

        # 3 Convolutional Layers as specified in Mnih DQN paper
        # 32 20x20 feature map
        conv_layer_1 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=input_layer,kernel_size=[4,2],padding='valid',filters=32,strides=(1,1),activation=tf.nn.relu,reuse=reuse_check,name='layer1'))

        # 64 9x9 feature map
        conv_layer_2 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=conv_layer_1,kernel_size=[3,2],padding='valid',filters=64,strides=(1,1),activation=tf.nn.relu,reuse=reuse_check,name='layer2'))

        # 64 7x7 feature map
        conv_layer_3 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=conv_layer_2, kernel_size=2, padding='valid', filters=64, strides=(1,1),activation=tf.nn.relu,reuse=reuse_check,name='layer3'))

        conv_2d_flatten = tf.contrib.layers.flatten(conv_layer_3)

        fully_connected = tf.layers.batch_normalization(tf.layers.dense(inputs=conv_2d_flatten,units=512,activation=tf.nn.relu,reuse=reuse_check,name='fulllayer1'))

        Q = tf.layers.batch_normalization(tf.layers.dense(inputs=fully_connected,units=num_actions*self.discrete_levels,reuse=reuse_check,name='fulllayer2'))

        return Q

    # T Network function creation
    def create_t_network(self,num_actions,state_dims):

        input_layer = tf.reshape(state_dims,[-1,self.state_dims,self.frame_stack_size,1])

        # 3 Convolutional Layers as specified in Mnih DQN paper
        # 32 20x20 feature map
        conv_layer_1 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=input_layer,kernel_size=[4,2],padding='valid',filters=32,strides=(1,1),activation=tf.nn.relu))

        # 64 9x9 feature map
        conv_layer_2 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=conv_layer_1,kernel_size=[3,2],padding='valid',filters=64,strides=(1,1),activation=tf.nn.relu))

        # 64 7x7 feature map
        conv_layer_3 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=conv_layer_2, kernel_size=2, padding='valid', filters=64, strides=(1,1),activation=tf.nn.relu))

        conv_2d_flatten = tf.contrib.layers.flatten(conv_layer_3)

        fully_connected = tf.layers.batch_normalization(tf.layers.dense(inputs=conv_2d_flatten,units=512,activation=tf.nn.relu))

        Q = tf.layers.batch_normalization(tf.layers.dense(inputs=fully_connected,units=num_actions*self.discrete_levels))

        return Q

    # Function to train network
    def train(self,sess,current_states,next_states,rewards,done_flag):

        return sess.run([self.optimize,self.loss,self.summary_merged,self.q_network_current,self.q_network_current_next],
                        feed_dict = {self.state_arrays: current_states,
                                     self.next_state_arrays: next_states,
                                     self.rewards_holder: rewards,
                                     self.done_holder: done_flag})

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

    def replay(self,epochs,batch_size,training_mode=False,construct_agent=False,record_videos=False):

        with tf.Session() as sess:

            sess.run(self.init)

            iterations = round(self.experience_buffer_size/batch_size)
            count_rendering = 0
            low_score = -50
            epoch_rewards_curve = []
            epoch_episode_length_curve = []
            epoch_scores_curve = []
            epoch_loss_curve_training = [0.99]

            if training_mode:

                self.create_experience_replay_buffer(sess)

                print('------ Training mode underway-----')

                for epoch in range(epochs):

                    for i in range(iterations):

                        state = self.env.reset()

                        pre_proc_state = self.frame_preprocess(state)

                        self.add_frame_train(pre_proc_state,4)

                        state = self.compile_frames_train()


                        for j in range(self.max_time_steps):

                            # Check state expanding
                            temp_state = state
                            temp_state = np.reshape(temp_state,(1,-1))

                            action, _ = self.q_prediction(sess,temp_state)
                            action = np.squeeze(action)

                            rand_number = np.random.rand(1)

                            # Epsilon Greedy strategy exploration (Exploitation vs exploration)
                            if(self.epsilon>rand_number):
                                action = self.env.action_space.sample()

                            if(self.epsilon > self.final_epsilon):
                                self.epsilon -= (self.initial_epsilon-self.final_epsilon)/self.experience_buffer_size
                            else:
                                self.epsilon = self.final_epsilon

                            next_state,reward,done,_ = self.env.step(action)

                            limited_reward = limit_return(reward)

                            pre_proc_state = self.frame_preprocess(next_state)

                            self.add_frame_train(pre_proc_state)

                            next_state = self.compile_frames_train()

                            if(done):
                                next_state = np.zeros((self.frame_stack_size,self.state_dims))
                            else:
                                pass

                            self.add_episode((state,action,limited_reward,next_state,done))

                            batch_data = self.sample_episodes(batch_size)

                            batch_states = list(map(lambda x:x[0], batch_data))
                            batch_actions = list(map(lambda x:x[1], batch_data))
                            batch_rewards = list(map(lambda x:x[2], batch_data))
                            next_states = list(map(lambda x:x[3], batch_data))

                            done_flags = list(map(lambda x:x[4], batch_data))
                            done_flags_values = (~np.array(done_flags))*1

                            temp_batch_states = np.reshape(batch_states,(batch_size,-1))
                            temp_next_states = np.reshape(next_states,(batch_size,-1))

                            next_actions, _ = self.q_prediction_target(sess,temp_next_states)

                            _, agent_loss,_, q_vals,q_vals_next = self.train(sess,temp_batch_states,
                                                                 temp_next_states,
                                                                 batch_rewards,
                                                                 done_flags_values)

                            state = next_state

                            if(done):
                                break

                        if (i % self.result_display == 0):

                            mean_reward,std_reward, mean_experiment_length,std_experiment_length, mean_score,std_score = self.action_during_training(sess, 10,construct_agent,count_rendering)
                            count_rendering=1
                            print('Epoch: ' + str(epoch) + ', Iteration Reward:' + str(mean_reward)+ ', Std Reward:' + str(std_reward)+ ', Mean Epi Length:' + str(mean_experiment_length)+ ', Std Epi Length:' + str(std_experiment_length)+', Loss:'+str(agent_loss))

                        if(i % self.save_model_step ==0 and mean_score>low_score):
                            with tf.device("/cpu:0"):
                                self.saver.save(sess,self.model_path+'model'+str(i)+'.ckpt')

                        if (i % self.target_network_up_count == 0 and i is not 0):
                            self.update_target_network(sess)
                            print('---Target Network Updated---')

                    epoch_reward, epoch_std_reward, epoch_experiment_length, epoch_std_experiment_length, epoch_score, epoch_std_score = self.action_during_training(sess, 10, construct_agent, count_rendering)
                    count_rendering = 1
                    print('Epoch: ' + str(epoch) + ', Reward:' + str(epoch_reward) + ', Std Reward:' + str(
                        epoch_std_reward) + ', Epi Length:' + str(epoch_experiment_length) + ', Std Epi Length:' + str(
                        epoch_std_experiment_length)+', Score: '+str(epoch_score)+',Std Score:'+str(epoch_std_score)+', Loss:'+str(agent_loss))

                    epoch_rewards_curve.append(epoch_reward)
                    epoch_episode_length_curve.append(epoch_experiment_length)
                    epoch_scores_curve.append(epoch_score)
                    epoch_loss_curve_training.append(agent_loss)

                    np.savez(self.variable_path+'/saved_curves.npz', epoch_rewards_saved=epoch_rewards_curve, epoch_episode_length_saved=epoch_episode_length_curve, epoch_scores_saved=epoch_scores_curve,
                             epoch_loss_curve_saved=epoch_loss_curve_training)

                with tf.device("/cpu:0"):
                    save_path = self.saver.save(sess,self.model_path+'model.ckpt')
                    print('Model saved to: ',save_path)

                plot_data(metric=epoch_rewards_curve, xlabel='Epochs',ylabel='Discounted Return',colour='b',filename=self.plot_path+'rewards_'+self.algorithm_name+'_'+self.game_name)
                plot_data(metric=epoch_episode_length_curve, xlabel='Epochs',ylabel='Episode Length', colour='g', filename=self.plot_path+'episodes_'+self.algorithm_name+'_'+self.game_name)
                plot_data(metric=epoch_scores_curve, xlabel='Epochs',ylabel='Undiscounted Return', colour='m', filename=self.plot_path+'scores_'+self.algorithm_name+'_'+self.game_name)
                plot_data(metric=epoch_loss_curve_training,xlabel='Epochs',ylabel='Loss', colour='r', filename=self.plot_path+'loss_'+self.algorithm_name+'_'+self.game_name)
                print('---Results Plotted---')
            else:

                print('------ Testing Mode underway-----')

                self.saver.restore(sess,self.model_path+'model.ckpt')
                print('Retrieving model from:'+self.model_path+'model.ckpt')
                num_parameters = self.count_parameters(sess)
                mean_reward, std_reward, mean_experiment_length, std_experiment_length, mean_score, std_score = self.action(sess, 100,
                                                                                                            construct_agent,record_videos)
                print('Mean Reward:' + str(mean_reward) + ', Std Reward:' + str(
                    std_reward) + ', Mean Epi Length:' + str(mean_experiment_length) + ', Std Epi Length:' + str(
                    std_experiment_length))

                test_performance = {'reward':[mean_reward],'std_reward':[std_reward],'epi_length':[mean_experiment_length],'std_spi_length':[std_experiment_length],'mean_score':[mean_score],'std_score':[std_score]}
                df = pd.DataFrame(test_performance)
                writer = pd.ExcelWriter(self.table_path + 'test_result.xlsx', engine='xlsxwriter')
                df.to_excel(writer, sheet_name='Sheet1')
                writer.save()

# # Function for preprocessing input screen for games
    def frame_preprocess(self,input):

        return input

    def count_parameters(self,sess):
        """Returns the number of parameters of a computational graph."""
        parameter_trace_network = dict()
        variables_names     = [v.name for v in tf.trainable_variables()]
        values              = sess.run(variables_names)
        n_params            = 0

        for k, v in zip(variables_names, values):
            print('-----------------------------------------')
            print('Variable:\t{:20s} \t{:20} \tShape: {:}\t{:20} parameters'.format(k,str(v.dtype), v.shape, v.size))
            parameter_trace_network[k]= [str(v.dtype),v.shape,v.size]

            n_params += v.size

        parameter_trace_network['number_params']= n_params

        print('\n\n-----------------------------\nTotal # parameters:\t{}'.format(n_params))

        df = pd.DataFrame(parameter_trace_network)
        writer = pd.ExcelWriter(self.table_path + 'parameter_trace_network.xlsx', engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Sheet1')
        writer.save()
        return n_params