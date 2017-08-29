# Course: COMPGI99 - Thesis
# Institution: University College London
# Developer: Russel Daries (16079408)

import tensorflow as tf
import roboschool
import pandas as pd
import gym
import threading
import multiprocessing
import numpy as np
import signal
import random
import os
import time
import sys
sys.path.insert(0,'../../Common')

from time import sleep
from common_methods import *
from misc_definitions import *
from ac_network import lstm_ac_network
from training_thread import worker_training_thread
from rmsprop_applier import RMSPropApplier

USE_GPU = True

if USE_GPU:
    device = "/gpu:0"
else:
    device = "/cpu:0"

global_t = 0
learning_rate = 0.0001
epoch_size = 10**6
max_time_step_env = 30*epoch_size
construct_agent = False
discount = 0.99
wall_t = 0.0
rand_seeding = 200
save_path_var = True
training_mode = True
record_test_videos = False
stop_requested = False

# Problem Number
algorithm = 'A3C'

# Game
games = ['hopper','walker','humanoid','humanoidflag']
games_dict = {'hopper':env_hop,'walker':env_walk,'humanoid':env_human,'humanoidflag':env_human_flag}
game_name = games[0]

# Problem Model Path
model_path = '../../Models/'+algorithm+'/'+game_name+'/model/'
# Problem Variable Path
variable_path = '../../Models/'+algorithm+'/'+game_name+'/variables/'
# Problem Plot Path
plot_path = '../../Results/'+algorithm+'/'+game_name+'/plots/'
# Problem Table Path
table_path = '../../Results/'+algorithm+'/'+game_name+'/tables/'
# Problem Table Path
video_path = '../../Results/'+algorithm+'/'+game_name+'/videos/'

# Tensorflow Summary Path
tf_path = model_path

all_paths = [model_path,variable_path,plot_path,table_path,video_path]

env = gym.make(games_dict[game_name])
print(games_dict[game_name])
action_size = env.action_space.shape[0]
observation_size = env.observation_space.shape[0]

if(training_mode==True):

    if(save_path_var):
        # All paths save directory
        all_save_path = '../../Models/'+algorithm+'/'+game_name+'/variables'+'/saved_paths.npz'
        np.savez(all_save_path,model_path=model_path,variable_path=variable_path,plot_path=plot_path,table_path=table_path,video_path=video_path,tf_path=tf_path)
        print('Variables saved to: '+ all_save_path)
    else:
        pass

    print("------Training mode underway------")

    with tf.name_scope('Master_Network'):
        master_network = lstm_ac_network(action_size,observation_size,-1,all_paths,device)

    if USE_GPU:
        print('Using GPU')
        num_workers = multiprocessing.cpu_count()
    else:
        print('Using CPU')
        num_workers = multiprocessing.cpu_count()

    grad_applier = RMSPropApplier(learning_rate = learning_rate,decay = 0.99,momentum = 0.0,epsilon = 0.1,clip_norm = 40.0,device = device)

    worker_threads = []

    for i in range(num_workers):
        training_thread = worker_training_thread(i,master_network,learning_rate,grad_applier,max_time_step_env,action_size,observation_size,game_name,all_paths,epoch_size,training_mode,device)
        worker_threads.append(training_thread)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True))

    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    def train_function(num_workers):

        global global_t

        training_thread = worker_threads[num_workers]
        # set start_time
        start_time = time.time() - wall_t
        training_thread.set_start_time(start_time)

        while True:
            if stop_requested:
                break
            if global_t > max_time_step_env:
                if(num_workers==0):
                    np.savez(variable_path + '/saved_curves.npz', epoch_rewards_saved=reward_discounted,
                             epoch_episode_length_saved=episode_length, epoch_scores_saved=reward_undiscounted,
                             epoch_loss_curve_saved=loss)

                    plot_data(metric=reward_discounted, xlabel='Epochs', ylabel='Discounted Return', colour='b',
                              filename=plot_path + 'rewards_' + algorithm + '_' + game_name)
                    plot_data(metric=episode_length, xlabel='Epochs', ylabel='Episode Length', colour='g',
                              filename=plot_path + 'episodes_' + algorithm + '_' + game_name)
                    plot_data(metric=reward_undiscounted, xlabel='Epochs', ylabel='Undiscounted Return', colour='m',
                              filename=plot_path + 'scores_' + algorithm + '_' + game_name)
                    plot_data(metric=loss, xlabel='Epochs', ylabel='Loss', colour='r',
                              filename=plot_path + 'loss_' + algorithm + '_' + game_name)

                break

            diff_global_t,reward_discounted,reward_undiscounted,episode_length,loss= training_thread.process(sess, global_t)
            global_t += diff_global_t

    def signal_handler(signal, frame):
        global stop_requested
        print('You pressed Ctrl+C!')
        stop_requested = True


    train_threads = []
    for i in range(num_workers):
        train_threads.append(threading.Thread(target=train_function, args=(i,)))

    signal.signal(signal.SIGINT, signal_handler)

    # set start time
    start_time = time.time() - wall_t
    for t in train_threads:
        t.start()

    print('Press Ctrl+C to stop')
    # signal.pause()

    print('Now saving data. Please wait')

    for t in train_threads:
        t.join()

    with tf.device("/cpu:0"):
        save_path = saver.save(sess, model_path + 'model.ckpt')
        print('Model saved to: ', save_path)

else:

    print('------ Testing Mode underway-----')
    with tf.name_scope('Master_Network'):
        master_network = lstm_ac_network(action_size,observation_size,-1,all_paths,device)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True))
    saver = tf.train.Saver()
    saver.restore(sess, model_path + 'model.ckpt')
    print('Retrieving model from:' + model_path + 'model.ckpt')
    num_parameters = master_network.count_parameters(sess)
    mean_reward, std_reward, mean_experiment_length, std_experiment_length, mean_score, std_score = master_network.process_test(sess, 100,game_name,rand_seeding,construct_agent,all_paths,training_mode,record_test_videos)
    print('Mean Reward:' + str(mean_reward) + ', Std Reward:' + str(
        std_reward) + ', Mean Epi Length:' + str(mean_experiment_length) + ', Std Epi Length:' + str(
        std_experiment_length))

    test_performance = {'reward': [mean_reward], 'std_reward': [std_reward], 'epi_length': [mean_experiment_length],
                        'std_spi_length': [std_experiment_length], 'mean_score': [mean_score], 'std_score': [std_score]}
    df = pd.DataFrame(test_performance)
    writer = pd.ExcelWriter(table_path + 'test_result.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

