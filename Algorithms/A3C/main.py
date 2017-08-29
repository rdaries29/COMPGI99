# Course: COMPGI99 - Thesis
# Institution: University College London
# Developer: Russel Daries (16079408)

import tensorflow as tf
import roboschool
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

USE_GPU = False

if USE_GPU:
    device = "/gpu:0"
else:
    device = "/cpu:0"

global_t = 0
learning_rate = 0.00025
epochs = 30
batch_size = 256
experience_buffer_size = 1000000
construct_agent = False
discount = 0.99
wall_t = 0.0
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

if(training_mode==True):
    if(save_path_var):
        # All paths save directory
        all_save_path = '../../Models/'+algorithm+'/'+game_name+'/variables'+'/saved_paths.npz'
        np.savez(all_save_path,model_path=model_path,variable_path=variable_path,plot_path=plot_path,table_path=table_path,video_path=video_path,tf_path=tf_path)
        print('Variables saved to: '+ all_save_path)

env = gym.make(games_dict[game_name])
print(games_dict[game_name])
action_size = env.action_space.shape[0]
observation_size = env.observation_space.shape[0]
max_time_step_env = env._max_episode_steps

master_network = lstm_ac_network(action_size,observation_size,-1,device)

if USE_GPU:
     pass
else:
    num_workers = multiprocessing.cpu_count()


grad_applier = RMSPropApplier(learning_rate = learning_rate,decay = 0.99,momentum = 0.0,epsilon = 0.1,clip_norm = 40.0,device = device)

worker_threads = []

for i in range(num_workers):
    training_thread = worker_training_thread(i,master_network,learning_rate,grad_applier,max_time_step_env,action_size,observation_size,game_name,device)
    worker_threads.append(training_thread)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True))

init = tf.global_variables_initializer()
sess.run(init)

# summary for tensorboard
score_input = tf.placeholder(tf.int32)
tf.summary.scalar("score", score_input)

summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(model_path+'tensorboard', sess.graph)

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
            break

        diff_global_t = training_thread.process(sess, global_t, summary_writer,
                                                summary_op, score_input)
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
i=1
for t in train_threads:
    t.start()

print('Press Ctrl+C to stop')
signal.pause()

print('Now saving data. Please wait')

for t in train_threads:
    t.join()

saver.save(sess,model_path+'checkpoint',global_step=global_t)


# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     if load_model == True:
#         print('Loading Model...')
#         ckpt = tf.train.get_checkpoint_state(model_path)
#         saver.restore(sess, ckpt.model_checkpoint_path)
#     else:
#         sess.run(tf.global_variables_initializer())
#
#     # This is where the asynchronous magic happens.
#     # Start the "work" process for each worker in a separate threat.
#     worker_threads = []
#     for worker in workers:
#         worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
#         t = threading.Thread(target=(worker_work))
#         t.start()
#         sleep(0.5)
#         worker_threads.append(t)
#     coord.join(worker_threads)

