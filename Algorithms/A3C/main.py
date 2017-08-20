# Course: COMPGI99 - Thesis
# Institution: University College London
# Developer: Russel Daries (16079408)

import tensorflow as tf
import threading
import multiprocessing

from time import time
from time import sleep
from common_methods import *

USE_GPU = False

learning_rate = [0.001]
epochs = 3
batch_size = 32
num_episodes = 100
seeding = 200
experience_buffer_size = 100
construct_agent = False
discount = 0.99
save_path_var = False
training_mode = False
load_model = False

# Problem Number
algorithm = 'A3C'

# Problem Model Path
model_path = '../../Models/'+algorithm
# Problem Variable Path
variable_path = '../../Models/'+algorithm+'/variables'
# Problem Plot Path
plot_path = '../../Results/'+algorithm+'/plots'
# Problem Table Path
table_path = '../../Results/'+algorithm+'/tables'
# Tensorflow Summary Path
tf_path = model_path
# Gif frame path
frames_path = '../../Results/'+algorithm+'/frames'


if(save_path_var):
    # All paths save directory
    all_save_path = '../../Models/'+algorithm+'/variables'+'/saved_paths.npz'
    np.savez(all_save_path,model_path=model_path,variable_path=variable_path,plot_path=plot_path,table_path=table_path,tf_path=tf_path,frames_path=frames_path)
    print('Variables saved to: '+ all_save_path)


with tf.device(device):

    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)

    master_network = lstm_ac_network()

    if USE_GPU:
         pass
    else:
        num_workers = multiprocessing.cpu_count()

    workers = []

    for i in range(num_workers):
        workers.append(Worker())

    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)

