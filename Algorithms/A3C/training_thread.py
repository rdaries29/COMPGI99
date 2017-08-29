# Course: COMPGI99 - Thesis
# Institution: University College London
# Developer: Russel Daries (16079408)

import numpy as np
import tensorflow as tf
import gym
import roboschool
import random
import time

from common_methods import *
from misc_definitions import *
from ac_network import lstm_ac_network
from game_state import game_state

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000

class worker_training_thread(object):

    def __init__(self,thread_index,master_network,initial_learning_rate,grad_applier,max_time_step_env,action_size,observation_size,game_name,device):

        self.thread_index = thread_index
        self.learning_rate = tf.placeholder(dtype = tf.float32)
        self.initial_learning_rate = initial_learning_rate
        self.max_time_steps = max_time_step_env
        self.state_dims = observation_size
        self.num_actions = action_size
        self.entropy_beta = 0.01
        self.discount = 0.99
        self.game_name = game_name
        self.rand_seeding = 200
        # if(self.thread_index==0):
        #     self.construct_agent = True
        # else:
        #     self.construct_agent = False
        self.construct_agent = False
        self.max_global_time_step = max_time_step_env

        self.local_network = lstm_ac_network(action_size,observation_size,thread_index,device)
        self.local_network.prepare_loss(self.entropy_beta)

        with tf.device(device):
            var_refs = [v._ref() for v in self.local_network.get_vars()]
            self.gradients = tf.gradients(self.local_network.total_loss, var_refs)

        self.apply_gradients = grad_applier.apply_gradients(master_network.get_vars(),self.gradients)

        self.sync = self.local_network.sync_from(master_network)
        self.game_state = game_state(self.game_name,self.rand_seeding,self.construct_agent)
        self.local_t = 0
        self.initial_learning_rate = initial_learning_rate

        self.episode_reward_undiscounted = 0
        self.episode_reward_discounted = 0
        self.prev_local_t = 0

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def choose_action(self, pi_values):
        return np.random.choice(range(len(pi_values)), p=pi_values)

    def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
        summary_str = sess.run(summary_op, feed_dict={score_input: score})
        summary_writer.add_summary(summary_str, global_t)
        summary_writer.flush()

    def set_start_time(self, start_time):
        self.start_time = start_time

    def process(self, sess, global_t, summary_writer, summary_op, score_input):

        states = []
        actions = []
        rewards_undiscounted = []
        rewards_discounted = []
        values = []
        episode_steps = []

        done = False

        # copy weights from shared to local
        sess.run(self.sync)

        start_local_t = self.local_t
        start_lstm_state = self.local_network.lstm_state_out

        # t_max times loop
        for i in range(self.max_time_steps):
            temp_state = np.reshape(self.game_state.s_t,(1,-1))
            pi_, value_ = self.local_network.predict_policy_and_value(sess, temp_state)
            action = self.choose_action(pi_)
            action_vector = np.zeros([self.num_actions])
            action_vector[action] = 1

            states.append(np.reshape(self.game_state.s_t,(1,-1)))
            actions.append(action_vector)
            values.append(value_)

            if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
                print("pi={}".format(pi_))
                print(" V={}".format(value_))


            # process game
            next_state,reward,done,_ = self.game_state.env.step(action_vector)

            next_frame = self.game_state.frame_preprocess(next_state)

            self.game_state.add_frame_train(next_frame)

            self.game_state.s_t1 = self.game_state.compile_frames_train()

            limited_reward = limit_return(reward)

            terminal = done

            self.episode_reward_undiscounted += limited_reward
            self.episode_reward_discounted = (limited_reward * self.discount ** i)

            # clip reward
            rewards_undiscounted.append(self.episode_reward_undiscounted)
            rewards_discounted.append(self.episode_reward_discounted)
            episode_steps.append(i+1)

            self.local_t += 1

            # s_t1 -> s_t
            self.game_state.update()

            if terminal:
                terminal_end = True
                print("score_undiscounted={}".format(self.episode_reward_undiscounted))
                print("score_discounted={}".format(self.episode_reward_discounted))
                print("episode_length={}".format(i+1))

                self._record_score(sess, summary_writer, summary_op, score_input,
                                   self.episode_reward_undiscounted, global_t)

                self.episode_reward_undiscounted = 0
                self.epsiode_reward_discounted = 0
                self.game_state.env.reset()
                self.local_network.reset_state()
                break

        R = 0.0
        if not terminal_end:
            R = self.local_network.run_value(sess, np.reshape(self.game_state.s_t,(1,-1)))

        actions.reverse()
        states.reverse()
        rewards_undiscounted.reverse()
        rewards_discounted.reverse()
        values.reverse()

        batch_si = []
        batch_a = []
        batch_td = []
        batch_R = []

        # compute and accmulate gradients
        for (ai, ri, si, Vi) in zip(actions, rewards_undiscounted, states, values):
            R = ri + (self.discount * R)
            td = R - Vi
            a = ai
            # a[ai] = 1
            si = np.squeeze(si,axis=0)

            batch_si.append(si)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)

        cur_learning_rate = self._anneal_learning_rate(global_t)

        batch_si.reverse()
        batch_a.reverse()
        batch_td.reverse()
        batch_R.reverse()

        print("------Applying gradients------")
        sess.run(self.apply_gradients,
                 feed_dict={
                     self.local_network.state_arrays: batch_si,
                     self.local_network.a: batch_a,
                     self.local_network.td: batch_td,
                     self.local_network.r: batch_R,
                     self.local_network.initial_lstm_state: start_lstm_state,
                     self.local_network.step_size: [len(batch_a)],
                     self.learning_rate:cur_learning_rate})

        print("------Finishing applying gradients-----")
        if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
            self.prev_local_t += PERFORMANCE_LOG_INTERVAL
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_t / elapsed_time
            print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
                global_t, elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

        # return advanced local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t

