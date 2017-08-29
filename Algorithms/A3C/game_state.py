# Course: COMPGI99 - Thesis
# Institution: University College London
# Developer: Russel Daries (16079408)

import tensorflow as tf
import gym
import roboschool
import numpy as np
from collections import deque
from gym import wrappers

from misc_definitions import *

class game_state(object):

    def __init__(self,game_name,rand_seed,construct_agent,thread_index,all_paths,training_mode,count_rendering):

        self.game_name = game_name
        self.frame_stack_size = 4
        self.frame_buffer_train = deque(maxlen = self.frame_stack_size)
        self.frame_buffer_test = deque(maxlen= self.frame_stack_size)
        self.video_path = all_paths[4]
        self.render_agent = construct_agent
        self.seeding = rand_seed

        games_dict = {'hopper': env_hop, 'walker': env_walk, 'humanoid': env_human, 'humanoidflag': env_human_flag}

        self.environment_name = games_dict[self.game_name]

        self.env = gym.make(self.environment_name)

        if(count_rendering):
            if(thread_index==-1):
                self.env = wrappers.Monitor(self.env, self.video_path, video_callable=None, force=True)
            else:
                pass
        if(self.render_agent):
            self.env.render()
        else:
            pass

        state = self.env.reset()

        pre_proc_state = self.frame_preprocess(state)

        self.add_frame_train(pre_proc_state, 4)

        self.s_t = self.compile_frames_train()

    def update(self):
        self.s_t = self.s_t1

    def frame_preprocess(self,state):
        return state

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