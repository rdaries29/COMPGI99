# Course: COMPGI99 - Thesis
# Institution: University College London
# Developer: Russel Daries (16079408)

import gym
import roboschool

class robot_manager:

    def __init__(self,env_name,display):

        self.env_name = env_name
        self.display = display

        self.env = gym.make(env_name)
        self.reset()

    def reset(self):

        observation = self.env.reset()
        return observation

    def step(self,action):

        self._update_display()
        observation, reward, done, info = self.env.step(action)

        return observation,reward,done,info

    def _update_display(self):

        if self.display:
            self.env.render()
