# Course: COMPGI99 - Thesis
# Institution: University College London
# Developer: Russel Daries (16079408)

import scipy.misc as misc
import gym
import roboschool

class enviroment:

    def __init__(self,env_name,display):

        self.env_name = env_name
        self.display = display
        self.env = gym.make(env_name)

        self.nb_frames = config.STACKED_FRAMES
        self.frame_q = Queue(maxsize=self.nb_frames)
        self.previous_state = None
        self.current_state = None
        self.total_reward = 0

        self.reset()

    def _preprocess(image):

        image = misc.resize(image,[config.IMAGE_HEIGHT,config.IMAGE_WIDTH],'bilinear')

        return image



    def _get_current_state(self):

        if not self.frame_q.full():
            return None

        x_ = np.array(self.frame_q.queue)


        return x_

    def _update_frame_q(self,frame):

        if self.frame_q.full():
            self.frame_q.get()
        image = enviroment._preprocess(frame)
        self.frame_q.put(image)


    def _get_num_actions(self):
        return self.game.env.action_space.n

    def reset(self):

        self.total_reward = 0
        self.frame_q.queue.clear()
        self._update_frame_q(self.env.reset())
        self.previous_state = None
        self.current_state = None

    def step(self,action):

        observation, reward, done, _ = self.env.step(action)

        self.total_reward+= reward
        self._update_frame_q(observation)

        self.previous_state = self.current_state
        self.current_state = self._get_current_state()

        return observation, reward, done


