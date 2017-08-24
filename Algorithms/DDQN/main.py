# Course: COMPGI99 - Thesis
# Institution: University College London
# Developer: Russel Daries (16079408)

import gym
import roboschool
import sys
sys.path.insert(0,'../../Common')

from ddqn_common_methods import *
from class_definitions import *
from misc_definitions import *

learning_rate = 0.0001
epochs = 30
batch_size = 256
experience_buffer_size = 1000000
construct_agent = False
discount = 0.99
save_path_var = True
training_mode = True
record_test_videos = False

# Problem Number
algorithm = 'DDQN'

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

if(save_path_var):
    # All paths save directory
    all_save_path = '../../Models/'+algorithm+'/'+game_name+'/variables'+'/saved_paths.npz'
    np.savez(all_save_path,model_path=model_path,variable_path=variable_path,plot_path=plot_path,table_path=table_path,video_path=video_path,tf_path=tf_path)
    print('Variables saved to: '+ all_save_path)

def main(agent):

    env = gym.make(agent)
    print('-------Creating Agent-------')

    agent = Agent(env,learning_rate,experience_buffer_size,discount,all_paths,algorithm,training_mode)

    if training_mode:
        print('-------Training Model-------')
    else:
        print('-------Testing Model-------')

    agent.replay(epochs,batch_size,training_mode,construct_agent,record_test_videos)

if __name__ =='__main__':
    print(games_dict[game_name])
    main(games_dict[game_name])