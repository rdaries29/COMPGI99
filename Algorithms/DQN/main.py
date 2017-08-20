# Course: COMPGI99 - Thesis
# Institution: University College London
# Developer: Russel Daries (16079408)

import sys
sys.path.insert(0, '/Users/russeldaries/Documents/University_College_London/Computer_Science/Courses/Dissertation/UCABRSD/Code/Source_Code/COMPGI99/Common')

from common_imports import *

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

# Problem Number
algorithm = 'DQN'

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

if(save_path_var):
    # All paths save directory
    all_save_path = '../../Models/'+algorithm+'/variables'+'/saved_paths.npz'
    np.savez(all_save_path,model_path=model_path,variable_path=variable_path,plot_path=plot_path,table_path=table_path,tf_path=tf_path)
    print('Variables saved to: '+ all_save_path)

def main(agent):

    env = gym.make(agent)

    print('-------Creating Agent-------')


    agent = Agent(env,learning_rate)

    print('-------Training Model-------')

    agent.replay(num_episodes,batch_size,training_mode)

if __name__ =='__main__':
    main(env_hop)