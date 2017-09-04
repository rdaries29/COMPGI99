# Course: COMPGI99 - Thesis
# Institution: University College London
# Developer: Russel Daries (16079408)

# Import required libraries and packages
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Function for limiting returns from environment
def limit_return(reward):
    if reward > 1:
        return 1
    elif reward < -1:
        return -1
    elif reward < 1 and reward > -1:
        return reward
    else:
        return 0

# Function to return mean and standard deviation of input vector
def results(results_vector):

    mean_result = np.mean(results_vector)
    std_result = np.std(results_vector)

    return mean_result, std_result

# Convert an input vector to one-hot encoding
def one_hot_convert(vector,num_actions):

    vector = np.array(vector)

    one_hot_vec = np.zeros(len(vector),num_actions)

    one_hot_vec[np.arange(len(vector)),vector] = 1

    return one_hot_vec

# Function to check if terminal state reached.
def done_state_check(next_states,batch_size):

    done_flags_vec = np.ones(batch_size)
    count = 0

    for next_state in next_states:

        if(next_state==None):
            done_flags_vec[count] = 0
        else:
            pass

        count += 1

    return done_flags_vec

# Plotting function for data metrics
def plot_data(metric, xlabel, ylabel,colour,filename):

    plt.plot(metric, colour, lw=2)
    plt.grid(True)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.savefig(filename+ '_metrics.pdf', bbox_inches='tight', format='pdf', dpi=50)
    plt.close()

def action_value_selection(action_selection_matrix,value_selection_matrix,ddqn_prediction,discrete_level):
    '''Function to map output node index (output_nodes_vec) to quantized action vector for continious control '''
    # Constructing the action vector
    maximum_indices = tf.argmax(action_selection_matrix,axis=2)
    action_matrix = quantize(maximum_indices)

    if(ddqn_prediction==False):
        # Constructing the value estimate
        maximum_values = tf.reduce_max(value_selection_matrix,axis=2)
        maximum_values = tf.reduce_sum(maximum_values,axis=1)
    else:
        maximum_indices_one_hot = tf.one_hot(maximum_indices,depth=discrete_level)
        maximum_values = tf.multiply(maximum_indices_one_hot,value_selection_matrix)
        maximum_values = tf.reduce_sum(maximum_values,axis=2)
        maximum_values = tf.reduce_sum(maximum_values,axis=1)

    return action_matrix,maximum_values

def quantize(maximum_index_vector):
    '''Apply map function for discretization of selected nodes'''
    out_tensor = tf.map_fn(lambda x: (0.5*x)-1,tf.cast(maximum_index_vector,dtype=tf.float32))

    return out_tensor