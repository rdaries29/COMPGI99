# Course: COMPGI99 - Thesis
# Institution: University College London
# Developer: Russel Daries (16079408)

# Import nesscary libraries and packages
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

# Plotting function for data metrics
def plot_data(metric, xlabel, ylabel,colour,filename):

    plt.plot(metric, colour, lw=2)
    plt.grid(True)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.savefig(filename+ '_metrics.pdf', bbox_inches='tight', format='pdf', dpi=50)
    plt.close()

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder