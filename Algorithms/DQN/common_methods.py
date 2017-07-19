# Course: COMPGI99 - Thesis
# Institution: University College London
# Developer: Russel Daries (16079408)

def limit_return(reward):
    if reward > 0:
        return 1
    elif reward < 0:
        return -1
    else:
        return 0

# Defining weight variable function
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

# Defining bias variable function
def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def results(results_vector):

    mean_result = np.mean(results_vector)
    std_result = np.std(results_vector)

    return mean_result, std_result

def one_hot_convert(vector,num_actions):

    vector = np.array(vector)

    one_hot_vec = np.zeros(len(vector),num_actions)

    one_hot_vec[np.arange(len(vector)),vector] = 1

    return one_hot_vec

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
