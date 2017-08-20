# Course: COMPGI99 - Thesis
# Institution: University College London
# Developer: Russel Daries (16079408)

import tensorflow as tf
import numpy as np

class PolicyEstimator():
  """
  Policy Function approximator. Given a observation, returns probabilities
  over all possible actions.
  Args:
    num_outputs: Size of the action space.
    reuse: If true, an existing shared network will be re-used.
    trainable: If true we add train ops to the network.
      Actor threads that don't update their local models and don't need
      train ops would set this to false.
  """

  def __init__(self, num_outputs, reuse=False, trainable=True):

    self.img_width = config.IMAGE_WIDTH
    self.img_height = config.IMAGE_HEIGHT
    self.img_stacked_frames = config.STACKED_FRAMES
    self.lstm_cell_size = 256

    self.num_outputs = num_outputs

    # Placeholders for our input
    # Our input are 4 RGB frames of shape 160, 160 each
    self.states = tf.placeholder(shape=[None, self.img_height, self.img_width, self.img_stacked_frames], dtype=tf.float32, name="x")
    # The TD target value
    self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
    # Integer id of which action was selected
    self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

    # Normalize
    X = tf.to_float(self.states) / 255.0
    batch_size = tf.shape(self.states)[0]

    # Graph shared with Value Net
    with tf.variable_scope("shared", reuse=reuse):
      fc1 = build_shared_network(X, add_summaries=(not reuse))


    with tf.variable_scope("policy_net"):
      self.logits = tf.contrib.layers.fully_connected(fc1, num_outputs, activation_fn=None)
      self.probs = tf.nn.softmax(self.logits) + 1e-8

      self.predictions = {
        "logits": self.logits,
        "probs": self.probs
      }

      # We add entropy to the loss to encourage exploration
      self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs), 1, name="entropy")
      self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")

      # Get the predictions for the chosen actions only
      gather_indices = tf.range(batch_size) * tf.shape(self.probs)[1] + self.actions
      self.picked_action_probs = tf.gather(tf.reshape(self.probs, [-1]), gather_indices)

      self.losses = - (tf.log(self.picked_action_probs) * self.targets + 0.01 * self.entropy)
      self.loss = tf.reduce_sum(self.losses, name="loss")

      tf.summary.scalar(self.loss.op.name, self.loss)
      tf.summary.scalar(self.entropy_mean.op.name, self.entropy_mean)
      tf.summary.histogram(self.entropy.op.name, self.entropy)

      if trainable:
        # self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
          global_step=tf.contrib.framework.get_global_step())

    # Merge summaries from this network and the shared network (but not the value net)
    var_scope_name = tf.get_variable_scope().name
    summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
    sumaries = [s for s in summary_ops if "policy_net" in s.name or "shared" in s.name]
    sumaries = [s for s in summary_ops if var_scope_name in s.name]
    self.summaries = tf.summary.merge(sumaries)


class ValueEstimator():
  """
  Value Function approximator. Returns a value estimator for a batch of observations.
  Args:
    reuse: If true, an existing shared network will be re-used.
    trainable: If true we add train ops to the network.
      Actor threads that don't update their local models and don't need
      train ops would set this to false.
  """

  def __init__(self, reuse=False, trainable=True):
    # Placeholders for our input
    # Our input are 4 RGB frames of shape 160, 160 each
    self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
    # The TD target value
    self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

    X = tf.to_float(self.states) / 255.0

    # Graph shared with Value Net
    with tf.variable_scope("shared", reuse=reuse):
      fc1 = build_shared_network(X, add_summaries=(not reuse))

    with tf.variable_scope("value_net"):
      self.logits = tf.contrib.layers.fully_connected(
        inputs=fc1,
        num_outputs=1,
        activation_fn=None)
      self.logits = tf.squeeze(self.logits, squeeze_dims=[1], name="logits")

      self.losses = tf.squared_difference(self.logits, self.targets)
      self.loss = tf.reduce_sum(self.losses, name="loss")

      self.predictions = {
        "logits": self.logits
      }

      # Summaries
      prefix = tf.get_variable_scope().name
      tf.summary.scalar(self.loss.name, self.loss)
      tf.summary.scalar("{}/max_value".format(prefix), tf.reduce_max(self.logits))
      tf.summary.scalar("{}/min_value".format(prefix), tf.reduce_min(self.logits))
      tf.summary.scalar("{}/mean_value".format(prefix), tf.reduce_mean(self.logits))
      tf.summary.scalar("{}/reward_max".format(prefix), tf.reduce_max(self.targets))
      tf.summary.scalar("{}/reward_min".format(prefix), tf.reduce_min(self.targets))
      tf.summary.scalar("{}/reward_mean".format(prefix), tf.reduce_mean(self.targets))
      tf.summary.histogram("{}/reward_targets".format(prefix), self.targets)
      tf.summary.histogram("{}/values".format(prefix), self.logits)

      if trainable:
        # self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
          global_step=tf.contrib.framework.get_global_step())

    var_scope_name = tf.get_variable_scope().name
    summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
    sumaries = [s for s in summary_ops if "policy_net" in s.name or "shared" in s.name]
    sumaries = [s for s in summary_ops if var_scope_name in s.name]
    self.summaries = tf.summary.merge(sumaries)



class base_ac_network(object):

    def __init__(self,action_space,thread_index,device = "/cpu:0"):

        self._num_actions = action_space
        self._thread_index = thread_index
        self._device = device

    def predict_value(self,x):
        raise NotImplementedError()

    def predict_policy(self,x):
        raise NotImplementedError()

    def predict_value_and_policy(self,x):
        raise NotImplementedError()

    def get_vars(self):
        raise NotImplementedError()

    def sync_from(self,src_network, name = None):

        src_vars = src_network.get_vars()
        dst_vars = self.get_vars()

        sync_ops = []


        with tf.device(self._device):
            with tf.name_scope(name,"GameACNetwork",[]) as name:
                for(src_vars,dst_vars) in zip(src_vars,dst_vars):

                    sync_op = tf.assign(dst_vars,src_vars)
                    sync_ops.append(sync_op)

                return tf.group(*sync_ops,name=name)




class lstm_ac_network(base_ac_network):

    def __init__(self,action_space,thread_index,device = "/cpu:0"):

        base_ac_network.__init__(action_space,thread_index,device)

        scope_name = "network_" + str(self._thread_index) + "_thread"

        self.img_width = config.IMAGE_WIDTH
        self.img_height = config.IMAGE_HEIGHT
        self.img_stacked_frames = config.STACKED_FRAMES
        self.lstm_cell_size = 256


    def _create_network(self):

        self.x = tf.placeholder(tf.float32, [None, self.img_height,self.img_width,self.img_stacked_frames], name = 'x')

        self.target_v = tf.placeholder(tf.float32,[None],name='target_v')
        self.action_index = tf.placeholder(tf.float32, [None,self._num_actions])
        self.actions = tf.placeholder(tf.float32, [None, self._num_actions])
        self.action_onehot = tf.one_hot(self.actions, self._num_actions, dtype=tf.float32)

        self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)



        self.var_beta = tf.placeholder(tf.float32,shape=[],name= 'beta')
        self.var_learning_rate = tf.placeholder(tf.float32,shape=[], name='lr')

        self.global_step = tf.Variable(0,trainable=False,name='step')


        # First Convolutional Layer
        self.conv_layer1 = self.conv2d_variable(self.x,8,name='conv_layer1',filter_size=16,strides=[1,4,4,1])
        # Second Convolutional Layer
        self.conv_layer2 = self.conv2d_variable(self.conv_layer1,4,name='conv_layer1',filter_size=32,strides=[1,2,2,1])

        flatten_input_shape = self.conv_layer2.get_shape()
        nb_elements = flatten_input_shape[1]*flatten_input_shape[2]*flatten_input_shape[3]

        self.flat = tf.reshape(self.conv_layer2,shape=[-1,nb_elements._value])

        self.dense_layer1 = self.fully_connected_variable(self.flat,self.lstm_cell_size,name='dense_layer1')

        # LSTM cell inputs

        self.lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_cell_size,state_is_tuple=True)

        self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, self.lstm_cell_size])
        self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, self.lstm_cell_size])
        self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state0,
                                                                self.initial_lstm_state1)

        lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                          self.dense_layer1,
                                                          initial_state=self.initial_lstm_state,
                                                          sequence_length=self.step_size,
                                                          time_major=False,
                                                          scope=scope)

        lstm_outputs = tf.reshape(lstm_outputs, [-1, self.lstm_cell_size])

        self.step_size = tf.placeholder(tf.float32,[1])


        # Outputs from LSTM 256 cells to fully connected layer again (256->num_actions)


        # Create network for policy function
        self.logits_p = self.fully_connected_variable(lstm_outputs,self._num_actions,name='logits_p',func=None)
        self.logits_p_softmax = tf.nn.softmax(self.logits_p)

        # Create network for value function

        self.logits_v = self.fully_connected_variable(lstm_outputs,1,name='logits_v',func=None)

        self.responsible_outputs = tf.reduce_sum(self.logits_p_softmax * self.action_onehot, [1])

        # Loss functions

        self.loss_v = 0.5 * tf.reduce_sum(tf.square(self.target_v-tf.reshape(self.logits_v,[-1])))
        self.entropy = - tf.reduce_sum(self.logits_p_softmax*tf.log(self.logits_p))
        self.loss_p = - tf.reduce_sum(self.advantages * tf.log(self.responsible_outputs))

        self.loss = (0.5*self.loss_v) + self.loss_p - self.entropy * 0.01

        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope)
        self.gradients = tf.gradients(self.loss,local_vars)
        self.var_norms = tf.global_norm(local_vars)
        grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)

        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'global')
        self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))


    def fully_connected_variable(self,input,output_dim,name, func = tf.nn.relu):

        input_dim = input.get_shape().as_list()[-1]
        d = 1.0/np.sqrt(input_dim)

        with tf.variable_scope(name):

            weight_init = tf.random_uniform_initializer(-d,d)
            bias_init = tf.random_uniform_initializer(-d,d)

            w = tf.get_variable('w',shape=[input_dim,output_dim],dtype=tf.float32, initializer=weight_init)
            b = tf.get_variable('b',shape=[output_dim],dtype=tf.float32,initializer=bias_init)

            if func is None:
                output = tf.matmul(input, w) + b
            else:
                output = func(tf.matmul(input, w) + b)

        return output

    def conv2d_variable(self,input,output_dim,name, filter_size,strides,func = tf.nn.relu):

        input_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(input_dim * filter_size * filter_size)

        with tf.variable_scope(name):

            weight_init = tf.random_uniform_initializer(-d,d)
            bias_init = tf.random_uniform_initializer(-d,d)

            w = tf.get_variable('w',shape=[filter_size,filter_size,input_dim,output_dim],dtype=tf.float32, initializer=weight_init)
            b = tf.get_variable('b',shape=[output_dim],dtype=tf.float32,initializer=bias_init)

        output = tf.nn.conv2d(input,w,strides=strides,padding='SAME') + b

        return output

    def _create_tensorboard(self):

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar("Pcost_advantage", self.cost_p_1_agg))
        summaries.append(tf.summary.scalar("Pcost_entropy", self.cost_p_2_agg))
        summaries.append(tf.summary.scalar("Pcost", self.cost_p))
        summaries.append(tf.summary.scalar("Vcost", self.cost_v))
        summaries.append(tf.summary.scalar("LearningRate", self.var_learning_rate))
        summaries.append(tf.summary.scalar("Beta", self.var_beta))
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        summaries.append(tf.summary.histogram("activation_n1", self.n1))
        summaries.append(tf.summary.histogram("activation_n2", self.n2))
        summaries.append(tf.summary.histogram("activation_d2", self.d1))
        summaries.append(tf.summary.histogram("activation_v", self.logits_v))
        summaries.append(tf.summary.histogram("activation_p", self.softmax_p))

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)


    # Function for variables to feed into network
    def _get_var_feed_dict(self):

        step = self.sess.run(self.global_step)
        return step

    # prediction for value network

    def predict_value(self,x):

        prev_lstm_state_out = self.lstm_state_out

        prediction = self.sess.run(self.logits_value, feed_dict = {self.x: x,
                                                                   self.initial_lstm_state0: self.lstm_state_out[0],
                                                                   self.initial_lstm_state1: self.lstm_state_out[1],
                                                                   self.step_size:[1]})
        self.lstm_state_out = prev_lstm_state_out
        return prediction

    # prediction for policy network
    def predict_policy(self,x):
        prediction = self.sess.run(self.softmax_policy, feed_dict = {self.x: x,
                                                                     self.initial_lstm_state0: self.lstm_state_out[0],
                                                                     self.initial_lstm_state1: self.lstm_state_out[1],
                                                                     self.step_size:[1]})
        return prediction

    # function for single prediction
    def predict_single(self, x):
        return self.predict_p(x[None, :])[0]

    # prediction for p & v network
    def predict_policy_and_value(self,x):

        return self.sess.run([self.softmax_policy,self.logit_value,self.lstm_state],
                             feed_dict = {self.x: x,
                                          self.initial_lstm_state0: self.lstm_state_out[0],
                                          self.initial_lstm_state1: self.lstm_state_out[1],
                                          self.step_size:[1]})


    # function to train networks
    def train_network(self,x,y_r,action):

        feed_dict = self._get_var_feed_dict()
        feed_dict.update({self.x:x,self.target_v:y_r,self.action:action})
        self.sess.run(self.train_op, feed_dict = feed_dict)

    # function to log data
    def log_data(self,x,y_r,action):

        feed_dict = self._get_var_feed_dict()
        feed_dict.update({self.x:x,self.target_v:y_r,self.action:action})
        step, summary = self.sess.run([self.global_step,self.summary_op], feed_dict = {feed_dict})
        self.log_writer.add_summary(summary,step)

    def get_variables_names(self):
        return [var.name for var in self.graph.get_collection('trainable_variables')]

    def get_variable_value(self, name):
        return self.sess.run(self.graph.get_tensor_by_name(name))

    def reset_state(self):
        self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros[1,self.lstm_cell_size],
                                                            np.zeros[1,self.lstm_cell_size])

    def get_vars(self):

        tempt_var = [self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,
                     self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2,]

        return
    # NB Functions to save and load checkpoint files need to be added




