import sys
import numpy as np
import tensorflow as tf
import math
import os
import pickle
#from tensorflow.python.ops import ops
#from tensorflow.python.ops import math_ops
#from tensorflow.python.ops.losses.losses_impl import * #compute_weighted_loss

''' use original layer-based to implement model
    The main difference is that here we initially the thresholds
    as the format of their embedding using Tensorflow, and then
    use a light DNN to train the embedding to new format embedding
    according to the loss function
'''
from sklearn.decomposition import PCA
from sklearn.metrics import *
#from tensorflow.python.framework import ops
from sklearn.cross_validation import train_test_split

#from deepautoencoder import StackedAutoEncoder

from timeit import default_timer as timer

def mean_absolute_percentage_error(labels, predictions):
    return np.mean(np.abs((predictions - labels) * 1.0 / (labels + 0.000001))) * 100

def __eval__(predictions, labels):
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    mape = mean_absolute_percentage_error(labels, predictions)
    return (mse, mae, mape)


# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

# Def custom square function using np.square instead of tf.square:
def myround(x, name=None):

    with ops.op_scope([x], name, "Myround") as name:
        sqr_x = py_func(np.round,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=_MyRoundGrad)  # <-- here's the call to the gradient
        return sqr_x[0]

# Actual gradient:
def _MyRoundGrad(op, grad):
    x = op.inputs[0]
    #print(grad)
    return (np.round(x) + 0.000001) / (x + 0.000001)  # add a "small" error just to see the difference:


class SelNet(object):
    def __init__(self,
                hidden_units,
                vae_hidden_units,
                batch_size,
                epochs,
                epochs_vae,
                learning_rate,
                log_option,
                tau_embedding_size,
                original_x_dim,
                dimreduce_x_dim,
                test_data_predictions_labels_file,
                valid_data_predictions_labels_file,
                regressor_name,
                model_file,
                unit_len,
                max_tau,
                tau_part_num,
                partition_option,
                loss_type):
        self.hidden_units = hidden_units
        self.vae_hidden_units = vae_hidden_units
        self.epochs = epochs
        self.epochs_vae = epochs_vae
        self.learning_rate = learning_rate
        self.log_option = log_option
        self.tau_embedding_size = tau_embedding_size
        self.original_x_dim = original_x_dim
        self.dimreduce_x_dim = dimreduce_x_dim
        self._vae_n_z = dimreduce_x_dim
        self.test_data_predictions_labels_file = test_data_predictions_labels_file
        self.valid_data_predictions_labels_file = valid_data_predictions_labels_file
        self.batch_size = batch_size
        self.regressor_name = regressor_name
        self.model_file = model_file
        # check whether model_dir exist, if not mkdir one
        # Here note that model_file is just a dir of model
        if not os.path.exists(self.model_file):
            os.makedirs(self.model_file)

        # prediction time
        self.prediction_time = 0.0
        # define the maximum threshold
        self.max_tau = max_tau
        self.tau_max = max_tau
        self.unit_len = unit_len # default = 1
        #self.gate_layer = self.unit_len * self.max_tau

        self.hidden_num = len(hidden_units)
        # assume the unit_len is divisible by hidden_num
        self.hidden_unit_len = self.unit_len // self.hidden_num

        self.tau_part_num = tau_part_num
        self.gate_layer = self.unit_len * self.tau_part_num

        self.partition_option = partition_option
        # MSLE, huber loss, huber log loss
        self.loss_type = loss_type

        # types of spline
        self.spline_type = 'selnet_linear'

    # change spline type to qud
    def change_spline_type(self, spline_type):
        self.spline_type = spline_type


    def __ae__(self, x_input):
        '''
        Transfer original X to dense representation
        :param x_input:
        :return:
        '''
        # Encoder
        fc1 = tf.layers.dense(inputs=x_input, units=self.vae_hidden_units[0],
                              activation=tf.nn.elu, name=self.regressor_name + 'vae_fc_e1')
        fc2 = tf.layers.dense(inputs=fc1, units=self.vae_hidden_units[1],
                              activation=tf.nn.elu, name=self.regressor_name + 'vae_fc_e2')
        fc3 = tf.layers.dense(inputs=fc2, units=self.vae_hidden_units[2],
                              activation=tf.nn.elu, name=self.regressor_name + 'vae_fc_e3')

        # generate hidden layer z
        z_mu = tf.layers.dense(inputs=fc3, units=self._vae_n_z,
                            activation=None, name=self.regressor_name + 'vae_fc_e4')

        # hidden layer
        hidden_z = z_mu #+ tf.sqrt(tf.exp(z_log_sigma_sq)) * eps

        # Decoder
        g1 = tf.layers.dense(inputs=hidden_z, units=self.vae_hidden_units[2],
                             activation=tf.nn.elu, name=self.regressor_name + 'vae_fc_d1')
        g2 = tf.layers.dense(inputs=g1, units=self.vae_hidden_units[1],
                             activation=tf.nn.elu, name=self.regressor_name + 'vae_fc_d2')
        g3 = tf.layers.dense(inputs=g2, units=self.vae_hidden_units[0],
                             activation=tf.nn.elu, name=self.regressor_name + 'vae_fc_d3')

        x_hat = tf.layers.dense(inputs=g3, units=self.original_x_dim,
                                activation=tf.nn.relu)

        recon_loss = tf.losses.mean_squared_error(predictions=x_hat, labels=x_input)
        recon_loss = tf.reduce_mean(recon_loss)

        ae_loss = recon_loss
        return ae_loss, hidden_z #tf.cond(self.vae_option > 0, hidden_z, z_mu) #hidden_z

    def _construct_rhos(self, x_fea, x_fea_dr):
        '''
        :param x_fea:
        :param x_fea_dr:
        :param tau: a matrix with N * num_deltataus
        :return:
        '''
        # first concatenate X
        new_x = tf.concat([x_fea, x_fea_dr], 1)

        # concatenate new X with threshold embedding
        # new_x_fea = new_x # tf.concat([new_x, tau_embed], 1)
        new_x_fea = new_x #tf.concat([new_x, tau_embed], 1)

        rhos = []
        # fc layers
        out = tf.layers.dense(inputs=new_x_fea, units=self.hidden_units[0],
                              activation=tf.nn.relu, name=self.regressor_name + 'fc_1')

        # 1st embedding
        rho_1 = tf.layers.dense(inputs=out, units=self.hidden_unit_len * (self.tau_part_num + 1),
                            activation=tf.nn.relu, name=self.regressor_name + 'embed_1')

        # reshape 1st embedding
        rho_1 = tf.reshape(rho_1, [-1, self.tau_part_num + 1, self.hidden_unit_len])

        rhos.append(rho_1)

        out = tf.layers.dense(inputs=out, units=self.hidden_units[1],
                              activation=tf.nn.relu, name=self.regressor_name + 'fc_2')

        # 2nd embedding
        rho_2 = tf.layers.dense(inputs=out, units=self.hidden_unit_len * (self.tau_part_num + 1),
                            activation=tf.nn.relu, name=self.regressor_name + 'embed_2')

        # reshape 2nd embedding
        rho_2 = tf.reshape(rho_2, [-1, self.tau_part_num + 1, self.hidden_unit_len])

        rhos.append(rho_2)

        out = tf.layers.dense(inputs=out, units=self.hidden_units[2],
                              activation=tf.nn.relu, name=self.regressor_name + 'fc_3')
        # out = tf.nn.dropout(out, keep_prob=self.keep_prob, name=self.regressor_name + 'dropout')

        # 3rd embedding
        rho_3 = tf.layers.dense(inputs=out, units=self.hidden_unit_len * (self.tau_part_num + 1),
                            activation=tf.nn.relu, name=self.regressor_name + 'embed_3')

        # reshape 3rd embedding
        rho_3 = tf.reshape(rho_3, [-1, self.tau_part_num + 1, self.hidden_unit_len])

        rhos.append(rho_3)

        out = tf.layers.dense(inputs=out, units=self.hidden_units[3],
                              activation=tf.nn.relu, name=self.regressor_name + 'fc_4')

        # 4th embedding
        rho_4 = tf.layers.dense(inputs=out, units=self.hidden_unit_len * (self.tau_part_num + 1),
                            activation=tf.nn.relu, name=self.regressor_name + 'embed_4')

        # reshape 3rd embedding
        rho_4 = tf.reshape(rho_4, [-1, self.tau_part_num + 1, self.hidden_unit_len])

        rhos.append(rho_4)

        # concate embedding matrix RHO
        gate = rhos[0]
        for hidden_id in range(1, 4, 1):
            gate = tf.concat([gate, rhos[hidden_id]], 2)

        return gate


    def _partition_threshold(self, x_fea, x_fea_dr, tau, eps=0.0000001):
        # first concat X
        new_x = tf.concat([x_fea, x_fea_dr], 1)
        out = tf.layers.dense(inputs=new_x, units=self.hidden_units[0], activation=tf.nn.elu,
                              name=self.regressor_name + 'tau_part_1')
        out = tf.layers.dense(inputs=out, units=self.hidden_units[1], activation=tf.nn.elu,
                              name=self.regressor_name + 'tau_part_2')

        out = tf.layers.dense(inputs=out, units=self.tau_part_num, activation=tf.nn.elu,
                              name=self.regressor_name + 'tau_part_3')

        if self.partition_option == 'softmax':
            dist_tau = tf.nn.softmax(out)
        elif self.partition_option == 'l2':
            out = tf.multiply(out, out) + eps
            norm = tf.expand_dims(tf.reduce_sum(out, 1), 1)
            norm = tf.tile(norm, [1, self.tau_part_num])
            dist_tau = tf.truediv(out, norm)
        else:
            raise ValueError('wrong partition option')

        accum_tau = tf.cumsum(dist_tau, 1) - dist_tau
        residue_tau = tf.nn.relu(tau - accum_tau * self.max_tau)
        #residue_tau_s = tf.manip.roll(residue_tau, shift=-1, axis=1)
        residue_tau_s = residue_tau[:, 1:]
        residue_tau_s = tf.concat([residue_tau_s, tf.expand_dims(tf.zeros(self.input_num), axis=1)], 1)

        precent_tau = tf.divide(tf.nn.relu(residue_tau - residue_tau_s), dist_tau * self.max_tau)

        precent_tau = tf.concat([tf.expand_dims(tf.ones(self.input_num), axis=1), precent_tau], 1)

        return precent_tau


    def _construct_model(self, x_fea, x_fea_dr, tau):

        gate = self._construct_rhos(x_fea, x_fea_dr)

        # integrate
        w_t = tf.get_variable(self.regressor_name + 'w_t', [self.tau_part_num + 1, self.unit_len], tf.float32)
        b_t = tf.get_variable(self.regressor_name + 'b_t', [self.tau_part_num + 1, self.unit_len], tf.float32)
        gate = tf.nn.relu(tf.multiply(gate, w_t) + b_t)

        # conv with mask
        kernel_ = tf.ones([self.unit_len], dtype=tf.float32, name=self.regressor_name + 'k')
        kernel = tf.reshape(kernel_, [1, int(kernel_.shape[0]), 1], name=self.regressor_name + 'kernel')
        # reshape layer
        #gate = tf.reshape(gate, [-1, self.gate_layer, 1], name=self.regressor_name + 'gate_v1')
        gate = tf.nn.relu(tf.squeeze(tf.nn.conv1d(gate, kernel, 1, 'VALID')) )

        # narrow down the domain of Delta Y
        tau_gate = self._partition_threshold(x_fea, x_fea_dr, tau)
        gate = tf.multiply(gate, tau_gate)

        prediction = tf.reduce_sum(gate, 1)
        # expand dim
        prediction = tf.expand_dims(prediction, 1)

        # add exp
        #prediction = tf.exp(prediction)

        return prediction, gate


    def _partition_threshold_quadratic(self, x_fea, x_fea_dr, eps=1e-6):
        # first concat x
        new_x = tf.concat([x_fea, x_fea_dr], 1)
        out = tf.layers.dense(inputs=new_x, units=self.hidden_units[0], activation=tf.nn.elu,
                              name=self.regressor_name + 'tau_part_1')
        out = tf.layers.dense(inputs=out, units=self.hidden_units[1], activation=tf.nn.elu,
                              name=self.regressor_name + 'tau_part_2')

        out = tf.layers.dense(inputs=out, units=self.tau_part_num, activation=tf.nn.elu,
                              name=self.regressor_name + 'tau_part_3')

        if self.partition_option == 'softmax':
            dist_tau = tf.nn.softmax(out)
        elif self.partition_option == 'l2':
            out = tf.multiply(out, out) + eps
            norm = tf.expand_dims(tf.reduce_sum(out, 1), 1)
            norm = tf.tile(norm, [1, self.tau_part_num])
            dist_tau = tf.truediv(out, norm)
        else:
            raise ValueError('wrong partition option')

        partition_tau = tf.cumsum(dist_tau, 1) * self.tau_max
        return dist_tau, partition_tau



    def tridiagonal(self, n):
        r = tf.range(n)
        diag = tf.fill([n], 1.0)
        sub = tf.fill([n-1], 1.0)
        sup = tf.fill([n-1], 0.0)
        ii = tf.concat([r, r[1:], r[:-1]], axis=0)
        jj = tf.concat([r, r[:-1], r[1:]], axis=0)
        idx = tf.stack([ii, jj], axis=1)
        values = tf.concat([diag, sub, sup], axis=0)
        return tf.expand_dims(tf.scatter_nd(idx, values, [n, n]), 0)


    def _construct_model_quadratic(self, x_fea, x_fea_dr, tau, trid):
        ''' quad spline
        '''
        gate = self._construct_rhos(x_fea, x_fea_dr)

        # integrate
        w_t = tf.get_variable(self.regressor_name + 'w_t', [self.tau_part_num + 1, self.unit_len], tf.float32)
        b_t = tf.get_variable(self.regressor_name + 'b_t', [self.tau_part_num + 1, self.unit_len], tf.float32)
        gate = tf.nn.relu(tf.multiply(gate, w_t) + b_t)

        # conv with mask
        kernel_ = tf.ones([self.unit_len], dtype=tf.float32, name=self.regressor_name + 'k')
        kernel = tf.reshape(kernel_, [1, int(kernel_.shape[0]), 1], name=self.regressor_name + 'kernel')
        # reshape layer
        #gate = tf.reshape(gate, [-1, self.gate_layer, 1], name=self.regressor_name + 'gate_v1')
        gate = tf.nn.relu(tf.squeeze(tf.nn.conv1d(gate, kernel, 1, 'VALID')) )

        # quad:
        #trid = self.tridiagonal(self.tau_part_num + 1)

        # 1st value is p_0, others are z_0, z_1, ..., z_n
        f_first = gate[:, :1]
        Z = gate[:, 1:]
        Z_zeros = tf.fill([tf.shape(Z)[0], 1], 0.0)
        Z = tf.concat([Z_zeros, Z], 1)
        D = tf.squeeze(tf.matmul(trid, tf.expand_dims(Z, 2)), 2)

        dist_tau, partition_tau = self._partition_threshold_quadratic(x_fea, x_fea_dr)
        H = dist_tau * self.tau_max
        DeltaP = tf.multiply(D[:, 1:], H) / 2.0
        Px = tf.cumsum(tf.concat([f_first, DeltaP], 1), 1)

        tau_first = tf.fill([tf.shape(partition_tau)[0], 1], 0.0)
        partition_tau_ = tf.concat([tau_first, partition_tau[:, :-1]], 1)
        indices = tf.searchsorted(partition_tau_, tau, side='left')
        indices = tf.squeeze(indices) - 1
        mask = tf.one_hot(indices, self.tau_part_num)

        tau_exp = tf.tile(tau, [1, self.tau_part_num])
        P = tf.multiply(Z[:, :-1], tau_exp - partition_tau_) + 0.5 * (Z[:, 1:] - Z[:, :-1]) / H * ((tau_exp - partition_tau_) ** 2) + Px[:, :-1]

        prediction = tf.reduce_sum(tf.multiply(P, mask), 1)
        # expand dim
        prediction = tf.expand_dims(prediction, 1)


        return prediction, gate

    def predict_vae_dnn(self, test_X, test_tau):
        ''' Prediction
        '''
        x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.original_x_dim], name=self.regressor_name + 'original_X')


        tau_input = tf.placeholder(dtype=tf.float32, shape=[None, self.tau_part_num], name=self.regressor_name + 'tau')
        self.bn_phase = tf.placeholder(dtype=tf.bool, name=self.regressor_name + 'Phase')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name=self.regressor_name + 'Dropout')
        # input number
        self.input_num = tf.placeholder(dtype=tf.int32, name=self.regressor_name + 'input_num')
        # vae_option
        self.vae_option = tf.placeholder(dtype=tf.int32, name=self.regressor_name + 'vae_option')

        _, x_input_dr = self.__ae__(x_input)
        # reconstruct the graph
        predictions_tensor, gate = self._construct_model(x_input, x_input_dr, tau_input)

        # tensorflow saver
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_file)
            # make prediction
            startTime = timer()
            predictions = sess.run(predictions_tensor,
                            feed_dict={x_input: test_X,
                                            tau_input: test_tau,
                                            self.bn_phase: 0,
                                            self.keep_prob: 1.0,
                                            self.input_num:test_X.shape[0],
                                            self.vae_option: 0})

            if self.log_option:
                predictions = np.hstack(predictions)
                predictions = np.exp(predictions)

            self.prediction_time += timer() - startTime

        return predictions

    def train_vae_dnn(self, train_X, train_tau, train_y, valid_X, valid_tau, valid_y):
        ''' Train and validate
            train_X: original Hamming (or Euclidean) features
            train_tau_gate: tau_max dimensional mask vector with tau prefix 1s (0s remaining)
        '''
        tf.reset_default_graph()

        x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.original_x_dim], name=self.regressor_name + 'original_X')

        # tau_input is an one-hot vector -- maybe useless
        tau_input = tf.placeholder(dtype=tf.float32, shape=[None, self.tau_part_num], name=self.regressor_name + 'tau')

        target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name=self.regressor_name + 'Target')
        target_taus = tf.placeholder(dtype=tf.float32, shape=[None, self.tau_max], name=self.regressor_name + 'Target_taus')

        # to control option of batch normalization
        self.bn_phase = tf.placeholder(dtype=tf.bool, name=self.regressor_name + 'Phase')

        # add dropout to avoid overfitting
        self.keep_prob = tf.placeholder(dtype=tf.float32, name=self.regressor_name + 'Dropout')

        # input number
        self.input_num = tf.placeholder(dtype=tf.int32, name=self.regressor_name + 'input_num')
        # vae_option
        self.vae_option = tf.placeholder(dtype=tf.int32, name=self.regressor_name + 'vae_option')

        # set up learning rate
        self.learning_rate_vae = tf.placeholder(dtype=tf.float32, name=self.regressor_name + 'lr_vae')
        self.learning_rate_nn = tf.placeholder(dtype=tf.float32, name=self.regressor_name + 'lr_nn')

        # if use log, then process log operation
        if self.log_option:
            train_y = np.array(np.log(train_y + 1), dtype=np.float32)
            test_y = np.array(np.log(test_y + 1), dtype=np.float32)
            test_y = np.exp(test_y)
            if len(test_y.shape) <= 1:
                test_y = test_y[:, np.newaxis]

        vae_loss, x_input_dr = self.__ae__(x_input)

        predictions_tensor, gate_tensor = self._construct_model(x_input, x_input_dr, tau_input)

        #loss = vae_loss + tf.losses.mean_squared_error(predictions=tf.log(predictions_tensor + 1), labels=tf.log(target + 1))
        #loss = tf.losses.mean_squared_error(predictions=predictions_tensor, labels=target)

        loss = vae_loss
        loss_nn = 0.0
        #for t in range(self.tau_max):
            #loss_nn += tf.losses.mean_squared_error(predictions=tf.log(tf.slice(gate_tensor, [0, t], [self.batch_size, 1]) + 1), labels=tf.log(tf.slice(target_taus, [0, t], [self.batch_size, 1]) + 1))

        if self.loss_type == 'msle':
            loss = loss + 0.1 * loss_nn + tf.losses.mean_squared_error(predictions=tf.log(predictions_tensor + 1), labels=tf.log(target + 1))
        elif self.loss_type == 'abs_diff':
            loss = loss + 0.1 * loss_nn + tf.losses.absolute_difference(labels=tf.log(target + 1), predictions=tf.log(predictions_tensor + 1))
        elif self.loss_type == 'huber':
            # huber loss
            loss = loss + 0.1 * loss_nn + tf.losses.huber_loss(labels=target, predictions=predictions_tensor, delta=1.345)
        elif self.loss_type == 'huber_log':
            loss = loss + 0.1 * loss_nn + tf.losses.huber_loss(labels=tf.log(target + 1), predictions=tf.log(predictions_tensor + 1), delta=1.345)
        elif self.loss_type == 'huber_log_opt':
            # calculate the median value
            residue = tf.log(target + 1) - tf.log(predictions_tensor + 1)
            residue_median = tf.contrib.distributions.percentile(residue, q=50.)
            mad = tf.contrib.distributions.percentile(math_ops.abs(residue - residue_median), q=50.) / 0.6745
            residue_opt = residue / mad
            # huber loss
            delta = 1.345
            error = residue_opt
            abs_error = math_ops.abs(error)
            quadratic = math_ops.minimum(abs_error, delta)
            linear = math_ops.subtract(abs_error, quadratic)
            _losses = math_ops.add(
                    math_ops.multiply(
                        ops.convert_to_tensor(0.5, dtype=quadratic.dtype),
                        math_ops.multiply(quadratic, quadratic)),
                    math_ops.multiply(delta, linear))
            loss = loss + 0.1 * loss_nn + compute_weighted_loss(
                    _losses, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
        else:
            raise ValueError('Wrong Loss Function Option')

        # optimizer
        #optimizer_vae = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(vae_loss)
        #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

        optimizer_vae = tf.train.AdamOptimizer(self.learning_rate_vae).minimize(vae_loss)
        optimizer = tf.train.AdamOptimizer(self.learning_rate_nn).minimize(loss)

        # init all variables
        init = tf.global_variables_initializer()

        # Save the model
        saver = tf.train.Saver()
        step = 0

        session_config = tf.ConfigProto(log_device_placement=True)
        session_config.gpu_options.allow_growth = True
        with tf.Session(config=session_config) as sess:
            #with tf.Session() as sess:
            sess.run(init)

            # 1. first initialize VAE (unsupervised)
            learning_rate_vae_ = self.learning_rate
            decay_rate, decay_step = 0.96, 5
            for epoch in range(self.epochs_vae):
                n_batches = int(train_X.shape[0] / self.batch_size) + 1
                if epoch != 0 and (epoch % decay_step == 0):
                    learning_rate_vae_ = learning_rate_vae_ * (decay_rate ** (epoch / decay_step))
                for b in range(n_batches):
                    batch_X = self.getBatch_vae(b, self.batch_size, train_X)
                    sess.run(optimizer_vae, feed_dict={x_input: batch_X, self.learning_rate_vae: learning_rate_vae_})

                    # check points
                    if b % 50 == 0:
                        eval_loss = sess.run(vae_loss, feed_dict={x_input: batch_X})
                        print('VAE Epoch: {}, batch: {}, loss: {}'.format(epoch, b, eval_loss))


            # 2. fit all the training data to estimate
            learning_rate_nn_ = self.learning_rate
            epoch_decay_start = 100
            decay_rate, decay_step = 0.96, 10
            for i in range(self.epochs):
                n_batches = int(train_X.shape[0] / self.batch_size) + 1
                for b in range(n_batches):
                    # get current batch
                    batch_original_X, batch_tau, batch_y = self.getBatch_(b,
                            self.batch_size, train_X, train_tau, train_y)
                    # batch_init_indices = np.zeros(self.batch_size, dtype=np.int32)
                    sess.run(optimizer, feed_dict={x_input: batch_original_X,
                                                    tau_input: batch_tau,
                                                    #target: batch_y[:, -1][:, np.newaxis],
                                                    #target_taus: batch_y[:, :-1],
                                                    target: batch_y,
                                                    self.learning_rate_nn: learning_rate_nn_,
                                                    self.bn_phase: 1,
                                                    self.keep_prob: 0.9,
                                                    self.input_num: self.batch_size,
                                                    self.vae_option: 1})
                    # make sure gate and final fc are non-negative
                    #sess.run(self.clip)

                    # check points
                    if b % 50 == 0:
                        eval_loss = sess.run(loss, feed_dict={x_input: batch_original_X,
                                                    tau_input: batch_tau,
                                                    #target: batch_y[:, -1][:, np.newaxis],
                                                    #target_taus: batch_y[:, :-1],
                                                    target: batch_y,
                                                    self.learning_rate_nn: learning_rate_nn_,
                                                    self.bn_phase: 0,
                                                    self.keep_prob: 1.0,
                                                    self.input_num: self.batch_size,
                                                    self.vae_option: 1})
                        print('Epoch: {}, batch: {}, loss: {}'.format(i, b, eval_loss))

                        # write training log
                        #train_writer.add_summary(summary, global_step=step)
                    step += 1

                if i == epoch_decay_start:
                    learning_rate_nn_ /= 4.0
                if i > epoch_decay_start and (i % decay_step == 0):
                    learning_rate_nn_ = learning_rate_nn_ * (decay_rate ** ((i - epoch_decay_start) / decay_step))

                # save the model
                if i % 100 == 0 or ((i + 1) == self.epochs):
                    saver.save(sess, save_path=self.model_file, global_step=i)

                # evaluate for testing data
                if i % 10 == 0 or ((i + 1) == self.epochs):
                    '''
                    # test !!!
                    # split original X, dimreduce X, and threshold
                    n_batch_test = int(test_X.shape[0] / self.batch_size) + 1
                    for b_ in range(n_batch_test):
                        batch_test_original_X, batch_test_tau, batch_test_y = self.getBatch_(b_,
                                        self.batch_size, test_X, test_tau, test_y)
                        predictions_batch = sess.run(predictions_tensor,
                                        feed_dict={x_input: batch_test_original_X,
                                                    tau_input: batch_test_tau,
                                                    self.bn_phase: 0,
                                                    self.keep_prob: 1.0,
                                                    self.input_num: self.batch_size,
                                                    self.vae_option: 0})
                        if b_ == 0:
                            predictions = predictions_batch
                        else:
                            predictions = np.concatenate((predictions, predictions_batch), axis=0)
                    # clip
                    predictions = predictions[:test_X.shape[0]]

                    # check whether log is used
                    if self.log_option:
                        predictions = np.exp(predictions)
                    predictions = np.hstack(predictions)
                    test_y_labels = np.hstack(test_y)

                    #predictions = np.array(predictions, dtype=np.float64)
                    #test_y_labels = np.array(test_y_labels, dtype=np.float64)
                    print('Test Epoch: {}, loss: {}'.format(i, __eval__(predictions, test_y_labels)))

                    # save to files
                    L = [[i_, j_] for i_, j_ in zip(predictions, test_y_labels)]
                    L = np.array(L)
                    #save_file = './test/test_customDNN_predictions_labels_epoch_' + str(i)
                    save_file = self.test_data_predictions_labels_file + str(i)
                    np.save(save_file, L)
                    '''
                    # valid !!!
                    # split original X, dimreduce X, and threshold
                    n_batch_valid = int(valid_X.shape[0] / self.batch_size) + 1
                    for b_ in range(n_batch_valid):
                        batch_valid_original_X, batch_valid_tau, batch_valid_y = self.getBatch_(b_,
                                        self.batch_size, valid_X, valid_tau, valid_y)
                        valid_predictions_batch = sess.run(predictions_tensor,
                                        feed_dict={x_input: batch_valid_original_X,
                                                    tau_input: batch_valid_tau,
                                                    self.bn_phase: 0,
                                                    self.keep_prob: 1.0,
                                                    self.input_num: self.batch_size,
                                                    self.vae_option: 0})
                        if b_ == 0:
                            valid_predictions = valid_predictions_batch
                        else:
                            valid_predictions = np.concatenate((valid_predictions, valid_predictions_batch), axis=0)
                    # clip
                    valid_predictions = valid_predictions[:valid_X.shape[0]]

                    # check whether log is used
                    if self.log_option:
                        valid_predictions = np.exp(valid_predictions)
                    valid_predictions = np.hstack(valid_predictions)
                    valid_y_labels = np.hstack(valid_y)

                    #predictions = np.array(predictions, dtype=np.float64)
                    #test_y_labels = np.array(test_y_labels, dtype=np.float64)
                    print('Valid Epoch: {}, loss: {}'.format(i, __eval__(valid_predictions, valid_y_labels)))

                    # save to files
                    L = [[i_, j_] for i_, j_ in zip(valid_predictions, valid_y_labels)]
                    L = np.array(L)
                    #save_file = './test/test_customDNN_predictions_labels_epoch_' + str(i)
                    save_file = self.valid_data_predictions_labels_file + str(i)
                    np.save(save_file, L)


    def train_vae_dnn_quad(self, train_X, train_tau, train_y, valid_X, valid_tau, valid_y, test_X, test_tau, test_y):
        ''' Train and validate
            train_X: original Hamming (or Euclidean) features
            train_tau_gate: tau_max dimensional mask vector with tau prefix 1s (0s remaining)
        '''
        tf.reset_default_graph()

        x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.original_x_dim], name=self.regressor_name + 'original_X')

        # tau_input is an one-hot vector -- maybe useless
        # tau_input = tf.placeholder(dtype=tf.float32, shape=[None, self.tau_part_num], name=self.regressor_name + 'tau')
        tau_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name=self.regressor_name + 'tau')

        target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name=self.regressor_name + 'Target')
        target_taus = tf.placeholder(dtype=tf.float32, shape=[None, self.tau_max], name=self.regressor_name + 'Target_taus')

        # to control option of batch normalization
        self.bn_phase = tf.placeholder(dtype=tf.bool, name=self.regressor_name + 'Phase')

        # add dropout to avoid overfitting
        self.keep_prob = tf.placeholder(dtype=tf.float32, name=self.regressor_name + 'Dropout')

        # input number
        self.input_num = tf.placeholder(dtype=tf.int32, name=self.regressor_name + 'input_num')
        # vae_option
        self.vae_option = tf.placeholder(dtype=tf.int32, name=self.regressor_name + 'vae_option')

        # set up learning rate
        self.learning_rate_vae = tf.placeholder(dtype=tf.float32, name=self.regressor_name + 'lr_vae')
        self.learning_rate_nn = tf.placeholder(dtype=tf.float32, name=self.regressor_name + 'lr_nn')

        # if use log, then process log operation
        if self.log_option:
            train_y = np.array(np.log(train_y + 1), dtype=np.float32)
            test_y = np.array(np.log(test_y + 1), dtype=np.float32)
            test_y = np.exp(test_y)
            if len(test_y.shape) <= 1:
                test_y = test_y[:, np.newaxis]

        vae_loss, x_input_dr = self.__ae__(x_input)

        trid = self.tridiagonal(self.tau_part_num + 1)
        predictions_tensor, gate_tensor = self._construct_model_quadratic(x_input, x_input_dr, tau_input, trid)

        #loss = vae_loss + tf.losses.mean_squared_error(predictions=tf.log(predictions_tensor + 1), labels=tf.log(target + 1))
        #loss = tf.losses.mean_squared_error(predictions=predictions_tensor, labels=target)

        loss = vae_loss
        loss_nn = 0.0
        #for t in range(self.tau_max):
            #loss_nn += tf.losses.mean_squared_error(predictions=tf.log(tf.slice(gate_tensor, [0, t], [self.batch_size, 1]) + 1), labels=tf.log(tf.slice(target_taus, [0, t], [self.batch_size, 1]) + 1))

        if self.loss_type == 'msle':
            loss = loss + 0.1 * loss_nn + tf.losses.mean_squared_error(predictions=tf.log(predictions_tensor + 1), labels=tf.log(target + 1))
        elif self.loss_type == 'abs_diff':
            loss = loss + 0.1 * loss_nn + tf.losses.absolute_difference(labels=tf.log(target + 1), predictions=tf.log(predictions_tensor + 1))
        elif self.loss_type == 'huber':
            # huber loss
            loss = loss + 0.1 * loss_nn + tf.losses.huber_loss(labels=target, predictions=predictions_tensor, delta=1.345)
        elif self.loss_type == 'huber_log':
            loss = loss + 0.1 * loss_nn + tf.losses.huber_loss(labels=tf.log(target + 1), predictions=tf.log(predictions_tensor + 1), delta=1.345)
        elif self.loss_type == 'huber_log_opt':
            # calculate the median value
            residue = tf.log(target + 1) - tf.log(predictions_tensor + 1)
            residue_median = tf.contrib.distributions.percentile(residue, q=50.)
            mad = tf.contrib.distributions.percentile(math_ops.abs(residue - residue_median), q=50.) / 0.6745
            residue_opt = residue / mad
            # huber loss
            delta = 1.345
            error = residue_opt
            abs_error = math_ops.abs(error)
            quadratic = math_ops.minimum(abs_error, delta)
            linear = math_ops.subtract(abs_error, quadratic)
            _losses = math_ops.add(
                    math_ops.multiply(
                        ops.convert_to_tensor(0.5, dtype=quadratic.dtype),
                        math_ops.multiply(quadratic, quadratic)),
                    math_ops.multiply(delta, linear))
            loss = loss + 0.1 * loss_nn + compute_weighted_loss(
                    _losses, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
        else:
            raise ValueError('Wrong Loss Function Option')

        # optimizer
        #optimizer_vae = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(vae_loss)
        #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

        optimizer_vae = tf.train.AdamOptimizer(self.learning_rate_vae).minimize(vae_loss)
        optimizer = tf.train.AdamOptimizer(self.learning_rate_nn).minimize(loss)

        # init all variables
        init = tf.global_variables_initializer()

        # Save the model
        saver = tf.train.Saver()
        step = 0

        session_config = tf.ConfigProto(log_device_placement=True)
        session_config.gpu_options.allow_growth = True
        with tf.Session(config=session_config) as sess:
            #with tf.Session() as sess:
            sess.run(init)

            # 1. first initialize VAE (unsupervised)
            learning_rate_vae_ = self.learning_rate
            decay_rate, decay_step = 0.96, 5
            for epoch in range(self.epochs_vae):
                n_batches = int(train_X.shape[0] / self.batch_size) + 1
                if epoch != 0 and (epoch % decay_step == 0):
                    learning_rate_vae_ = learning_rate_vae_ * (decay_rate ** (epoch / decay_step))
                for b in range(n_batches):
                    batch_X = self.getBatch_vae(b, self.batch_size, train_X)
                    sess.run(optimizer_vae, feed_dict={x_input: batch_X, self.learning_rate_vae: learning_rate_vae_})

                    # check points
                    if b % 50 == 0:
                        eval_loss = sess.run(vae_loss, feed_dict={x_input: batch_X})
                        print('VAE Epoch: {}, batch: {}, loss: {}'.format(epoch, b, eval_loss))


            # 2. fit all the training data to estimate
            learning_rate_nn_ = self.learning_rate
            epoch_decay_start = 100
            decay_rate, decay_step = 0.96, 10
            for i in range(self.epochs):
                n_batches = int(train_X.shape[0] / self.batch_size) + 1
                # sample some batches
                batch_num_ratio = 0.3
                np.random.seed(i**2)
                sc = np.random.choice(n_batches, int(n_batches * batch_num_ratio), replace=False)
                batches_IDS = np.arange(n_batches)[sc]
                #for b in range(n_batches):
                for b in batches_IDs:
                    # get current batch
                    batch_original_X, batch_tau, batch_y = self.getBatch_(b,
                            self.batch_size, train_X, train_tau, train_y)
                    # batch_init_indices = np.zeros(self.batch_size, dtype=np.int32)
                    sess.run(optimizer, feed_dict={x_input: batch_original_X,
                                                    tau_input: batch_tau,
                                                    #target: batch_y[:, -1][:, np.newaxis],
                                                    #target_taus: batch_y[:, :-1],
                                                    target: batch_y,
                                                    self.learning_rate_nn: learning_rate_nn_,
                                                    self.bn_phase: 1,
                                                    self.keep_prob: 0.9,
                                                    self.input_num: self.batch_size,
                                                    self.vae_option: 1})
                    # make sure gate and final fc are non-negative
                    #sess.run(self.clip)

                    # check points
                    if b % 50 == 0:
                        eval_loss = sess.run(loss, feed_dict={x_input: batch_original_X,
                                                    tau_input: batch_tau,
                                                    #target: batch_y[:, -1][:, np.newaxis],
                                                    #target_taus: batch_y[:, :-1],
                                                    target: batch_y,
                                                    self.learning_rate_nn: learning_rate_nn_,
                                                    self.bn_phase: 0,
                                                    self.keep_prob: 1.0,
                                                    self.input_num: self.batch_size,
                                                    self.vae_option: 1})
                        print('Epoch: {}, batch: {}, loss: {}'.format(i, b, eval_loss))

                        # write training log
                        #train_writer.add_summary(summary, global_step=step)
                    step += 1

                if i == epoch_decay_start:
                    learning_rate_nn_ /= 4.0
                if i > epoch_decay_start and (i % decay_step == 0):
                    learning_rate_nn_ = learning_rate_nn_ * (decay_rate ** ((i - epoch_decay_start) / decay_step))

                # save the model
                if i % 100 == 0 or ((i + 1) == self.epochs):
                    saver.save(sess, save_path=self.model_file, global_step=i)

                # evaluate for testing data
                if i % 10 == 0 or ((i + 1) == self.epochs):

                    # test !!!
                    # split original X, dimreduce X, and threshold
                    n_batch_test = int(test_X.shape[0] / self.batch_size) + 1
                    for b_ in range(n_batch_test):
                        batch_test_original_X, batch_test_tau, batch_test_y = self.getBatch_(b_,
                                        self.batch_size, test_X, test_tau, test_y)
                        predictions_batch = sess.run(predictions_tensor,
                                        feed_dict={x_input: batch_test_original_X,
                                                    tau_input: batch_test_tau,
                                                    self.bn_phase: 0,
                                                    self.keep_prob: 1.0,
                                                    self.input_num: self.batch_size,
                                                    self.vae_option: 0})
                        if b_ == 0:
                            predictions = predictions_batch
                        else:
                            predictions = np.concatenate((predictions, predictions_batch), axis=0)
                    # clip
                    predictions = predictions[:test_X.shape[0]]

                    # check whether log is used
                    if self.log_option:
                        predictions = np.exp(predictions)
                    predictions = np.hstack(predictions)
                    test_y_labels = np.hstack(test_y)

                    #predictions = np.array(predictions, dtype=np.float64)
                    #test_y_labels = np.array(test_y_labels, dtype=np.float64)
                    print('Test Epoch: {}, loss: {}'.format(i, __eval__(predictions, test_y_labels)))

                    # save to files
                    L = [[i_, j_] for i_, j_ in zip(predictions, test_y_labels)]
                    L = np.array(L)
                    #save_file = './test/test_customDNN_predictions_labels_epoch_' + str(i)
                    save_file = self.test_data_predictions_labels_file + str(i)
                    np.save(save_file, L)

                    # valid !!!
                    # split original X, dimreduce X, and threshold
                    n_batch_valid = int(valid_X.shape[0] / self.batch_size) + 1
                    for b_ in range(n_batch_valid):
                        batch_valid_original_X, batch_valid_tau, batch_valid_y = self.getBatch_(b_,
                                        self.batch_size, valid_X, valid_tau, valid_y)
                        valid_predictions_batch = sess.run(predictions_tensor,
                                        feed_dict={x_input: batch_valid_original_X,
                                                    tau_input: batch_valid_tau,
                                                    self.bn_phase: 0,
                                                    self.keep_prob: 1.0,
                                                    self.input_num: self.batch_size,
                                                    self.vae_option: 0})
                        if b_ == 0:
                            valid_predictions = valid_predictions_batch
                        else:
                            valid_predictions = np.concatenate((valid_predictions, valid_predictions_batch), axis=0)
                    # clip
                    valid_predictions = valid_predictions[:valid_X.shape[0]]

                    # check whether log is used
                    if self.log_option:
                        valid_predictions = np.exp(valid_predictions)
                    valid_predictions = np.hstack(valid_predictions)
                    valid_y_labels = np.hstack(valid_y)

                    #predictions = np.array(predictions, dtype=np.float64)
                    #test_y_labels = np.array(test_y_labels, dtype=np.float64)
                    print('Valid Epoch: {}, loss: {}'.format(i, __eval__(valid_predictions, valid_y_labels)))

                    # save to files
                    L = [[i_, j_] for i_, j_ in zip(valid_predictions, valid_y_labels)]
                    L = np.array(L)
                    #save_file = './test/test_customDNN_predictions_labels_epoch_' + str(i)
                    save_file = self.valid_data_predictions_labels_file + str(i)
                    np.save(save_file, L)



    def train(self, train_X, train_tau, train_y, valid_X, valid_tau, valid_y, test_X, test_tau, test_y):
        if self.spline_type == 'selnet_linear':
            self.train_vae_dnn(train_X, train_tau, train_y, valid_X, valid_tau, valid_y)
        elif self.spline_type == 'selnet_quad':
            self.train_vae_dnn_quad(train_X, train_tau, train_y, valid_X, valid_tau, valid_y, test_X, test_tau, test_y)
        elif self.spline_type == 'cardnet':
            pass
        else:
            raise ValueError('wrong spline type')


    def getBatch_vae(self, batch_id, batch_size, X):
        train_num = X.shape[0]
        start_index = (batch_id * batch_size) % train_num
        end_index = start_index + batch_size

        batch_x = X[start_index: end_index]

        if batch_x.shape[0] < batch_size:
            L = batch_size - batch_x.shape[0]
            batch_x = np.concatenate((batch_x, X[:L]), axis=0)

        return batch_x

    def getBatch(self, batch_id, batch_size, trainFeatures, trainTauGate, trainLabels):
        train_num = trainFeatures.shape[0]
        start_index = (batch_id * batch_size) % train_num
        end_index = start_index + batch_size

        batch_X = trainFeatures[start_index: end_index]
        batch_tau_gate = trainTauGate[start_index: end_index]
        batch_y = trainLabels[start_index: end_index]

        if batch_X.shape[0] < batch_size:
            ''' If reach the end of data
            '''
            L = batch_size - batch_X.shape[0]
            batch_X = np.concatenate((batch_X, trainFeatures[:L]), axis=0)
            batch_tau_gate = np.concatenate((batch_tau_gate, trainTauGate[:L]), axis=0)
            batch_y = np.concatenate((batch_y, trainLabels[:L]), axis=0)

        # if Y is a vector, transfer to [None, 1]
        if len(batch_y.shape) <= 1:
            batch_y = batch_y[:, np.newaxis]

        return batch_X, batch_tau_gate, batch_y


    def getBatch_(self, batch_id, batch_size, trainFeatures, trainTau, trainLabels):
        train_num = trainFeatures.shape[0]
        start_index = (batch_id * batch_size) % train_num
        end_index = start_index + batch_size

        batch_X = trainFeatures[start_index: end_index]
        batch_tau = trainTau[start_index: end_index]
        batch_y = trainLabels[start_index: end_index]

        if batch_X.shape[0] < batch_size:
            ''' If reach the end of data
            '''
            L = batch_size - batch_X.shape[0]
            batch_X = np.concatenate((batch_X, trainFeatures[:L]), axis=0)
            batch_tau = np.concatenate((batch_tau, trainTau[:L]), axis=0)
            batch_y = np.concatenate((batch_y, trainLabels[:L]), axis=0)

        # if Y is a vector, transfer to [None, 1]
        if len(batch_y.shape) <= 1:
            batch_y = batch_y[:, np.newaxis]

        return batch_X, batch_tau, batch_y


