import sys
import numpy as np
import tensorflow as tf
import math
import os
import pickle

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
from Dispatcher import Dispatcher

def mean_absolute_percentage_error(labels, predictions):
    return np.mean(np.abs((predictions - labels) * 1.0 / (labels + 0.000001))) * 100

def __eval__(predictions, labels):
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    mape = mean_absolute_percentage_error(labels, predictions)
    return (mse, mae, mape)


class SelNetPart(object):
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
		leaf_num,
                partition_option,
                loss_option):
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
        self.tau_part_num = tau_part_num
        self.unit_len = unit_len # default = 1
        self.gate_layer = self.unit_len * self.tau_part_num

        self.hidden_num = len(hidden_units)
        # assume the unit_len is divisible by hidden_num
        self.hidden_unit_len = self.unit_len // self.hidden_num

        self.partition_option = partition_option
        self.leaf_num = leaf_num
        self.loss_option = loss_option

        self.spline_type = 'selnet_linear'


    # change spline type to quad
    def change_spline_type(self, spline_type):
        self.spline_type = spline_type


    def expert_model(self, x_input, x_input_dr, tau_gate, mapping, target_leaf, expert_name, expert_id):
        ''' One expert deals with one leaf.
        '''
        predictions_tensor, gate_tensor = self._construct_model(x_input, x_input_dr, tau_gate, expert_name)
        predictions_tensor = tf.multiply(predictions_tensor, mapping[:, expert_id: expert_id + 1])
        if self.loss_option == 'msle':
            loss = tf.losses.mean_squared_error(predictions=tf.log(predictions_tensor + 1),
                                                   labels=tf.log(target_leaf + 1))
        elif self.loss_option == 'huber':
            loss = tf.losses.huber_loss(labels=target_leaf, predictions=predictions_tensor, delta=1.345)
        elif self.loss_option == 'huber_log':
            loss = tf.losses.huber_loss(labels=tf.log(target_leaf + 1), predictions=tf.log(predictions_tensor + 1), delta=1.345)
        else:
            raise ValueError('Wrong Loss Function Option')

        return loss, predictions_tensor


    def expert_model_quad(self, x_input, x_input_dr, tau_gate, mapping, target_leaf, expert_name, expert_id, trid):
        ''' One expert deals with one leaf using quad spline
        '''
        predictions_tensor, gate_tensor = self._construct_model_quadratic(x_input, x_input_dr, tau_gate, trid, expert_name)
        predictions_tensor = tf.multiply(predictions_tensor, mapping[:, expert_id: expert_id + 1])
        if self.loss_option == 'msle':
            loss = tf.losses.mean_squared_error(predictions=tf.log(predictions_tensor + 1),
                                                   labels=tf.log(target_leaf + 1))
        elif self.loss_option == 'huber':
            loss = tf.losses.huber_loss(labels=target_leaf, predictions=predictions_tensor, delta=1.345)
        elif self.loss_option == 'huber_log':
            loss = tf.losses.huber_loss(labels=tf.log(target_leaf + 1), predictions=tf.log(predictions_tensor + 1), delta=1.345)
        else:
            raise ValueError('Wrong Loss Function Option')

        return loss, predictions_tensor


    def __ae__(self, x_input, expert_name):
        '''
        Transfer original X to dense representation
        :param x_input:
        :return:
        '''
        # Encoder
        fc1 = tf.layers.dense(inputs=x_input, units=self.vae_hidden_units[0],
                              activation=tf.nn.elu, name=self.regressor_name + 'vae_fc_e1_' +  expert_name)
        fc2 = tf.layers.dense(inputs=fc1, units=self.vae_hidden_units[1],
                              activation=tf.nn.elu, name=self.regressor_name + 'vae_fc_e2_' + expert_name)
        fc3 = tf.layers.dense(inputs=fc2, units=self.vae_hidden_units[2],
                              activation=tf.nn.elu, name=self.regressor_name + 'vae_fc_e3_' + expert_name)

        # generate hidden layer z
        z_mu = tf.layers.dense(inputs=fc3, units=self._vae_n_z,
                            activation=None, name=self.regressor_name + 'vae_fc_e4_' + expert_name)

        # hidden layer
        hidden_z = z_mu #+ tf.sqrt(tf.exp(z_log_sigma_sq)) * eps #* self.vae_option

        # Decoder
        g1 = tf.layers.dense(inputs=hidden_z, units=self.vae_hidden_units[2],
                             activation=tf.nn.elu, name=self.regressor_name + 'vae_fc_d1_' + expert_name)
        g2 = tf.layers.dense(inputs=g1, units=self.vae_hidden_units[1],
                             activation=tf.nn.elu, name=self.regressor_name + 'vae_fc_d2_' + expert_name)
        g3 = tf.layers.dense(inputs=g2, units=self.vae_hidden_units[0],
                             activation=tf.nn.elu, name=self.regressor_name + 'vae_fc_d3_' + expert_name)

        x_hat = tf.layers.dense(inputs=g3, units=self.original_x_dim,
                                activation=tf.nn.relu)

        recon_loss = tf.losses.mean_squared_error(predictions=x_hat, labels=x_input)
        recon_loss = tf.reduce_mean(recon_loss)

        ae_loss = recon_loss
        return ae_loss, hidden_z

    def _construct_rhos(self, x_fea, x_fea_dr, expert_name):
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
                              activation=tf.nn.relu, name=self.regressor_name + 'fc_1_' + expert_name)

        # 1st embedding
        rho_1 = tf.layers.dense(inputs=out, units=self.hidden_unit_len * (self.tau_part_num + 1),
                            activation=tf.nn.relu, name=self.regressor_name + 'embed_1' + expert_name)

        # reshape 1st embedding
        rho_1 = tf.reshape(rho_1, [-1, self.tau_part_num + 1, self.hidden_unit_len])

        rhos.append(rho_1)

        out = tf.layers.dense(inputs=out, units=self.hidden_units[1],
                              activation=tf.nn.relu, name=self.regressor_name + 'fc_2_' + expert_name)

        # 2nd embedding
        rho_2 = tf.layers.dense(inputs=out, units=self.hidden_unit_len * (self.tau_part_num + 1),
                            activation=tf.nn.relu, name=self.regressor_name + 'embed_2_' + expert_name)

        # reshape 2nd embedding
        rho_2 = tf.reshape(rho_2, [-1, self.tau_part_num + 1, self.hidden_unit_len])

        rhos.append(rho_2)

        out = tf.layers.dense(inputs=out, units=self.hidden_units[2],
                              activation=tf.nn.relu, name=self.regressor_name + 'fc_3_' + expert_name)
        # out = tf.nn.dropout(out, keep_prob=self.keep_prob, name=self.regressor_name + 'dropout')

        # 3rd embedding
        rho_3 = tf.layers.dense(inputs=out, units=self.hidden_unit_len * (self.tau_part_num + 1),
                            activation=tf.nn.relu, name=self.regressor_name + 'embed_3_' + expert_name)

        # reshape 3rd embedding
        rho_3 = tf.reshape(rho_3, [-1, self.tau_part_num + 1, self.hidden_unit_len])

        rhos.append(rho_3)

        out = tf.layers.dense(inputs=out, units=self.hidden_units[3],
                              activation=tf.nn.relu, name=self.regressor_name + 'fc_4_' + expert_name)

        # 4th embedding
        rho_4 = tf.layers.dense(inputs=out, units=self.hidden_unit_len * (self.tau_part_num + 1),
                            activation=tf.nn.relu, name=self.regressor_name + 'embed_4_' + expert_name)

        # reshape 3rd embedding
        rho_4 = tf.reshape(rho_4, [-1, self.tau_part_num + 1, self.hidden_unit_len])

        rhos.append(rho_4)

        # concate embedding matrix RHO
        gate = rhos[0]
        for hidden_id in range(1, 4, 1):
            gate = tf.concat([gate, rhos[hidden_id]], 2)

        return gate

    def _partition_threshold(self, x_fea, x_fea_dr, tau, expert_name, eps=0.0000001):
        new_x = tf.concat([x_fea, x_fea_dr], 1)
        out = tf.layers.dense(inputs=new_x, units=self.hidden_units[0], activation=tf.nn.elu,
                                name=self.regressor_name + 'tau_part_1' + expert_name)
        out = tf.layers.dense(inputs=out, units=self.hidden_units[1], activation=tf.nn.elu,
                                name=self.regressor_name + 'tau_part_2' + expert_name)

        out = tf.layers.dense(inputs=out, units=self.tau_part_num, activation=tf.nn.elu,
                                name=self.regressor_name + 'tau_part_3' + expert_name)

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
        residue_tau_s = residue_tau[:, 1:]
        residue_tau_s = tf.concat([residue_tau_s, tf.expand_dims(tf.zeros(self.input_num), axis=1)], 1)

        precent_tau = tf.divide(tf.nn.relu(residue_tau - residue_tau_s), dist_tau * self.max_tau)

        precent_tau = tf.concat([tf.expand_dims(tf.ones(self.input_num), axis=1), precent_tau], 1)

        return precent_tau

    def _construct_model(self, x_fea, x_fea_dr, tau, expert_name):

        gate = self._construct_rhos(x_fea, x_fea_dr, expert_name)

        # integrate
        w_t = tf.get_variable(self.regressor_name + 'w_t_' + expert_name, [self.tau_part_num + 1, self.unit_len], tf.float32)
        b_t = tf.get_variable(self.regressor_name + 'b_t_' + expert_name, [self.tau_part_num + 1, self.unit_len], tf.float32)
        gate = tf.nn.relu(tf.multiply(gate, w_t) + b_t)

        # conv with mask
        kernel_ = tf.ones([self.unit_len], dtype=tf.float32, name=self.regressor_name + 'k_' + expert_name)
        kernel = tf.reshape(kernel_, [1, int(kernel_.shape[0]), 1], name=self.regressor_name + 'kernel_' + expert_name)
        # reshape layer
        #gate = tf.reshape(gate, [-1, self.gate_layer, 1], name=self.regressor_name + 'gate_v1')
        gate = tf.nn.relu(tf.squeeze(tf.nn.conv1d(gate, kernel, 1, 'VALID')) )

        # narrow down the domain of Delta Y
        tau_gate = self._partition_threshold(x_fea, x_fea_dr, tau, expert_name)
        gate = tf.multiply(gate, tau_gate)

        prediction = tf.reduce_sum(gate, 1)
        # expand dim
        prediction = tf.expand_dims(prediction, 1)

        return prediction, gate


    def _partition_threshold_quadratic(self, x_fea, x_fea_dr, expert_name, eps=1e-6):
       # first concat x
        new_x = tf.concat([x_fea, x_fea_dr], 1)
        out = tf.layers.dense(inputs=new_x, units=self.hidden_units[0], activation=tf.nn.elu,
                              name=self.regressor_name + 'tau_part_1_' + expert_name)
        out = tf.layers.dense(inputs=out, units=self.hidden_units[1], activation=tf.nn.elu,
                              name=self.regressor_name + 'tau_part_2_' + expert_name)

        out = tf.layers.dense(inputs=out, units=self.tau_part_num, activation=tf.nn.elu,
                              name=self.regressor_name + 'tau_part_3_' + expert_name)

        if self.partition_option == 'softmax':
            dist_tau = tf.nn.softmax(out)
        elif self.partition_option == 'l2':
            out = tf.multiply(out, out) + eps
            norm = tf.expand_dims(tf.reduce_sum(out, 1), 1)
            norm = tf.tile(norm, [1, self.tau_part_num])
            dist_tau = tf.truediv(out, norm)
        else:
            raise ValueError('wrong partition option')

        partition_tau = tf.cumsum(dist_tau, 1) * self.max_tau
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


    def _construct_model_quadratic(self, x_fea, x_fea_dr, tau, trid, expert_name):
        ''' quad spline
        '''
        gate = self._construct_rhos(x_fea, x_fea_dr, expert_name)

        # integrate
        w_t = tf.get_variable(self.regressor_name + 'w_t_' + expert_name, [self.tau_part_num + 1, self.unit_len], tf.float32)
        b_t = tf.get_variable(self.regressor_name + 'b_t_' + expert_name, [self.tau_part_num + 1, self.unit_len], tf.float32)
        gate = tf.nn.relu(tf.multiply(gate, w_t) + b_t)

        # conv with mask
        kernel_ = tf.ones([self.unit_len], dtype=tf.float32, name=self.regressor_name + 'k_' + expert_name)
        kernel = tf.reshape(kernel_, [1, int(kernel_.shape[0]), 1], name=self.regressor_name + 'kernel_' + expert_name)
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

        dist_tau, partition_tau = self._partition_threshold_quadratic(x_fea, x_fea_dr, expert_name)
        H = dist_tau * self.max_tau
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

    def predict_vae_dnn(self, test_X, test_mapping, test_tau_gate):
        ''' Prediction
        '''
        x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.original_x_dim], name=self.regressor_name + 'original_X')


        tau_input = tf.placeholder(dtype=tf.float32, shape=[None, self.tau_part_num], name=self.regressor_name + 'tau_gate')
        mapping = tf.placeholder(dtype=tf.float32, shape=[None, self.leaf_num], name=self.regressor_name + 'mapping')
        init_indices = tf.placeholder(dtype=tf.int32, shape=[None], name=self.regressor_name + 'init_indices')
        targets = tf.placeholder(dtype=tf.float32, shape=[None, self.leaf_num + 1], name=self.regressor_name + 'Target')
        self.bn_phase = tf.placeholder(dtype=tf.bool, name=self.regressor_name + 'Phase')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name=self.regressor_name + 'Dropout')
        # input number
        self.input_num = tf.placeholder(dtype=tf.int32, name=self.regressor_name + 'input_num')

        # VAE inference or training
        self.vae_option = tf.placeholder(dtype=tf.float32, name=self.regressor_name + 'VAE_Option')

        _, x_input_dr = self.__ae__(x_input, "AE")

        # deal with 1st Expert
        expert_name = '_Expert_0'
        _, prediction_expert = self.expert_model(x_input, x_input_dr, tau_input, mapping, targets[:, 0: 1], expert_name, 0)
        predictions_tensor = prediction_expert

        for lid in range(1, self.leaf_num, 1):
            expert_name = '_Expert_' + str(lid)
            _, prediction_expert = self.expert_model(x_input, x_input_dr, tau_input, mapping,
                                                                targets[:, lid: lid + 1], expert_name, lid)
            predictions_tensor += prediction_expert


        # tensorflow saver
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_file)
            # make prediction
            startTime = timer()
            n_batch_test = int(test_X.shape[0] / self.batch_size) + 1
            for b_ in range(n_batch_test):
                batch_test_original_X, batch_test_mapping, batch_test_tau_gate = self.getBatch_test(
                        b_, self.batch_size, test_X[:, :self.original_x_dim], test_mapping, test_tau_gate)
                batch_test_init_indices = np.zeros(self.batch_size, dtype=np.int32)
                predictions_batch = sess.run(predictions_tensor,
                                        feed_dict={x_input: batch_test_original_X,
                                            init_indices: batch_test_init_indices,
                                            mapping: batch_test_mapping,
                                            tau_input: batch_test_tau_gate,
                                            self.vae_option: 0.0,
                                            self.input_num: self.batch_size})
                if b_ == 0:
                    predictions = predictions_batch
                else:
                    predictions = np.concatenate((predictions, predictions_batch), axis=0)

            # clip
            predictions = predictions[:test_X.shape[0]]


            if self.log_option:
                predictions = np.hstack(predictions)
                predictions = np.exp(predictions)

            self.prediction_time += timer() - startTime

        return predictions

    def train_vae_dnn(self, train_X, train_mapping, train_tau_gate, train_y,
            valid_X, valid_mapping, valid_tau_gate, valid_y):

        tf.reset_default_graph()

        x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.original_x_dim], name=self.regressor_name + 'original_X')

        # tau_input is an one-hot vector -- maybe useless
        tau_input = tf.placeholder(dtype=tf.float32, shape=[None, self.tau_part_num], name=self.regressor_name + 'tau')

        # mapping
        mapping = tf.placeholder(dtype=tf.float32, shape=[None, self.leaf_num], name=self.regressor_name + 'mapping')

        # init indices of threshold one-hot
        init_indices = tf.placeholder(dtype=tf.int32, shape=[None], name=self.regressor_name + 'init_indices')

        targets = tf.placeholder(dtype=tf.float32, shape=[None, self.leaf_num + 1], name=self.regressor_name + 'Targets')

        # to control option of batch normalization
        self.bn_phase = tf.placeholder(dtype=tf.bool, name=self.regressor_name + 'Phase')

        # add dropout to avoid overfitting
        self.keep_prob = tf.placeholder(dtype=tf.float32, name=self.regressor_name + 'Dropout')

        # input number
        self.input_num = tf.placeholder(dtype=tf.int32, name=self.regressor_name + 'input_num')

        # VAE inference or training
        self.vae_option = tf.placeholder(dtype=tf.float32, name=self.regressor_name + 'VAE_Option')

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

        loss = 0
        vae_loss, x_input_dr = self.__ae__(x_input, "AE")

        # deal with 1st Expert
        expert_name = '_Expert_0'
        loss_expert, prediction_expert = self.expert_model(x_input, x_input_dr, tau_input, mapping, targets[:, 0: 1], expert_name, 0)
        predictions_tensor = prediction_expert
        loss += loss_expert

        for lid in range(1, self.leaf_num, 1):
            expert_name = '_Expert_' + str(lid)
            loss_expert, prediction_expert = self.expert_model(x_input, x_input_dr, tau_input, mapping,
                                                                targets[:, lid: lid + 1], expert_name, lid)
            predictions_tensor += prediction_expert
            loss += loss_expert

        if self.loss_option == 'msle':
            loss_one = 0.01 * loss + \
                    tf.losses.mean_squared_error(predictions=tf.log(predictions_tensor + 1),
                                                labels=tf.log(targets[:, -1:] + 1))
        elif self.loss_option == 'huber':
            loss_one = 0.01 * loss + \
                    tf.losses.huber_loss(labels=targets[:, -1:], predictions=predictions_tensor, delta=1.345)
        elif self.loss_option == 'huber_log':
            loss_one = 0.01 * loss + \
                    tf.losses.huber_loss(labels=tf.log(targets[:, -1:] + 1), predictions=tf.log(predictions_tensor + 1), delta=1.345)
        else:
            raise ValueError('Wrong Loss Function Option')

        optimizer_vae = tf.train.AdamOptimizer(self.learning_rate_vae).minimize(vae_loss)
        optimizer_expert = tf.train.AdamOptimizer(self.learning_rate_nn).minimize(loss)
        optimizer_one = tf.train.AdamOptimizer(self.learning_rate_nn).minimize(loss_one)

        # init all variables
        init = tf.global_variables_initializer()

        # Save the model
        saver = tf.train.Saver()
        step = 0

        session_config = tf.ConfigProto(log_device_placement=True)
        session_config.gpu_options.allow_growth = True
        with tf.Session(config=session_config) as sess:
            #with tf.Session() as sess:
            # writer for events and checkpoints

            sess.run(init)

            # 1. first initialize VAE (unsupervised)
            learning_rate_vae_ = self.learning_rate
            decay_rate, decay_step = 0.96, 5
            for epoch in range(self.epochs_vae):
                n_batches = int(train_X.shape[0] / self.batch_size) + 1
                if epoch != 0 and (epoch % decay_step == 0):
                    learning_rate_vae_ = learning_rate_vae_ * (decay_rate ** (epoch / decay_step))
                for b in range(n_batches):
                    batch_X, batch_mapping = self.getBatch_vae(b, self.batch_size, train_X, train_mapping)
                    sess.run(optimizer_vae, feed_dict={x_input: batch_X,
                                            mapping: batch_mapping,
                                            self.learning_rate_vae: learning_rate_vae_})

                    # check points
                    if b % 50 == 0:
                        eval_loss = sess.run(vae_loss, feed_dict={x_input: batch_X,
                                                                    mapping: batch_mapping,
                                                                    self.learning_rate_vae: learning_rate_vae_})
                        print('VAE Epoch: {}, batch: {}, loss: {}'.format(epoch, b, eval_loss))

            # 2. fit all the training data to estimate
            learning_rate_nn_ = self.learning_rate
            epoch_decay_start = 200
            epoch_one = 100
            decay_rate, decay_step = 0.96, 10
            for i in range(self.epochs):
                n_batches = int(train_X.shape[0] / self.batch_size) + 1
                for b in range(n_batches):
                    # get current batch
                    batch_original_X, batch_mapping, batch_tau_gate, batch_y = self.getBatch_(b,
                            self.batch_size, train_X, train_mapping, train_tau_gate, train_y)
                    batch_init_indices = np.zeros(self.batch_size, dtype=np.int32)
                    if i < epoch_one:
                        sess.run(optimizer_expert, feed_dict={x_input: batch_original_X,
                                                        init_indices: batch_init_indices,
                                                        mapping: batch_mapping,
                                                        tau_input: batch_tau_gate,
                                                        targets: batch_y,
                                                        self.learning_rate_nn: learning_rate_nn_,
                                                        self.keep_prob: 0.9,
						        self.vae_option: 1.0,
                                                        self.input_num: self.batch_size})
                    else:
                        sess.run(optimizer_one, feed_dict={x_input: batch_original_X,
                                                           init_indices: batch_init_indices,
                                                           mapping: batch_mapping,
                                                           tau_input: batch_tau_gate,
                                                           targets: batch_y,
                                                           self.learning_rate_nn: learning_rate_nn_,
                                                           self.vae_option: 1.0,
                                                           self.input_num: self.batch_size})

                    # check points
                    if b % 50 == 0:
                        [eval_loss, eval_loss_one] = sess.run([loss, loss_one], feed_dict={x_input: batch_original_X,
                                                                init_indices: batch_init_indices,
                                                                mapping: batch_mapping,
                                                                tau_input: batch_tau_gate,
                                                                targets: batch_y,
                                                                self.learning_rate_nn: learning_rate_nn_,
                                                                self.keep_prob: 1.0,
						                self.vae_option: 1.0,
                                                                self.input_num: self.batch_size})
                        print('Epoch: {}, batch: {}, loss: {}'.format(i, b, eval_loss))

                    step += 1

                if i == epoch_decay_start:
                    learning_rate_nn_ /= 4.0
                if i > epoch_decay_start and (i % decay_step == 0):
                    learning_rate_nn_ = learning_rate_nn_ * (decay_rate ** ((i - epoch_decay_start) / decay_step))

                # save the model
                if i % 100 == 0 or ((i + 1) == self.epochs):
                    saver.save(sess, save_path=self.model_file, global_step=i)


                # evaluate for testing data
                if (i % 10 == 0) or ((i + 1) == self.epochs):
                    '''
                    # test !!!
                    # split original X, dimreduce X, and threshold
                    n_batch_test = int(test_X.shape[0] / self.batch_size) + 1
                    for b_ in range(n_batch_test):
                        batch_test_original_X, batch_test_mapping, batch_test_tau_gate, _ = self.getBatch_(
                                b_, self.batch_size, test_X[:, :self.original_x_dim], test_mapping, test_tau_gate, test_y)
                        batch_test_init_indices = np.zeros(self.batch_size, dtype=np.int32)
                        predictions_batch = sess.run(predictions_tensor,
                                                feed_dict={x_input: batch_test_original_X,
                                                    init_indices: batch_test_init_indices,
                                                    mapping: batch_test_mapping,
                                                    tau_input: batch_test_tau_gate,
                                                    self.vae_option: 0.0,
                                                    self.input_num: self.batch_size})
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

                    print('Test Epoch: {}, loss: {}'.format(i, __eval__(predictions, test_y_labels)))

                    # save to files
                    L = [[i_, j_] for i_, j_ in zip(predictions, test_y_labels)]
                    L = np.array(L)
                    save_file = self.test_data_predictions_labels_file + str(i)
                    np.save(save_file, L)
                    '''
                    # valid !!!
                    # split original X, dimreduce X and threshold
                    n_batch_valid = int(valid_X.shape[0] / self.batch_size) + 1
                    for b_ in range(n_batch_valid):
                        batch_valid_original_X, batch_valid_mapping, batch_valid_tau_gate, _ = self.getBatch_(
                                b_, self.batch_size, valid_X[:, :self.original_x_dim], valid_mapping, valid_tau_gate, valid_y)
                        batch_valid_init_indices = np.zeros(self.batch_size, dtype=np.int32)
                        valid_predictions_batch = sess.run(predictions_tensor,
                                                feed_dict={x_input: batch_valid_original_X,
                                                    init_indices: batch_valid_init_indices,
                                                    mapping: batch_valid_mapping,
                                                    tau_input: batch_valid_tau_gate,
                                                    self.vae_option: 0.0,
                                                    self.input_num: self.batch_size})
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

                    print('Valid Epoch: {}, loss: {}'.format(i, __eval__(valid_predictions, valid_y_labels)))

                    # save to files
                    L = [[i_, j_] for i_, j_ in zip(valid_predictions, valid_y_labels)]
                    L = np.array(L)

                    save_file = self.valid_data_predictions_labels_file + str(i)
                    np.save(save_file, L)


    def train_vae_dnn_quad(self, train_X, train_mapping, train_tau_gate, train_y,
            valid_X, valid_mapping, valid_tau_gate, valid_y, test_X, test_mapping, test_tau_gate, test_y):

        tf.reset_default_graph()

        x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.original_x_dim], name=self.regressor_name + 'original_X')

        # tau_input is an one-hot vector -- maybe useless
        # tau_input = tf.placeholder(dtype=tf.float32, shape=[None, self.tau_part_num], name=self.regressor_name + 'tau')
        tau_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name=self.regressor_name + 'tau')

        # mapping
        mapping = tf.placeholder(dtype=tf.float32, shape=[None, self.leaf_num], name=self.regressor_name + 'mapping')

        # init indices of threshold one-hot
        init_indices = tf.placeholder(dtype=tf.int32, shape=[None], name=self.regressor_name + 'init_indices')

        targets = tf.placeholder(dtype=tf.float32, shape=[None, self.leaf_num + 1], name=self.regressor_name + 'Targets')

        # to control option of batch normalization
        self.bn_phase = tf.placeholder(dtype=tf.bool, name=self.regressor_name + 'Phase')

        # add dropout to avoid overfitting
        self.keep_prob = tf.placeholder(dtype=tf.float32, name=self.regressor_name + 'Dropout')

        # input number
        self.input_num = tf.placeholder(dtype=tf.int32, name=self.regressor_name + 'input_num')

        # VAE inference or training
        self.vae_option = tf.placeholder(dtype=tf.float32, name=self.regressor_name + 'VAE_Option')

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

        loss = 0
        vae_loss, x_input_dr = self.__ae__(x_input, "AE")

        # calculate dig
        trid = self.tridiagonal(self.tau_part_num + 1)

        # deal with 1st Expert
        expert_name = '_Expert_0'
        loss_expert, prediction_expert = self.expert_model_quad(x_input, x_input_dr, tau_input, mapping, targets[:, 0: 1], expert_name, 0, trid)
        predictions_tensor = prediction_expert
        loss += loss_expert

        for lid in range(1, self.leaf_num, 1):
            expert_name = '_Expert_' + str(lid)
            loss_expert, prediction_expert = self.expert_model_quad(x_input, x_input_dr, tau_input, mapping,
                                                                targets[:, lid: lid + 1], expert_name, lid, trid)
            predictions_tensor += prediction_expert
            loss += loss_expert

        if self.loss_option == 'msle':
            loss_one = 0.01 * loss + \
                    tf.losses.mean_squared_error(predictions=tf.log(predictions_tensor + 1),
                                                labels=tf.log(targets[:, -1:] + 1))
        elif self.loss_option == 'huber':
            loss_one = 0.01 * loss + \
                    tf.losses.huber_loss(labels=targets[:, -1:], predictions=predictions_tensor, delta=1.345)
        elif self.loss_option == 'huber_log':
            loss_one = 0.01 * loss + \
                    tf.losses.huber_loss(labels=tf.log(targets[:, -1:] + 1), predictions=tf.log(predictions_tensor + 1), delta=1.345)
        else:
            raise ValueError('Wrong Loss Function Option')

        optimizer_vae = tf.train.AdamOptimizer(self.learning_rate_vae).minimize(vae_loss)
        optimizer_expert = tf.train.AdamOptimizer(self.learning_rate_nn).minimize(loss)
        optimizer_one = tf.train.AdamOptimizer(self.learning_rate_nn).minimize(loss_one)

        # init all variables
        init = tf.global_variables_initializer()

        # Save the model
        saver = tf.train.Saver()
        step = 0

        session_config = tf.ConfigProto(log_device_placement=True)
        session_config.gpu_options.allow_growth = True
        with tf.Session(config=session_config) as sess:
            #with tf.Session() as sess:
            # writer for events and checkpoints

            sess.run(init)

            # 1. first initialize VAE (unsupervised)
            learning_rate_vae_ = self.learning_rate
            decay_rate, decay_step = 0.96, 5
            for epoch in range(self.epochs_vae):
                n_batches = int(train_X.shape[0] / self.batch_size) + 1
                if epoch != 0 and (epoch % decay_step == 0):
                    learning_rate_vae_ = learning_rate_vae_ * (decay_rate ** (epoch / decay_step))
                for b in range(n_batches):
                    batch_X, batch_mapping = self.getBatch_vae(b, self.batch_size, train_X, train_mapping)
                    sess.run(optimizer_vae, feed_dict={x_input: batch_X,
                                            mapping: batch_mapping,
                                            self.learning_rate_vae: learning_rate_vae_})

                    # check points
                    if b % 50 == 0:
                        eval_loss = sess.run(vae_loss, feed_dict={x_input: batch_X,
                                                                    mapping: batch_mapping,
                                                                    self.learning_rate_vae: learning_rate_vae_})
                        print('VAE Epoch: {}, batch: {}, loss: {}'.format(epoch, b, eval_loss))

            # 2. fit all the training data to estimate
            learning_rate_nn_ = self.learning_rate
            epoch_decay_start = 200
            epoch_one = 100
            decay_rate, decay_step = 0.96, 10
            for i in range(self.epochs):
                n_batches = int(train_X.shape[0] / self.batch_size) + 1
                # sample some batches
                batch_num_ratio = 0.3
                np.random.seed(i**2)
                sc = np.random.choice(n_batches, int(n_batches * batch_num_ratio), replace=False)
                batches_IDS = np.arange(n_batches)[sc]
                # for b in range(n_batches):
                for b in batches_IDs:
                    # get current batch
                    batch_original_X, batch_mapping, batch_tau_gate, batch_y = self.getBatch_(b,
                            self.batch_size, train_X, train_mapping, train_tau_gate, train_y)
                    batch_init_indices = np.zeros(self.batch_size, dtype=np.int32)
                    if i < epoch_one:
                        sess.run(optimizer_expert, feed_dict={x_input: batch_original_X,
                                                        init_indices: batch_init_indices,
                                                        mapping: batch_mapping,
                                                        tau_input: batch_tau_gate,
                                                        targets: batch_y,
                                                        self.learning_rate_nn: learning_rate_nn_,
                                                        self.keep_prob: 0.9,
						        self.vae_option: 1.0,
                                                        self.input_num: self.batch_size})
                    else:
                        sess.run(optimizer_one, feed_dict={x_input: batch_original_X,
                                                           init_indices: batch_init_indices,
                                                           mapping: batch_mapping,
                                                           tau_input: batch_tau_gate,
                                                           targets: batch_y,
                                                           self.learning_rate_nn: learning_rate_nn_,
                                                           self.vae_option: 1.0,
                                                           self.input_num: self.batch_size})

                    # check points
                    if b % 50 == 0:
                        [eval_loss, eval_loss_one] = sess.run([loss, loss_one], feed_dict={x_input: batch_original_X,
                                                                init_indices: batch_init_indices,
                                                                mapping: batch_mapping,
                                                                tau_input: batch_tau_gate,
                                                                targets: batch_y,
                                                                self.learning_rate_nn: learning_rate_nn_,
                                                                self.keep_prob: 1.0,
						                self.vae_option: 1.0,
                                                                self.input_num: self.batch_size})
                        print('Epoch: {}, batch: {}, loss: {}'.format(i, b, eval_loss))

                    step += 1

                if i == epoch_decay_start:
                    learning_rate_nn_ /= 4.0
                if i > epoch_decay_start and (i % decay_step == 0):
                    learning_rate_nn_ = learning_rate_nn_ * (decay_rate ** ((i - epoch_decay_start) / decay_step))

                # save the model
                if i % 100 == 0 or ((i + 1) == self.epochs):
                    saver.save(sess, save_path=self.model_file, global_step=i)


                # evaluate for testing data
                if (i % 10 == 0) or ((i + 1) == self.epochs):

                    # test !!!
                    # split original X, dimreduce X, and threshold
                    n_batch_test = int(test_X.shape[0] / self.batch_size) + 1
                    for b_ in range(n_batch_test):
                        batch_test_original_X, batch_test_mapping, batch_test_tau_gate, _ = self.getBatch_(
                                b_, self.batch_size, test_X[:, :self.original_x_dim], test_mapping, test_tau_gate, test_y)
                        batch_test_init_indices = np.zeros(self.batch_size, dtype=np.int32)
                        predictions_batch = sess.run(predictions_tensor,
                                                feed_dict={x_input: batch_test_original_X,
                                                    init_indices: batch_test_init_indices,
                                                    mapping: batch_test_mapping,
                                                    tau_input: batch_test_tau_gate,
                                                    self.vae_option: 0.0,
                                                    self.input_num: self.batch_size})
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

                    print('Test Epoch: {}, loss: {}'.format(i, __eval__(predictions, test_y_labels)))

                    # save to files
                    L = [[i_, j_] for i_, j_ in zip(predictions, test_y_labels)]
                    L = np.array(L)
                    save_file = self.test_data_predictions_labels_file + str(i)
                    np.save(save_file, L)

                    # valid !!!
                    # split original X, dimreduce X and threshold
                    n_batch_valid = int(valid_X.shape[0] / self.batch_size) + 1
                    for b_ in range(n_batch_valid):
                        batch_valid_original_X, batch_valid_mapping, batch_valid_tau_gate, _ = self.getBatch_(
                                b_, self.batch_size, valid_X[:, :self.original_x_dim], valid_mapping, valid_tau_gate, valid_y)
                        batch_valid_init_indices = np.zeros(self.batch_size, dtype=np.int32)
                        valid_predictions_batch = sess.run(predictions_tensor,
                                                feed_dict={x_input: batch_valid_original_X,
                                                    init_indices: batch_valid_init_indices,
                                                    mapping: batch_valid_mapping,
                                                    tau_input: batch_valid_tau_gate,
                                                    self.vae_option: 0.0,
                                                    self.input_num: self.batch_size})
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

                    print('Valid Epoch: {}, loss: {}'.format(i, __eval__(valid_predictions, valid_y_labels)))

                    # save to files
                    L = [[i_, j_] for i_, j_ in zip(valid_predictions, valid_y_labels)]
                    L = np.array(L)

                    save_file = self.valid_data_predictions_labels_file + str(i)
                    np.save(save_file, L)

    def train(self, train_X, train_mapping, train_tau_gate, train_y, valid_X, valid_mapping, valid_tau_gate, valid_y,
                        test_X, test_mapping, test_tau_gate, test_y):
        if self.spline_type == 'selnet_linear':
            self.train_vae_dnn(train_X, train_mapping, train_tau_gate, train_y, valid_X, valid_mapping, valid_tau_gate, valid_y)
        elif self.spline_type == 'selnet_quad':
            self.train_vae_dnn_quad(train_X, train_mapping, train_tau_gate, train_y, valid_X, valid_mapping, valid_tau_gate, valid_y,
                        test_X, test_mapping, test_tau_gate, test_y)
        elif self.spline_type == 'cardnet':
            pass
        else:
            raise ValueError('wrong spline type')

    def getBatch_vae(self, batch_id, batch_size, X, Mapping):
        train_num = X.shape[0]
        start_index = (batch_id * batch_size) % train_num
        end_index = start_index + batch_size

        batch_x = X[start_index: end_index]
        batch_mapping = Mapping[start_index: end_index]

        if batch_x.shape[0] < batch_size:
            L = batch_size - batch_x.shape[0]
            batch_x = np.concatenate((batch_x, X[:L]), axis=0)
            batch_mapping = np.concatenate((batch_mapping, Mapping[:L]), axis=0)

        return batch_x, np.array(batch_mapping, dtype=np.float32)

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


    def getBatch_(self, batch_id, batch_size, trainFeatures, trainMap, trainTauGate, trainLabels):
        train_num = trainFeatures.shape[0]
        start_index = (batch_id * batch_size) % train_num
        end_index = start_index + batch_size

        batch_X = trainFeatures[start_index: end_index]
        batch_mapping = trainMap[start_index: end_index]
        batch_tau_gate = trainTauGate[start_index: end_index]
        batch_y = trainLabels[start_index: end_index]

        if batch_X.shape[0] < batch_size:
            ''' If reach the end of data
            '''
            L = batch_size - batch_X.shape[0]
            batch_X = np.concatenate((batch_X, trainFeatures[:L]), axis=0)
            batch_tau_gate = np.concatenate((batch_tau_gate, trainTauGate[:L]), axis=0)
            batch_mapping = np.concatenate((batch_mapping, trainMap[:L]), axis=0)
            batch_y = np.concatenate((batch_y, trainLabels[:L]), axis=0)

        # if Y is a vector, transfer to [None, 1]
        if len(batch_y.shape) <= 1:
            batch_y = batch_y[:, np.newaxis]

        return batch_X, np.array(batch_mapping, dtype=np.float32), batch_tau_gate, batch_y


    def getBatch_test(self, batch_id, batch_size, trainFeatures, trainMap, trainTauGate):
        train_num = trainFeatures.shape[0]
        start_index = (batch_id * batch_size) % train_num
        end_index = start_index + batch_size

        batch_X = trainFeatures[start_index: end_index]
        batch_mapping = trainMap[start_index: end_index]
        batch_tau_gate = trainTauGate[start_index: end_index]

        if batch_X.shape[0] < batch_size:
            ''' If reach the end of data
            '''
            L = batch_size - batch_X.shape[0]
            batch_X = np.concatenate((batch_X, trainFeatures[:L]), axis=0)
            batch_tau_gate = np.concatenate((batch_tau_gate, trainTauGate[:L]), axis=0)
            batch_mapping = np.concatenate((batch_mapping, trainMap[:L]), axis=0)


        return batch_X, np.array(batch_mapping, dtype=np.float32), batch_tau_gate


