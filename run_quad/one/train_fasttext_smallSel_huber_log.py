import sys
import numpy as np
import os
import math

sys.path.append('../../model')
from selnet import *

loss_option = 'huber_log'
partition_option = 'l2'

test_file = '/data/yaoshuw/data_spherical/fasttext/smallSel/fasttext_testingDataL_smallSel-mixlabels.npy'
valid_file = '/data/yaoshuw/data_spherical/fasttext/smallSel/fasttext_validationDataL_smallSel-mixlabels.npy'
dataFile = '/data/yaoshuw/data_spherical/fasttext/smallSel/fasttext_trainingDataL_smallSel-mixlabels.npy'

'''
dataFile = '../../data/fasttext_eu/train/fasttext_eu_trainingData.npy'
test_file = '../../data/fasttext_eu/train/fasttext_eu_testingData.npy'
valid_file = '../../data/fasttext_eu/train/fasttext_eu_valdiationData.npy'
'''

x_dim = 300
x_reducedim = 80

train_data = np.load(dataFile)
 
test_data = np.load(test_file)
valid_data = np.load(valid_file)

tau_part_num = 50

train_original_X = np.array(train_data[:, :x_dim], dtype=np.float32)
train_tau_ = []
for rid in range(train_data.shape[0]):
    t = train_data[rid, x_dim] #hm_to_l2(train_data[rid, x_dim])
    train_tau_.append(t)

train_tau_ = np.array(train_tau_)
train_tau = np.zeros((train_data.shape[0], tau_part_num))
for cid in range(tau_part_num):
    train_tau[:, cid] = train_tau_

train_Y = np.array(train_data[:, -1], dtype=np.float32)

test_original_X = np.array(test_data[:, :x_dim], dtype=np.float32)
test_tau_ = []
for rid in range(test_data.shape[0]):
    t = test_data[rid, x_dim] #hm_to_l2(test_data[rid, x_dim])
    test_tau_.append(t)

test_tau_ = np.array(test_tau_)
test_tau = np.zeros((test_data.shape[0], tau_part_num))
for cid in range(tau_part_num):
    test_tau[:, cid] = test_tau_

test_Y = np.array(test_data[:, -1], dtype=np.float32)

valid_original_X = np.array(valid_data[:, :x_dim], dtype=np.float32)
valid_tau_ = []
for rid in range(valid_data.shape[0]):
    t = valid_data[rid, x_dim] #hm_to_l2(test_data[rid, x_dim])
    valid_tau_.append(t)

valid_tau_ = np.array(valid_tau_)
valid_tau = np.zeros((valid_data.shape[0], tau_part_num))
for cid in range(tau_part_num):
    valid_tau[:, cid] = valid_tau_

valid_Y = np.array(valid_data[:, -1], dtype=np.float32)


unit_len = 100
max_tau = 54.0

hidden_units = [1024, 512, 512, 512]
vae_hidden_units = [512, 256, 256]

batch_size = 512
epochs = 1500
epochs_vae = 10 #100
learning_rate = 0.0001
log_option = False
tau_embedding_size = 5
original_x_dim = train_original_X.shape[1]
dimreduce_x_dim = x_reducedim

test_data_predictions_labels_file = os.path.join('./test_fasttext_smallSel_huber_log/', 'test_predictions.npy')
valid_data_predictions_labels_file = os.path.join('./test_fasttext_smallSel_huber_log/', 'valid_predictions_labels_one_epoch_')
regression_name = 'fasttext_smallSel_huber_log_regressor_one_'
regression_model_dir = './model_dir_fasttext_smallSel_huber_log/regressor_one'


# train
regressor = SelNet(hidden_units, vae_hidden_units, batch_size, epochs, epochs_vae,
                         learning_rate, log_option, tau_embedding_size, original_x_dim, dimreduce_x_dim,
                         test_data_predictions_labels_file, valid_data_predictions_labels_file, regression_name, 
                         regression_model_dir, unit_len, max_tau, tau_part_num, partition_option, loss_option)


# change spline type to quad
regressor.change_spline_type('selnet_quad')
#regressor.change_spline_type('selnet_linear')

regressor.train(train_original_X, train_tau[:, :1], train_Y, valid_original_X, valid_tau[:, :1], valid_Y, test_original_X, test_tau[:, :1], test_Y)
#regressor.train(train_original_X, train_tau, train_Y, valid_original_X, valid_tau, valid_Y, test_original_X, test_tau, test_Y)


