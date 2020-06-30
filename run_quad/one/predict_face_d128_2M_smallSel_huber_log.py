import sys
import numpy as np
import os
import math

sys.path.append('../../model/')
from selnet import *

def eval_(predictions, labels):
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    mape = mean_absolute_percentage_error(labels, predictions)
    return (mse, mae, mape)


loss_option = 'huber_log'
partition_option = 'l2'

test_file = '../../data/face/train/face_testingData.npy'

x_dim = 128
x_reducedim = 80

test_data = np.load(test_file)

tau_part_num = 50

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


unit_len = 100
max_tau = 1 #54.0

hidden_units = [512, 512, 512, 256]
vae_hidden_units = [512, 256, 128]

batch_size = 512
epochs = 1500
epochs_vae = 100
learning_rate = 0.00003
log_option = False
tau_embedding_size = 5
original_x_dim = test_original_X.shape[1]
dimreduce_x_dim = x_reducedim

test_data_predictions_labels_file = os.path.join('./test_face_d128_2M_smallSel_huber_log/', 'test_predictions.npy')
valid_data_predictions_labels_file = os.path.join('./test_face_d128_2M_smallSel_huber_log/', 'valid_predictions_labels_one_epoch_')
regression_name = 'face_d128_2M_smallSel_huber_log_regressor_one_'
regression_model_dir = './model_dir_face_d128_2M_smallSel_huber_log/regressor_one-1499'


# train
regressor = SelNet(hidden_units, vae_hidden_units, batch_size, epochs, epochs_vae,
                         learning_rate, log_option, tau_embedding_size, original_x_dim, dimreduce_x_dim,
                         test_data_predictions_labels_file, valid_data_predictions_labels_file, regression_name, 
                         regression_model_dir, unit_len, max_tau, tau_part_num, partition_option, loss_option)


predictions = regressor.predict_vae_dnn(test_original_X, test_tau)

predictions = np.array(predictions)

# save test file
np.save(test_data_predictions_labels_file, predictions)

# evaluation
print('Errors : ', eval_(predictions, test_Y))

