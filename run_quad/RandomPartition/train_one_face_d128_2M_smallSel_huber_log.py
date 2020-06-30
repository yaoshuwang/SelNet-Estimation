import numpy as np
import sys
sys.path.append('../../model')
from selnetpart import *

train_data_file = '../../data/face/train/face_RandomPartition_trainingData.npy'
test_data_file = '../../data/face/train/face_testingData.npy'
valid_data_file = '../../data/face/train/face_valdiationData.npy'

train_data_ = np.load(train_data_file)
test_data_ = np.load(test_data_file)
valid_data_ = np.load(valid_data_file)

x_dim = 128
x_reducedim = 80

max_tau = 1
tau_part_num = 50
leaf_num = 3

loss_option = 'huber_log'
partition_option = 'l2'


'''
test_data = np.zeros((test_data_.shape[0], x_dim + 1 + leaf_num + 1))
test_data[:, :(x_dim + 1)] = test_data_[:, :(x_dim + 1)]
for lid in range(len(greedy_clusters)):
    cluster = greedy_clusters[lid]
    # deal with the first one
    old_lid = int(cluster[0][0])
    test_data[:, x_dim + 1 + lid] = test_data_[:, x_dim + 1 + old_lid]
    for cid in range(1, len(cluster), 1):
        old_lid = int(cluster[cid][0])
        test_data[:, x_dim + 1 + lid] += test_data_[:, x_dim + 1 + old_lid]
'''
train_data = train_data_
test_data = test_data_
valid_data = valid_data_

# deal with mapping
train_mapping = np.ones((train_data_.shape[0], leaf_num))
test_mapping = np.ones((test_data_.shape[0], leaf_num))
valid_mapping = np.ones((valid_data_.shape[0], leaf_num))

train_original_X = np.array(train_data[:, :x_dim], dtype=np.float32)
train_taus_ = []
for rid in range(train_data.shape[0]):
    t = train_data[rid, x_dim]
    train_taus_.append(t)
train_taus_ = np.array(train_taus_)
train_taus = np.zeros((train_data.shape[0], tau_part_num))
for cid in range(tau_part_num):
    train_taus[:, cid] = train_taus_

train_y = np.array(train_data[:, x_dim + 1:], dtype=np.float32)

test_original_X = np.array(test_data[:, :x_dim], dtype=np.float32)
test_taus_ = test_data[:, x_dim]
test_taus_ = []
for rid in range(test_data.shape[0]):
    t = test_data[rid, x_dim]
    test_taus_.append(t)
test_taus_ = np.array(test_taus_)
test_taus = np.zeros((test_data.shape[0], tau_part_num))
for cid in range(tau_part_num):
    test_taus[:, cid] = test_taus_

test_y = np.array(test_data[:, x_dim + 1:], dtype=np.float32)

valid_original_X = np.array(valid_data[:, :x_dim], dtype=np.float32)
valid_taus_ = valid_data[:, x_dim]
valid_taus_ = []
for rid in range(valid_data.shape[0]):
    t = valid_data[rid, x_dim]
    valid_taus_.append(t)
valid_taus_ = np.array(valid_taus_)
valid_taus = np.zeros((valid_data.shape[0], tau_part_num))
for cid in range(tau_part_num):
    valid_taus[:, cid] = valid_taus_

valid_y = np.array(valid_data[:, x_dim + 1:], dtype=np.float32)


unit_len = 80
max_tau = 1

hidden_units = [512, 512, 256, 256]
vae_hidden_units = [512, 256, 128]

batch_size = 512 #1024
epochs = 1500
epochs_vae = 100
learning_rate = 0.00001
log_option = False
tau_embedding_size = 5
original_x_dim = train_original_X.shape[1]
dimreduce_x_dim = x_reducedim

test_data_predictions_labels_file = os.path.join('./test_face_d128_2M_huber_log_new_smallSel_greedy_clusters', 'test_predictions.npy')
valid_data_predictions_labels_file = os.path.join('./valid_face_d128_2M_huber_log_new_smallSel_greedy_clusters', 'valid_predictions_labels')
regression_name = 'face_d128_2M_huber_log_new_smallSel_greedy_clusters_'
regression_model_dir = './model_dir_face_d128_2M_huber_log_new_smallSel_greedy_clusters/regression_CTNet'

regressor = SelNetPart(hidden_units, vae_hidden_units, batch_size, epochs, epochs_vae,
                            learning_rate, log_option, tau_embedding_size, original_x_dim, dimreduce_x_dim,
                            test_data_predictions_labels_file, valid_data_predictions_labels_file, 
                            regression_name, regression_model_dir, unit_len,
                            max_tau, tau_part_num, leaf_num, partition_option, loss_option)

# train
regressor.train_vae_dnn(train_original_X, train_mapping, train_taus, train_y, 
                        valid_original_X, valid_mapping, valid_taus, valid_y)




