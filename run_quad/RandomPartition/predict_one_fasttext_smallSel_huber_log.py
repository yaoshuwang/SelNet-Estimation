import numpy as np
import sys
sys.path.append('../../model')
from selnetpart import *

def eval_(predictions, labels):
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    mape = mean_absolute_percentage_error(labels, predictions)
    return (mse, mae, mape)


test_data_file = '../../data/fasttext_eu/train/fasttext_eu_testingData.npy'


test_data_ = np.load(test_data_file)


x_dim = 300
x_reducedim = 80

max_tau = 54.0
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
test_data = test_data_

test_mapping = np.ones((test_data_.shape[0], leaf_num))

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


unit_len = 80
max_tau = 54.0

hidden_units = [1024, 512, 512, 256]
vae_hidden_units = [512, 256, 256]

batch_size = 512 #1024
epochs = 1500
epochs_vae = 100
learning_rate = 0.00001
log_option = False
tau_embedding_size = 5
original_x_dim = test_original_X.shape[1]
dimreduce_x_dim = x_reducedim

test_data_predictions_labels_file = os.path.join('./test_fasttext_smallSel_greedy_clusters', 'test_predictions_labels')
valid_data_predictions_labels_file = os.path.join('./valid_fasttext_smallSel_greedy_clusters', 'valid_predictions_labels')
regression_name = 'fasttext_smallSel_greedy_clusters_'
regression_model_dir = './model_dir_fasttext_smallSel_greedy_clusters/regression_CTNet-1499'

regressor = SelNetPart(hidden_units, vae_hidden_units, batch_size, epochs, epochs_vae,
                            learning_rate, log_option, tau_embedding_size, original_x_dim, dimreduce_x_dim,
                            test_data_predictions_labels_file, valid_data_predictions_labels_file, 
                            regression_name, regression_model_dir, unit_len,
                            max_tau, tau_part_num, leaf_num, partition_option, loss_option)


predictions = regressor.predict_vae_dnn(test_original_X, test_mapping, test_taus)

predictions = np.array(predictions)

np.save(test_data_predictions_labels_file, predictions)

# evaluation
print('Errors : ', eval_(predictions, test_y))



