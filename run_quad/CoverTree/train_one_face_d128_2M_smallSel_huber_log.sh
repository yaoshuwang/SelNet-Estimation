#!/bin/bash

mkdir ./test_face_d128_2M_huber_log_new_smallSel_greedy_clusters
mkdir ./valid_face_d128_2M_huber_log_new_smallSel_greedy_clusters
mkdir ./model_dir_face_d128_2M_huber_log_new_smallSel_greedy_clusters

rm -r ./test_face_d128_2M_huber_log_new_smallSel_greedy_clusters/*
rm -r ./valid_face_d128_2M_huber_log_new_smallSel_greedy_clusters/*
rm -r ./model_dir_face_d128_2M_huber_log_new_smallSel_greedy_clusters/*


export CUDA_VISIBLE_DEVICES=6
python3 train_one_face_d128_2M_smallSel_huber_log.py >> 'info_face_d128_2M_huber_log_new_smallSel_greedy_clusters.txt'
