#!/bin/bash

mkdir ./test_youtube_huber_log_greedy_clusters
mkdir ./valid_youtube_huber_log_greedy_clusters
mkdir ./model_dir_youtube_huber_log_greedy_clusters

rm -r ./test_youtube_huber_log_greedy_clusters/*
rm -r ./valid_youtube_huber_log_greedy_clusters/*
rm -r ./model_dir_youtube_huber_log_greedy_clusters/*


export CUDA_VISIBLE_DEVICES=0
python3 train_one_youtube_huber_log.py >> 'info_youtube_huber_log_greedy_clusters.txt'
