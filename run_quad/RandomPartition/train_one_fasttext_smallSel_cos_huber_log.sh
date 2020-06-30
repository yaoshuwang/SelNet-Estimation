#!/bin/bash

mkdir ./test_fasttext_smallSel_Cosine_huber_log_greedy_clusters
mkdir ./valid_fasttext_smallSel_Cosine_huber_log_greedy_clusters
mkdir ./model_dir_fasttext_smallSel_Cosine_huber_log_greedy_clusters

rm -r ./test_fasttext_smallSel_Cosine_huber_log_greedy_clusters/*
rm -r ./valid_fasttext_smallSel_Cosine_huber_log_greedy_clusters/*
rm -r ./model_dir_fasttext_smallSel_Cosine_huber_log_greedy_clusters/*


export CUDA_VISIBLE_DEVICES=0
python3 train_one_fasttext_smallSel_cos_huber_log.py >> 'info_fasttext_smallSel_Cosine_huber_log_greedy_clusters.txt'
