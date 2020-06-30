#!/bin/bash

mkdir ./test_youtube_smallSel_huber_log
mkdir ./model_dir_youtube_smallSel_huber_log

rm -r ./model_dir_youtube_smallSel_huber_log/*
rm -r ./test_youtube_smallSel_huber_log/*

resultFile='./info_youtube_smallSel_huber_log_kdd.txt'

> ${resultFile}

export CUDA_VISIBLE_DEVICES=2
python3 train_youtube_huber_log.py >> ${resultFile}
