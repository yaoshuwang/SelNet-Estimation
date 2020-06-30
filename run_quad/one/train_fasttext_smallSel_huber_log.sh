#!/bin/bash

mkdir ./test_fasttext_smallSel_huber_log
mkdir ./model_dir_fasttext_smallSel_huber_log

rm -r ./model_dir_fasttext_smallSel_huber_log/*
rm -r ./test_fasttext_smallSel_huber_log/*

resultFile='./info_fasttext_smallSel_huber_log_kdd.txt'

> ${resultFile}

export CUDA_VISIBLE_DEVICES=7
python3 train_fasttext_smallSel_huber_log.py >> ${resultFile}
