#!/bin/bash

mkdir ./test_face_d128_2M_smallSel_huber_log
mkdir ./model_dir_face_d128_2M_smallSel_huber_log

rm -r ./model_dir_face_d128_2M_smallSel_huber_log/*
rm -r ./test_face_d128_2M_smallSel_huber_log/*

resultFile='./info_face_d128_2M_smallSel_huber_log_kdd.txt'

> ${resultFile}

export CUDA_VISIBLE_DEVICES=5
python3 train_face_d128_2M_smallSel_huber_log.py >> ${resultFile}
