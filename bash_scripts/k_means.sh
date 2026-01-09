#!/bin/bash

PYTHON='/home/zhanglu/.conda/envs/gcd/bin/python'

export CUDA_VISIBLE_DEVICES=2

SAVE_DIR=/home/zhanglu/GCD/osr_categories/dev_outputs/

mkdir -p ${SAVE_DIR}

EXP_NUM=$(ls ${SAVE_DIR} 2>/dev/null | wc -l)
EXP_NUM=$((${EXP_NUM}+1))

${PYTHON} -m methods.clustering.k_means \
            --batch_size 32 \
            --num_workers 0 \
            --model_name 'bert-base-uncased' \
            --local_model_path '/home/zhanglu/bert/bert-base-uncased' \
            --dataset_name 'talkmoves' \
            --semi_sup 'True' \
            --use_best_model 'True' \
            --warmup_model_exp_id '(08.01.2026_|_47.448)' \
            --max_kmeans_iter 10 \
            --k_means_init 10 \
            --prop_train_labels 0.8 \
            --eval_funcs 'v1' 'v2' \
            2>&1
