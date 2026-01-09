#!/bin/bash

PYTHON='/home/zhanglu/.conda/envs/gcd/bin/python'

export CUDA_VISIBLE_DEVICES=2

${PYTHON} -m methods.contrastive_training.contrastive_training_text \
            --dataset_name 'talkmoves' \
            --model_name 'bert-base-uncased' \
            --local_model_path '/home/zhanglu/bert/bert-base-uncased' \
            --batch_size 32 \
            --epochs 20 \
            --num_workers 0 \
            --lr 0.1 \
            --momentum 0.9 \
            --weight_decay 5e-5 \
            --temperature 1.0 \
            --sup_con_weight 0.5 \
            --prop_train_labels 0.8 \
            --seed 1 \
            --eval_funcs 'v1' 'v2'
