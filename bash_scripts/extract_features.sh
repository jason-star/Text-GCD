#!/bin/bash

PYTHON='/home/zhanglu/.conda/envs/gcd/bin/python'

export CUDA_VISIBLE_DEVICES=2

LATEST_CHECKPOINT=$(ls -td /home/zhanglu/GCD/generalized-category-discovery-main/osr_categories/metric_learn_gcd/log/*/ 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "未找到训练后的模型检查点"
    echo "请先运行bash bash_scripts/contrastive_train.sh"
    exit 1
fi

if [ ! -f "${LATEST_CHECKPOINT}model_best.pt" ] && [ ! -f "${LATEST_CHECKPOINT}model_epoch_19.pt" ]; then
    echo "未找到模型文件"
    exit 1
fi

${PYTHON} -m methods.clustering.extract_features_text \
            --batch_size 32 \
            --num_workers 0 \
            --model_name 'bert-base-uncased' \
            --local_model_path '/home/zhanglu/bert/bert-base-uncased' \
            --dataset_name 'talkmoves' \
            --warmup_model_dir "${LATEST_CHECKPOINT}model_best.pt" \
            --use_best_model 'True'
