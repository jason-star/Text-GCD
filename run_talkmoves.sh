#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

# 获取脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 设置日志文件
LOG_FILE="talkmoves_pipeline_$(date +%Y%m%d_%H%M%S).log"

# 第一步：运行对比训练
bash bash_scripts/contrastive_train.sh 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "对比训练失败，流程停止！"
    echo "查看日志: $LOG_FILE"
    exit 1
fi

echo ""
sleep 2

# 第二步：运行特征提取
bash bash_scripts/extract_features.sh 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "查看日志: $LOG_FILE"
    exit 1
fi

echo ""
sleep 2

# 第三步：运行K-Means聚类
bash bash_scripts/k_means.sh 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "查看日志: $LOG_FILE"
    exit 1
fi

