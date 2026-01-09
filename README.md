# TalkMoves 文本数据集 GCD 实现


## 概述

由于实验需要该模型作为baseline，已成功将最初GCD模型从**图像处理**完全转换为**文本数据集**处理，遵循原始的三步流程架构。

S. Vaze, K. Hant, A. Vedaldi and A. Zisserman, "Generalized Category Discovery," 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), New Orleans, LA, USA, 2022, pp. 7482-7491, doi: 10.1109/CVPR52688.2022.00734.

## 三步流程验证状态

### 步骤 1: 对比学习训练 (contrastive_train.sh)

**脚本**: `bash_scripts/contrastive_train.sh`  
**Python模块**: `methods/contrastive_training/contrastive_training_text.py`

**功能**:
- 使用 BERT-base-uncased 模型处理文本
- 有监督对比学习损失函数
- 自动模型检查点保存


**关键代码**:
```python
- SimpleSupConLoss: 改进的对比损失，数值稳定
- collate_fn: 处理可变长度文本批处理
- train(): 完整的训练循环
- test_kmeans(): K-Means评估（已修复mask定义）
```

### 步骤 2: 特征提取 (extract_features.sh)

**脚本**: `bash_scripts/extract_features.sh`  
**Python模块**: `methods/clustering/extract_features_text.py`

**功能**:
- 从训练的模型中提取文本特征
- 保存为按类别组织的 .npy 文件
- 自动检测最新模型


**关键代码**:
```python
- extract_features_text(): 批量特征提取
- 特征向量维度: (N, 768)
- 保存结构: save_dir/{class}/{sample_id}.npy
```

### 步骤 3: K-Means 聚类 (k_means.sh)

**脚本**: `bash_scripts/k_means.sh`  
**Python模块**: `methods/clustering/k_means.py` (通用)

**功能**:
- K-Means 聚类新类别发现
- 已知类别精度评估
- 聚类结果保存


## 修改的现有文件

| 文件 | 修改内容 |
|------|---------|
| `data/data_utils.py` | 添加 TextDataset 类 |
| `data/get_datasets.py` | 注册 talkmoves 数据集 |
| `config.py` | 添加 talkmoves 数据路径 |


### 已修复的问题
1. **Mask 为空数组问题** (✓ 已修复)
   - 原因: test_kmeans 中未定义 mask
   - 解决: 添加 `mask = targets < args.num_labeled_classes`
   - 状态: 评估指标正常计算

2. **数据类型不匹配** (✓ 已修复)
   - 原因: tuple vs list 文本格式
   - 解决: 修改 collate_fn 进行转换
   - 状态: 批处理正常

3. **模型加载错误** (✓ 已修复)
   - 原因: 在线加载 HuggingFace 模型失败
   - 解决: 添加本地模型路径支持 `--local_model_path`
   - 状态: 本地模型加载成功

## 数据集信息
Abhijit Suresh, Jennifer Jacobs, Charis Harty, Margaret Perkoff, James H. Martin, and Tamara Sumner. 2022. The TalkMoves Dataset: K-12 Mathematics Lesson Transcripts Annotated for Teacher and Student Discursive Moves. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 4654–4662, Marseille, France. European Language Resources Association.
```
TalkMoves 文本数据集
├── 训练集: 135,672 样本 (train.tsv)
├── 开发集: 46,013 样本 (dev.tsv)
├── 测试集: 27,827 样本 (test.tsv)
├── 总计: ~181,000 样本
├── 类别数: 7 (0-6)
├── 已知类别: 0-3 (4个)
└── 未知类别: 4-6 (3个)
```

## 系统配置

- **Python**: 3.8
- **PyTorch**: 1.13.0 (cu117)
- **Transformers**: 4.20.0
- **CUDA**: 11.7
- **显存需求**: ~8GB (batch_size=32)

## 执行方式

### 完整流程（推荐）
```bash
bash run_talkmoves.sh
```

### 分步执行
```bash
# 1. 对比学习训练
bash bash_scripts/contrastive_train.sh

# 2. 特征提取
bash bash_scripts/extract_features.sh

# 3. K-Means聚类
bash bash_scripts/k_means.sh
```


## 输出位置

```
├── osr_categories/
│   ├── metric_learn_gcd/log/
│   │   └── (DD.MM.YYYY_|_SS.mmm)/
│   │       ├── model_best.pt
│   │       ├── model_epoch_*.pt
│   │       └── checkpoints/
│   └── dev_outputs/
│       └── logfile_*.out
├── osr_novel_categories/
│   └── extracted_features_public_impl/
│       └── bert-base-uncased_talkmoves_best/
│           ├── train/
│           └── test/
└── talkmoves_pipeline_*.log
```







