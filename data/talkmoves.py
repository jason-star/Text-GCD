import os
import pandas as pd
import numpy as np
from copy import deepcopy
from data.data_utils import TextDataset, subsample_instances
from config import talkmoves_root

def load_tsv_dataset(file_path):
    """
    加载TSV格式的数据集
    格式为：text\tlabel（第一行是header）
    """
    df = pd.read_csv(file_path, sep='\t')
    
    # 检查列名
    if 'text' in df.columns and 'label' in df.columns:
        texts = df['text'].tolist()
        labels = [int(x) for x in df['label'].tolist()]
    else:
        # 如果没有header，尝试使用列位置
        texts = df.iloc[:, 0].tolist()
        labels = [int(x) for x in df.iloc[:, 1].tolist()]
    
    return texts, labels

def subsample_dataset(dataset, idxs):
    """
    对文本数据集进行子采样
    """
    texts = [dataset.texts[i] for i in idxs]
    labels = [dataset.labels[i] for i in idxs]
    uq_idxs = dataset.uq_idxs[idxs]
    return TextDataset(texts, labels, uq_idxs=uq_idxs)

def subsample_classes(dataset, include_classes=None):
    """
    根据类别进行子采样
    """
    if include_classes is None:
        return dataset
    
    cls_idxs = [x for x, t in enumerate(dataset.labels) if t in include_classes]
    
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i
    
    dataset = subsample_dataset(dataset, np.array(cls_idxs))
    dataset.target_transform = lambda x: target_xform_dict[x]
    
    return dataset

def get_train_val_indices(train_dataset, val_split=0.2):
    """
    获取训练集和验证集的索引
    """
    train_classes = list(set(train_dataset.labels))
    
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(np.array(train_dataset.labels) == cls)[0]
        v_ = np.random.choice(cls_idxs, replace=False, size=(int(val_split * len(cls_idxs)),))
        t_ = [x for x in cls_idxs if x not in v_]
        
        train_idxs.extend(t_)
        val_idxs.extend(v_)
    
    return train_idxs, val_idxs

def get_talkmoves_datasets(train_transform=None, test_transform=None, 
                          train_classes=range(4), prop_train_labels=0.8, 
                          split_train_val=False, seed=0):
    """
    加载TalkMoves数据集，返回训练集和测试集
    TalkMoves共有7个类别（0-6），默认0-3为已知类别，4-6为未知类别
    """
    np.random.seed(seed)
    
    # 加载数据集
    print(f'Loading talkmoves dataset from {talkmoves_root}')
    train_texts, train_labels = load_tsv_dataset(os.path.join(talkmoves_root, 'train.tsv'))
    dev_texts, dev_labels = load_tsv_dataset(os.path.join(talkmoves_root, 'dev.tsv'))
    test_texts, test_labels = load_tsv_dataset(os.path.join(talkmoves_root, 'test.tsv'))
    
    print(f'Loaded {len(train_texts)} train samples, {len(dev_texts)} dev samples, {len(test_texts)} test samples')
    
    # 创建完整的训练集（train + dev）
    all_texts = train_texts + dev_texts
    all_labels = train_labels + dev_labels
    all_dataset = TextDataset(all_texts, all_labels)
    all_dataset.uq_idxs = np.array(range(len(all_dataset)))
    
    # 创建测试集
    test_dataset = TextDataset(test_texts, test_labels)
    test_dataset.uq_idxs = np.array(range(len(test_dataset)))
    
    # 选择已知类别作为有标签数据
    train_dataset_labelled = subsample_classes(deepcopy(all_dataset), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)
    
    # 获取无标签数据（其他类别）
    unlabelled_indices = set(all_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(all_dataset), np.array(list(unlabelled_indices)))
    
    # 如果需要分割训练集和验证集
    if split_train_val:
        train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
        train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
        val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
        
        train_dataset_labelled = train_dataset_labelled_split
        val_dataset_labelled = val_dataset_labelled_split
    else:
        val_dataset_labelled = None
    
    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }
    
    return all_datasets

if __name__ == '__main__':
    x = get_talkmoves_datasets(split_train_val=False,
                              train_classes=range(4), prop_train_labels=0.5)
    
    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')
    
    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))
    
    print(f'Num Labelled Classes: {len(set(x["train_labelled"].labels))}')
    print(f'Num Unlabelled Classes: {len(set(x["train_unlabelled"].labels))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')
