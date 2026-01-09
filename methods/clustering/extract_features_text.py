import torch
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import numpy as np

from models.text_transformer import TextTransformer
from data.get_datasets import get_datasets, get_class_splits
from project_utils.general_utils import str2bool, strip_state_dict
from config import feature_extract_dir

device = torch.device('cuda:0')

def extract_features_text(model, loader, save_dir):
    """
    为文本数据提取特征并保存
    """
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            texts, labels, idxs = batch[:3]
            
            # 将texts转换为list（tokenizer期望list格式）
            if isinstance(texts, tuple):
                texts = list(texts)
            
            # 文本编码
            features = model.encode_texts(texts)  # [batch_size, feature_dim]
            
            # 保存特征
            for f, t, uq in zip(features, labels, idxs):
                t = t.item()
                uq = uq.item()
                
                # 创建目录结构：save_dir/class_id/sample_id.npy
                save_path = os.path.join(save_dir, f'{t}', f'{uq}.npy')
                np.save(save_path, f.detach().cpu().numpy())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='Extract text features',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--root_dir', type=str, default=feature_extract_dir)
    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--use_best_model', type=str2bool, default=True)
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--local_model_path', type=str, default='/home/zhanglu/bert/bert-base-uncased',
                        help='Local path to BERT model')
    parser.add_argument('--dataset_name', type=str, default='talkmoves')
    parser.add_argument('--prop_train_labels', type=float, default=0.8)
    
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    args = get_class_splits(args)
    
    args.save_dir = os.path.join(args.root_dir, f'{args.model_name}_{args.dataset_name}')
    print(f'Feature save directory: {args.save_dir}')

    print('Loading model...')
    # ----------------------
    # MODEL
    # ----------------------
    model = TextTransformer(model_name=args.model_name, output_dim=768, pretrained=True, 
                           local_model_path=args.local_model_path)
    
    # 加载训练后的模型权重
    if args.warmup_model_dir is not None:
        warmup_id = args.warmup_model_dir.split('(')[1].split(')')[0]
        
        if args.use_best_model:
            # 如果已经是_best.pt结尾，不需要修改路径
            if not args.warmup_model_dir.endswith("_best.pt"):
                args.warmup_model_dir = args.warmup_model_dir[:-3] + "_best.pt"
            args.save_dir += '_(' + args.warmup_model_dir.split('(')[1].split(')')[0] + ')_best'
        else:
            args.save_dir += '_(' + args.warmup_model_dir.split('(')[1].split(')')[0] + ')'
        
        print(f'Loading weights from {args.warmup_model_dir}...')
        state_dict = torch.load(args.warmup_model_dir)
        model.load_state_dict(state_dict)
        print(f'Saving features to {args.save_dir}')

    print('Loading data...')
    # ----------------------
    # DATASET
    # ----------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(
        args.dataset_name, train_transform=None, test_transform=None, args=args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers)

    # 获取所有类别标签 - 对于TalkMoves数据集固定为0-6
    targets = list(range(7))  # TalkMoves数据集有7个类别：0-6
    print("Available targets:", targets)
    targets.sort()

    print('Creating base directories...')
    # ----------------------
    # INIT SAVE DIRS
    # Create a directory for each class
    # ----------------------
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for fold in ('train', 'test'):
        fold_dir = os.path.join(args.save_dir, fold)
        if not os.path.exists(fold_dir):
            os.mkdir(fold_dir)

        for t in targets:
            target_dir = os.path.join(fold_dir, f'{t}')
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

    # ----------------------
    # EXTRACT FEATURES
    # ----------------------
    # Extract train features
    train_save_dir = os.path.join(args.save_dir, 'train')
    print('Extracting features from train split...')
    extract_features_text(model=model, loader=train_loader, save_dir=train_save_dir)

    # Extract test features
    test_save_dir = os.path.join(args.save_dir, 'test')
    print('Extracting features from test split...')
    extract_features_text(model=model, loader=test_loader, save_dir=test_save_dir)

    print('✅ Feature extraction complete!')
