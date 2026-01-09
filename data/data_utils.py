import numpy as np
from torch.utils.data import Dataset

def subsample_instances(dataset, prop_indices_to_subsample=0.8):
    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))
    return subsample_indices

class TextDataset(Dataset):
    """
    文本数据集类，支持加载和处理文本数据
    """
    def __init__(self, texts, labels, uq_idxs=None):
        self.texts = texts
        self.labels = np.array(labels, dtype=np.int64)
        self.uq_idxs = uq_idxs if uq_idxs is not None else np.arange(len(texts))
        self.target_transform = None

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        uq_idx = self.uq_idxs[index]
        
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        return text, label, uq_idx

    def __len__(self):
        return len(self.texts)

class MergedDataset(Dataset):
    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):
        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):
        if item < len(self.labelled_dataset):
            text, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1
        else:
            text, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0

        return text, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)
