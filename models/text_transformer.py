import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
import os

class TextTransformer(nn.Module):
    """
    基于Transformer的文本编码器（如BERT）
    """
    def __init__(self, model_name='bert-base-uncased', output_dim=768, pretrained=True, local_model_path=None):
        super(TextTransformer, self).__init__()
        
        self.model_name = model_name
        self.local_model_path = local_model_path
        
        # 尝试从本地路径或缓存加载
        try:
            # 如果指定了本地模型路径，优先使用
            if local_model_path and os.path.exists(local_model_path):
                print(f"✓ 从本地路径加载模型: {local_model_path}")
                self.backbone = AutoModel.from_pretrained(local_model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            else:
                # 否则尝试从缓存加载
                if pretrained:
                    cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
                    try:
                        self.backbone = AutoModel.from_pretrained(
                            model_name, 
                            local_files_only=True,
                            cache_dir=cache_dir
                        )
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_name,
                            local_files_only=True,
                            cache_dir=cache_dir
                        )
                    except:
                        # 如果缓存中没有，尝试在线下载
                        print(f"正在从 HuggingFace 下载模型: {model_name}")
                        self.backbone = AutoModel.from_pretrained(model_name)
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                else:
                    config = AutoConfig.from_pretrained(model_name)
                    self.backbone = AutoModel.from_config(config)
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"❌ 错误：无法加载模型 {model_name}: {e}")
            raise
        
        self.hidden_size = self.backbone.config.hidden_size
        self.output_dim = output_dim
        
        # 投影层将BERT输出投影到指定维度
        if output_dim != self.hidden_size:
            self.projection = nn.Linear(self.hidden_size, output_dim)
        else:
            self.projection = nn.Identity()
    
    def encode_texts(self, texts, max_length=512):
        """
        编码文本列表为特征向量
        """
        if isinstance(texts, str):
            texts = [texts]
        
        encodings = self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # 获取设备
        device = next(self.parameters()).device
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # 应用投影层
        features = self.projection(cls_output)
        return features
    
    def forward(self, texts):
        """
        前向传播
        """
        return self.encode_texts(texts)


class SimpleTextEncoder(nn.Module):
    """
    LSTM文本编码器（备用方案）
    """
    def __init__(self, vocab_size=10000, embedding_dim=300, hidden_dim=256, output_dim=128):
        super(SimpleTextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        output = self.projection(hidden)
        return output
