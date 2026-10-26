#!/usr/bin/env python3
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config, ModelConfig

PAD = 0

class BasicCNN(nn.Module):
    """The absolute bare-minimum CNN baseline"""

    def __init__(self, vocab_size: int, num_labels: int, m: ModelConfig):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, m.emb_dim, padding_idx=PAD)
        # Single standard convolution 
        self.conv = nn.Conv1d(m.emb_dim, m.emb_dim, kernel_size=5, padding=2)
        # Direct linear head to labels
        self.head = nn.Linear(m.emb_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        # 1. Embed and transpose for Conv1d: (B, L) -> (B, L, C) -> (B, C, L)
        x = self.emb(input_ids).transpose(1, 2)
        
        # 2. Single conv + activation
        x = F.relu(self.conv(x))
        
        # 3. Simple masked mean pooling over non-pad positions
        mask = attention_mask.unsqueeze(1)               # (B, 1, L)
        denom = mask.sum(dim=2).clamp(min=1)
        pooled = (x * mask).sum(dim=2) / denom           # (B, C)
        
        # 4. Final projection to class logits
        return self.head(pooled)


def build_model(cfg: Config, vocab_size: int, device: torch.device) -> nn.Module:
    model = BasicCNN(vocab_size, cfg.data.num_labels, cfg.model)
    return model.to(device)

def resolve_device(name) -> torch.device: 
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")