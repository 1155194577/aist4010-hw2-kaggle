#!/usr/bin/env python3
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config, ModelConfig

PAD = 0
class ConvBlock(nn.Module):
    """conv (length-preserving) -> activation"""

    def __init__(self, cin: int, cout: int, kernel: int, dilation: int):
        super().__init__()
        pad = dilation * (kernel - 1) // 2          # keeps L unchanged
        self.conv = nn.Conv1d(cin, cout, kernel, padding=pad, dilation=dilation)

    def forward(self, x):
        return F.relu(self.conv(x))

class BasicCNN(nn.Module):
    """The absolute bare-minimum CNN baseline"""

    def __init__(self, vocab_size: int, num_labels: int, m: ModelConfig):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, m.emb_dim, padding_idx=PAD)
        self.blocks = nn.Sequential(
            ConvBlock(m.emb_dim,   m.widths[0], m.kernels[0], m.dilations[0]),
            ConvBlock(m.widths[0], m.widths[1], m.kernels[1], m.dilations[1]),
            ConvBlock(m.widths[1], m.widths[2], m.kernels[2], m.dilations[2]),
        )
        

    def forward(self, input_ids, attention_mask):
        # 1. Embed and transpose for Conv1d: (B, L) -> (B, L, C) -> (B, C, L)
        x = self.emb(input_ids).transpose(1, 2)
        
        
        x = self.blocks(x)
        # 3. Simple masked mean pooling over non-pad positions
        mask = attention_mask.unsqueeze(1)               # (B, 1, L)
        denom = mask.sum(dim=2).clamp(min=1)
        pooled = (x * mask).sum(dim=2) / denom           # (B, C)
        
        # 4. Final projection to class logits

        # TODO : does dropout do anything? 
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