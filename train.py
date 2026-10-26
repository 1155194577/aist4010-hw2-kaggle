#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config, load_config
from dataPreprocessor import ProteinDataset, Tokenizer, collate, load_split
from model import build_model, resolve_device


def train_one_epoch(model, loader, optimizer, cfg, device):
    model.train()
    total_loss, seen = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["labels"], label_smoothing=cfg.train.label_smoothing)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        
        total_loss += loss.item() * batch["labels"].size(0)
        seen += batch["labels"].size(0)
    return total_loss / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    total_loss, seen, preds, golds = 0.0, 0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["labels"])
        total_loss += loss.item() * batch["labels"].size(0)
        seen += batch["labels"].size(0)
        preds += logits.argmax(-1).cpu().tolist()
        golds += batch["labels"].cpu().tolist()
        
    from sklearn.metrics import f1_score
    macro_f1 = float(f1_score(golds, preds, average="macro", zero_division=0))
    return total_loss / max(seen, 1), macro_f1


def main() -> None:
    cfg = load_config()
    t, d = cfg.train, cfg.data
    device = resolve_device(t.device)

    tokenizer = Tokenizer(d.aa_alphabet)
    model = build_model(cfg, tokenizer.vocab_size, device)
    print(f"device={device}  arch={cfg.model.arch}  params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    train_seqs, train_labels, _ = load_split(cfg, "train", d.limit, t.seed)
    eval_seqs, eval_labels, _ = load_split(cfg, d.eval_split, d.eval_limit, t.seed)

    train_loader = DataLoader(
        ProteinDataset(train_seqs, train_labels, tokenizer, d.max_len),
        batch_size=t.batch_size, shuffle=True, collate_fn=collate, num_workers=t.num_workers,
    )
    eval_loader = DataLoader(
        ProteinDataset(eval_seqs, eval_labels, tokenizer, d.max_len),
        batch_size=max(t.batch_size, 32), shuffle=False, collate_fn=collate, num_workers=t.num_workers,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    t.save_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    for epoch in range(1, t.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg, device)
        eval_loss, f1 = evaluate(model, eval_loader, cfg, device)
        print(f"epoch {epoch}/{t.epochs}: train_loss={train_loss:.4f}  eval_loss={eval_loss:.4f}  macro_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model_state": model.state_dict(), "config": cfg.to_dict()}, t.save_dir / f"{cfg.model.arch}_best.pt")

    print(json.dumps({"best_macro_f1": round(float(best_f1), 4)}))


if __name__ == "__main__":
    main()