#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config, load_config
from dataPreprocessor import ProteinDataset, Tokenizer, collate, load_split
from logging_utils import TrainingLogger
from model import build_model, resolve_device


def train_one_epoch(model, loader, optimizer, cfg, device, epoch: int, log=None) -> float:
    model.train()
    total_loss, seen = 0.0, 0
    steps = len(loader)
    # cfg.train.log_every == 0 means "auto": a line every 10% of the epoch, so a
    # long silent epoch is still visible in the log and on the console.
    every = cfg.train.log_every if cfg.train.log_every > 0 else max(1, steps // 10)
    started = time.time()

    for step, batch in enumerate(loader, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["labels"], label_smoothing=cfg.train.label_smoothing)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()

        total_loss += loss.item() * batch["labels"].size(0)
        seen += batch["labels"].size(0)

        if log is not None and every and (step % every == 0 or step == steps):
            elapsed = time.time() - started
            log.progress(
                epoch, step, steps, total_loss / seen,
                rate=seen / max(elapsed, 1e-9),
                eta=(steps - step) * elapsed / step,
            )

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
    log = TrainingLogger(t.save_dir.parent / "logs", arch=cfg.model.arch, tag=t.tag)

    tokenizer = Tokenizer(d.aa_alphabet)
    model = build_model(cfg, tokenizer.vocab_size, device)
    print(model)
    params = sum(p.numel() for p in model.parameters())

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

    log.start(
        cfg, device=str(device), params=params,
        n_train=len(train_seqs), n_eval=len(eval_seqs), steps_per_epoch=len(train_loader),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    t.save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = t.save_dir / f"{cfg.model.arch}{'_' + t.tag if t.tag else ''}_best.pt"

    best_f1 = -1.0
    for epoch in range(1, t.epochs + 1):
        started = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg, device, epoch, log=log)
        eval_loss, f1 = evaluate(model, eval_loader, cfg, device)

        improved = f1 > best_f1
        if improved:
            best_f1 = f1
            torch.save({"model_state": model.state_dict(), "config": cfg.to_dict()}, checkpoint)

        log.epoch(
            epoch, t.epochs,
            {"train_loss": train_loss, "eval_loss": eval_loss, "macro_f1": f1,
             "lr": optimizer.param_groups[0]["lr"]},
            best=improved, seconds=time.time() - started,
        )
        if improved:
            log.checkpoint(best_f1, checkpoint)

    log.finish({"best_macro_f1": round(float(best_f1), 4)})


if __name__ == "__main__":
    main()
