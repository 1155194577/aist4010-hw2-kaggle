#!/usr/bin/env python3
"""Write `submission.csv` from a trained checkpoint.

    python predict.py                                   # newest checkpoints/*_best.pt
    python predict.py --checkpoint checkpoints/cnn_v2_best.pt --out submission.csv

Reuses the tokeniser, dataset, collate and label space from training so
inference cannot drift from it: the preprocessing parameters (max_len, label
map, num_labels) are read back out of the checkpoint, not from today's defaults.

The output matches data/sample_submission.csv exactly: a `id,label` header and
one row per test sequence, in test.fasta order, with integer class ids.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import load_config
from dataPreprocessor import (
    TestDataset,
    Tokenizer,
    collate_unlabelled,
    label_name,
    load_test,
)
from model import build_model, resolve_device


def newest_checkpoint() -> Path:
    """Most recently written checkpoints/*_best.pt (handles --tag names)."""
    candidates = sorted(Path("checkpoints").glob("*_best.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit("no checkpoints/*_best.pt found; train first or pass --checkpoint")
    return candidates[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict test.fasta -> submission.csv")
    ap.add_argument("--checkpoint", default=None, help="default: newest checkpoints/*_best.pt")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = ap.parse_args()

    cfg = load_config([])
    ckpt_path = Path(args.checkpoint) if args.checkpoint else newest_checkpoint()
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    device = resolve_device(args.device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Inference must use the preprocessing that produced the weights.
    if "config" in ckpt:
        saved = ckpt["config"]["data"]
        cfg.data.max_len = int(saved["max_len"])
        cfg.data.label_map = dict(saved["label_map"])
        cfg.data.nonarg_label = int(saved["nonarg_label"])
        cfg.data.num_labels = int(saved["num_labels"])

    tokenizer = Tokenizer(cfg.data.aa_alphabet)
    model = build_model(cfg, tokenizer.vocab_size, device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    ids, seqs = load_test(cfg)
    print(f"checkpoint: {ckpt_path}  device={device}  test={len(seqs)}  max_len={cfg.data.max_len}")

    loader = DataLoader(
        TestDataset(seqs, tokenizer, cfg.data.max_len),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_unlabelled,
    )

    preds = []
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            preds += logits.argmax(-1).cpu().tolist()
            if i % 10 == 0:
                print(f"  {min(i * args.batch_size, len(seqs))}/{len(seqs)}", flush=True)

    out = Path(args.out)
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "label"])
        writer.writerows(zip(ids, preds))
    print(f"wrote {out} ({len(preds)} rows)")

    # The grader matches on id, but a reordered file is the classic silent bug.
    sample = cfg.data.data_dir / "sample_submission.csv"
    if sample.exists():
        with open(sample) as fh:
            expected = [row["id"] for row in csv.DictReader(fh)]
        status = "ok" if expected == ids else "MISMATCH"
        print(f"  [{status}] id order matches {sample.name} ({len(expected)} rows)")

    counts = Counter(preds)
    print("predicted distribution:")
    for label in range(cfg.data.num_labels):
        print(f"  {label:2d} {label_name(label, cfg):38s} {counts.get(label, 0):5d}")


if __name__ == "__main__":
    main()
