#!/usr/bin/env python3
from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from model import PAD

# Anything outside the configured alphabet maps here (X, U, Z, O, B, ...).
UNK = 1


# --------------------------------------------------------------------------- #
# Tokeniser -- built from config.data.aa_alphabet, no vocabulary is downloaded
# --------------------------------------------------------------------------- #


class Tokenizer:
    """Maps single residues to ids: 0 = pad, 1 = unknown, 2.. = alphabet."""

    def __init__(self, alphabet: str):
        self.map = {c: i + 2 for i, c in enumerate(alphabet)}
        self.vocab_size = len(alphabet) + 2

    def encode(self, seq: str) -> list[int]:
        """Uppercase and map to ids; anything outside the alphabet becomes UNK."""
        return [self.map.get(c, UNK) for c in seq.upper()]


# --------------------------------------------------------------------------- #
# FASTA parsing and labelling
# --------------------------------------------------------------------------- #


def read_fasta(path: Path):
    header, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:], []
            else:
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks)


def parse_header(header: str, cfg: Config):
    if header.startswith("SEQ") and header[3:].isdigit():
        return None
    if header.startswith("sp|"):
        return cfg.data.nonarg_label
    fields = header.split("|")
    if len(fields) == 7 and fields[1] == "FEATURES":
        return cfg.data.label_map[fields[3]]
    raise ValueError(f"unrecognised header: {header[:120]}")


def load_split(cfg: Config, split: str, limit: int = 0, seed: int = 0,
               shuffle: bool = True):
    seqs, labels, ids = [], [], []
    for header, seq in read_fasta(cfg.data.path(split)):
        label = parse_header(header, cfg)
        if label is None:  # unlabelled (test.fasta)
            continue
        seqs.append(seq.upper())
        labels.append(label)
        ids.append(header)
    if limit:
        order = list(range(len(seqs)))
        if shuffle:
            random.Random(seed).shuffle(order)
        order = order[:limit]
        seqs = [seqs[i] for i in order]
        labels = [labels[i] for i in order]
        ids = [ids[i] for i in order]
    return seqs, labels, ids


def load_test(cfg: Config):
    """Read the unlabelled test split. Returns (ids, sequences)."""
    ids, seqs = [], []
    for header, seq in read_fasta(cfg.data.path("test")):
        ids.append(header.split()[0])
        seqs.append(seq.upper())
    return ids, seqs


def label_name(label: int, cfg: Config) -> str:
    """Inverse of cfg.data.label_map, with the nonarg fallback."""
    for name, idx in cfg.data.label_map.items():
        if idx == label:
            return name
    return "nonarg"


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


class ProteinDataset(Dataset):
    """Tokenises up front and truncates to `max_len`; padding happens in collate."""

    def __init__(self, seqs, labels, tokenizer: Tokenizer, max_len: int):
        self.ids = [tokenizer.encode(s)[:max_len] for s in seqs]
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.ids[i], dtype=torch.long),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
        }


class TestDataset(Dataset):
    """Same encoding, no labels."""

    def __init__(self, seqs, tokenizer: Tokenizer, max_len: int):
        self.ids = [tokenizer.encode(s)[:max_len] for s in seqs]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": torch.tensor(self.ids[i], dtype=torch.long)}


# --------------------------------------------------------------------------- #
# Batching -- dynamic padding to the longest item in each batch
# --------------------------------------------------------------------------- #


def _pad(batch, with_labels: bool):
    width = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), width), PAD, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = b["input_ids"]
        attention[i, :n] = 1
    out = {"input_ids": input_ids, "attention_mask": attention}
    if with_labels:
        out["labels"] = torch.stack([b["labels"] for b in batch])
    return out


def collate(batch):
    """Pad a batch of ProteinDataset items to the longest sequence."""
    return _pad(batch, with_labels=True)


def collate_unlabelled(batch):
    """Pad a batch of TestDataset items (no labels)."""
    return _pad(batch, with_labels=False)


# --------------------------------------------------------------------------- #
# Standalone sanity check
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from config import load_config

    cfg = load_config([])
    tok = Tokenizer(cfg.data.aa_alphabet)

    for split in ("train", "val"):
        seqs, labels, _ = load_split(cfg, split)
        print(f"{split:5s}: {len(seqs):6d} sequences, "
              f"{len(set(labels)):2d} classes present, "
              f"mean len {sum(map(len, seqs)) / len(seqs):.0f}")

    ids, test_seqs = load_test(cfg)
    print(f"test : {len(ids):6d} sequences, first id {ids[0]!r}")

    seqs, labels, _ = load_split(cfg, "train", limit=8, seed=0)
    batch = collate([ProteinDataset(seqs, labels, tok, cfg.data.max_len)[i]
                     for i in range(8)])
    print(f"batch: input_ids{tuple(batch['input_ids'].shape)} "
          f"attention{tuple(batch['attention_mask'].shape)} "
          f"labels{tuple(batch['labels'].shape)} "
          f"pad_id={PAD} unk_id={UNK} vocab={tok.vocab_size}")
