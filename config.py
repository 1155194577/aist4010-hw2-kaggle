#!/usr/bin/env python3
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parent

# 15 labels: 14 antibiotic families + nonarg. Keep in sync with the Kaggle page.
LABEL_MAP: Dict[str, int] = {
    "aminoglycoside": 0,
    "macrolide-lincosamide-streptogramin": 1,
    "polymyxin": 2,
    "fosfomycin": 3,
    "trimethoprim": 4,
    "bacitracin": 5,
    "quinolone": 6,
    "multidrug": 7,
    "chloramphenicol": 8,
    "tetracycline": 9,
    "rifampin": 10,
    "beta_lactam": 11,
    "sulfonamide": 12,
    "glycopeptide": 13,
}


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


@dataclass
class DataConfig:
    """Paths, label space and sequence handling."""

    data_dir: Path = ROOT / "data"
    train_file: str = "train.fasta"
    val_file: str = "val.fasta"
    test_file: str = "test.fasta"

    label_map: Dict[str, int] = field(default_factory=lambda: dict(LABEL_MAP))
    nonarg_label: int = 14
    num_labels: int = 15

    # Which split to score during training.
    eval_split: str = "val"

    # 0 = use the whole split. `limit` is the smoke-test knob.
    limit: int = 0
    eval_limit: int = 0

    # Truncation length. Longest sequence in the data is 1,576 aa.
    max_len: int = 1024

    # Residues the model knows about; everything else maps to the UNK id.
    aa_alphabet: str = "ACDEFGHIKLMNPQRSTVWY"

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.label_map = dict(self.label_map)
        self.aa_alphabet = self.aa_alphabet.upper()
        # label_map holds the 14 antibiotic families; nonarg is the extra slot.
        if self.nonarg_label != len(self.label_map):
            raise ValueError(
                f"nonarg_label={self.nonarg_label} should equal len(label_map)="
                f"{len(self.label_map)}"
            )
        if self.num_labels != len(self.label_map) + 1:
            raise ValueError(
                f"num_labels={self.num_labels} but label_map has "
                f"{len(self.label_map)} families (+1 nonarg)"
            )
        if self.eval_split not in {"train", "val"}:
            raise ValueError(f"eval_split must be 'train' or 'val', got {self.eval_split!r}")
        if self.max_len < 8:
            raise ValueError(f"max_len={self.max_len} is too small")
        if len(set(self.aa_alphabet)) != len(self.aa_alphabet):
            raise ValueError("aa_alphabet contains duplicates")

    def path(self, split: str) -> Path:
        name = {
            "train": self.train_file,
            "val": self.val_file,
            "test": self.test_file,
        }[split]
        return self.data_dir / name


@dataclass
class ModelConfig:
    """Architecture. Both variants are initialised from scratch (no pretrained weights)."""

    arch: str = "cnn"  # "cnn" | "transformer"

    # --- shared ---
    emb_dim: int = 64          # residue embedding width (cnn)
    dropout: float = 0.3       # cnn dropout

    # --- cnn ---
    widths: Tuple[int, ...] = (256, 512, 512)
    kernels: Tuple[int, ...] = (7, 5, 3)
    dilations: Tuple[int, ...] = (1, 2, 4)
    hidden_dim: int = 256      # classifier bottleneck

    # --- transformer ---
    d_model: int = 128
    nhead: int = 4
    layers: int = 4
    ff_dim: int = 512
    attn_dropout: float = 0.1
    pos_init_std: float = 0.02

    def __post_init__(self) -> None:
        self.widths = tuple(int(w) for w in self.widths)
        self.kernels = tuple(int(k) for k in self.kernels)
        self.dilations = tuple(int(d) for d in self.dilations)
        if self.arch not in {"cnn", "transformer"}:
            raise ValueError(f"arch must be 'cnn' or 'transformer', got {self.arch!r}")
        if self.arch == "cnn":
            n = len(self.widths)
            if not (len(self.kernels) == len(self.dilations) == n):
                raise ValueError(
                    "cnn needs len(widths) == len(kernels) == len(dilations), got "
                    f"{n}, {len(self.kernels)}, {len(self.dilations)}"
                )
            if any(k % 2 == 0 for k in self.kernels):
                raise ValueError("cnn kernels must be odd so padding stays symmetric")
        if self.d_model % self.nhead:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by nhead={self.nhead}"
            )


@dataclass
class TrainConfig:
    """Optimiser, schedule, loop and I/O."""

    epochs: int = 1
    batch_size: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.01

    optimizer: str = "adamw"   # "adamw" | "sgd"
    momentum: float = 0.9      # sgd only
    scheduler: str = "linear"  # "none" | "linear" | "cosine"
    warmup_frac: float = 0.06
    grad_clip: float = 1.0

    # Loss shaping. Macro-F1 over a 52%-majority class usually wants class weights.
    class_weights: bool = False
    label_smoothing: float = 0.0

    seed: int = 0
    device: str = "auto"       # "auto" | "cpu" | "mps" | "cuda"
    num_workers: int = 0
    log_every: int = 0         # 0 = auto (every step on tiny runs, else silent)

    save: bool = True
    save_dir: Path = ROOT / "checkpoints"
    tag: str = ""              # appended to checkpoint names, e.g. --tag v2
    smoke: bool = False        # extra pipeline assertions

    def __post_init__(self) -> None:
        self.save_dir = Path(self.save_dir)
        if self.optimizer not in {"adamw", "sgd"}:
            raise ValueError(f"optimizer must be 'adamw' or 'sgd', got {self.optimizer!r}")
        if self.scheduler not in {"none", "linear", "cosine"}:
            raise ValueError(
                f"scheduler must be 'none'|'linear'|'cosine', got {self.scheduler!r}"
            )
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError(f"unknown device {self.device!r}")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")


# --------------------------------------------------------------------------- #
# Root config
# --------------------------------------------------------------------------- #

SECTIONS = {
    "data": DataConfig,
    "model": ModelConfig,
    "train": TrainConfig,
}


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly view (used for checkpointing and logging)."""
        out = asdict(self)
        out["data"]["data_dir"] = str(self.data.data_dir)
        out["train"]["save_dir"] = str(self.train.save_dir)
        for k, v in list(out["model"].items()):
            if isinstance(v, tuple):
                out["model"][k] = list(v)
        return out

    def summary(self) -> str:
        d, m, t = self.data, self.model, self.train
        return (
            f"data : {d.data_dir}  max_len={d.max_len}  "
            f"limit={d.limit or 'all'}  eval={d.eval_split}({d.eval_limit or 'all'})\n"
            f"model: {m.arch}  "
            + (
                f"emb={m.emb_dim} widths={m.widths} kernels={m.kernels} "
                f"dilations={m.dilations} dropout={m.dropout}"
                if m.arch == "cnn"
                else f"d_model={m.d_model} nhead={m.nhead} layers={m.layers} "
                     f"ff={m.ff_dim} dropout={m.attn_dropout}"
            )
            + f"\nopt  : {t.optimizer} lr={t.lr} wd={t.weight_decay} "
            f"sched={t.scheduler} warmup={t.warmup_frac:.0%} clip={t.grad_clip}  "
            f"epochs={t.epochs} bs={t.batch_size}  "
            f"class_weights={t.class_weights} ls={t.label_smoothing}  "
            f"seed={t.seed} device={t.device}"
        )


# --------------------------------------------------------------------------- #
# CLI generated from the dataclasses
# --------------------------------------------------------------------------- #


def _tuple_of_ints(text: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in text.replace(" ", "").split(",") if x)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train AIST4010-A2. Defaults come from config.py; "
                    "any field can be overridden here.",
    )
    seen: Dict[str, str] = {}
    for section, cls in SECTIONS.items():
        group = parser.add_argument_group(section)
        for f in fields(cls):
            if f.name in seen:
                raise RuntimeError(
                    f"field {f.name!r} appears in both {seen[f.name]} and {section}; "
                    "CLI flags must be unique"
                )
            seen[f.name] = section
            default = getattr(cls(), f.name)
            if isinstance(default, dict):
                continue  # structured fields stay in config.py (no safe CLI form)
            flag = "--" + f.name.replace("_", "-")

            if isinstance(default, bool):
                if default:
                    group.add_argument(flag.replace("--", "--no-"), dest=f.name,
                                       action="store_false", help="(default: on)")
                else:
                    group.add_argument(flag, dest=f.name, action="store_true",
                                       help="(default: off)")
            elif isinstance(default, Path):
                group.add_argument(flag, dest=f.name, type=Path, default=default)
            elif isinstance(default, tuple):
                group.add_argument(flag, dest=f.name, type=_tuple_of_ints,
                                   default=default, metavar="N,N,...")
            elif isinstance(default, int):
                group.add_argument(flag, dest=f.name, type=int, default=default)
            elif isinstance(default, float):
                group.add_argument(flag, dest=f.name, type=float, default=default)
            else:
                group.add_argument(flag, dest=f.name, type=str, default=default)
    return parser


def load_config(argv=None) -> Config:
    """Build a Config from config.py defaults plus optional CLI overrides."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    sections = {}
    for name, cls in SECTIONS.items():
        base = cls()  # fields not exposed on the CLI keep their config.py default
        sections[name] = cls(**{f.name: getattr(args, f.name, getattr(base, f.name))
                                for f in fields(cls)})
    return Config(**sections)


if __name__ == "__main__":
    # `python config.py` prints the effective defaults.
    print(load_config([]).summary())
