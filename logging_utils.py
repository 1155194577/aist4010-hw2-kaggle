#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

WALL_FMT = "%Y-%m-%d %H:%M:%S"
CLOCK_FMT = "%H:%M:%S"
STAMP_FMT = "%Y%m%d-%H%M%S"
RULE_WIDTH = 72


def format_duration(seconds: float) -> str:
    """Render seconds as ``mm:ss``, or ``h:mm:ss`` once the run passes an hour."""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _value(value: Any) -> str:
    """Compact formatting for metrics: 4 decimals, scientific for extremes."""
    if isinstance(value, float):
        if value and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.3e}"
        return f"{value:.4f}"
    return str(value)


class TrainingLogger:
    """Console + file + machine-readable log for a single training run."""

    def __init__(
        self,
        log_dir: Path | str = "logs",
        arch: str = "run",
        tag: str = "",
        run_name: Optional[str] = None,
        echo: bool = True,
    ) -> None:
        self.echo = echo
        self.started = time.time()
        self.run_name = run_name or self._make_run_name(arch, tag)

        self.dir = Path(log_dir) / self.run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "train.log"
        self.csv_path = self.dir / "history.csv"
        self.jsonl_path = self.dir / "metrics.jsonl"

        self._log_file = self.log_path.open("a", encoding="utf-8", buffering=1)
        self._jsonl_file = self.jsonl_path.open("a", encoding="utf-8", buffering=1)
        self._csv_fields: List[str] = []
        self._closed = False
        self.epochs: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}
        self.started_iso = datetime.now().strftime(WALL_FMT)

    # -- construction helpers -------------------------------------------------

    @staticmethod
    def _make_run_name(arch: str, tag: str) -> str:
        base = f"{arch}-{tag}" if tag else str(arch)
        return f"{base}-{datetime.now().strftime(STAMP_FMT)}"

    # -- low level ------------------------------------------------------------

    def _write(self, level: str, message: str) -> None:
        line = f"[{datetime.now().strftime(CLOCK_FMT)}] {level:<5} {message}"
        if self.echo:
            print(line, flush=True)
        if not self._closed:
            self._log_file.write(line + "\n")

    def _blank(self) -> None:
        if self.echo:
            print(flush=True)
        if not self._closed:
            self._log_file.write("\n")

    def info(self, message: str) -> None:
        self._write("INFO", message)

    def warn(self, message: str) -> None:
        self._write("WARN", message)

    def error(self, message: str) -> None:
        self._write("ERROR", message)

    def banner(self, title: str) -> None:
        self._blank()
        self.info(title)
        self.info("-" * RULE_WIDTH)

    # -- run lifecycle --------------------------------------------------------

    def start(
        self,
        cfg,
        *,
        device: str,
        params: int,
        n_train: int,
        n_eval: int,
        steps_per_epoch: int,
    ) -> None:
        """Header: where this run writes, its config, and the data it will see."""
        self.banner(f"run {self.run_name}")
        self.info(f"started   {datetime.now().strftime(WALL_FMT)}")
        self.info(f"log dir   {self.dir}")
        for line in cfg.summary().splitlines():
            head, sep, tail = line.partition(":")
            if sep:
                self.info(f"{head.strip():<9} {tail.strip()}")
            else:
                self.info(line.strip())
        self.info(
            f"{'device':<9} {device}  params={params / 1e6:.2f}M  "
            f"train={n_train}  eval={n_eval}  steps/epoch={steps_per_epoch}"
        )

    def progress(
        self,
        epoch: int,
        step: int,
        total: int,
        loss: float,
        rate: Optional[float] = None,
        eta: Optional[float] = None,
    ) -> None:
        """Mid-epoch line so a long epoch is not a silent black box."""
        parts = [
            f"epoch {epoch:>3}",
            f"step {step:>6}/{total:<6}",
            f"{100.0 * step / max(total, 1):5.1f}%",
            f"loss={_value(loss)}",
        ]
        if rate is not None:
            parts.append(f"{rate:7.1f} seq/s")
        if eta is not None:
            parts.append(f"eta {format_duration(eta)}")
        self.info("  ".join(parts))

    def epoch(
        self,
        number: int,
        total: int,
        metrics: Mapping[str, Any],
        *,
        best: bool = False,
        seconds: Optional[float] = None,
    ) -> None:
        """One row per epoch: console line, CSV row, JSONL record."""
        record: Dict[str, Any] = {
            "epoch": number,
            "elapsed_s": round(time.time() - self.started, 1),
        }
        record.update({k: v for k, v in metrics.items()})
        self.epochs.append(record)

        self._write_jsonl(record)
        self._write_csv(record)

        parts = [f"epoch {number:>3}/{total:<3}"]
        parts += [f"{key}={_value(val)}" for key, val in metrics.items()]
        if best:
            parts.append("*best")
        if seconds is not None:
            parts.append(f"time {format_duration(seconds)}")
        parts.append(f"elapsed {format_duration(time.time() - self.started)}")
        self.info("  ".join(parts))
        self._write_history()

    def checkpoint(self, value: float, path: Path | str) -> None:
        self.info(f"saved     macro_f1={_value(value)} -> {path}")

    def finish(self, summary: Mapping[str, Any]) -> Dict[str, Any]:
        """Close the run: summary.json, a final block, and one JSON line on stdout."""
        self.summary = dict(summary)
        payload: Dict[str, Any] = {
            "run_name": self.run_name,
            "log_dir": str(self.dir),
            "finished": datetime.now().strftime(WALL_FMT),
            "wall_seconds": round(time.time() - self.started, 1),
        }
        payload.update({k: v for k, v in summary.items()})
        if self.epochs:
            payload["last_epoch"] = self.epochs[-1]

        (self.dir / "summary.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        self._write_history()

        self.banner("done")
        for key, value in summary.items():
            self.info(f"{key} = {_value(value)}")
        self.info(
            f"wall {format_duration(payload['wall_seconds'])}   "
            f"artefacts -> {self.dir}"
        )

        final = json.dumps(dict(summary), sort_keys=True)
        if self.echo:
            print(final, flush=True)
        self._log_file.write(final + "\n")
        self.close()
        return payload

    # -- sinks ----------------------------------------------------------------

    def _write_jsonl(self, record: Mapping[str, Any]) -> None:
        self._jsonl_file.write(json.dumps(record, sort_keys=True) + "\n")

    def _write_csv(self, record: Mapping[str, Any]) -> None:
        if not self._csv_fields:
            self._csv_fields = list(record)
            with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, self._csv_fields).writeheader()
        extra = set(record) - set(self._csv_fields)
        if extra:
            self.warn(f"history.csv fixed schema, dropping new columns {sorted(extra)}")
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, self._csv_fields, extrasaction="ignore").writerow(record)

    def _history_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "run_name": self.run_name,
            "started": self.started_iso,
            "epochs": self.epochs,
        }
        payload.update(self.summary)
        return payload

    def _write_history(self) -> None:
        """One JSON document with every epoch so far; rewritten after each epoch.

        Unlike ``summary.json`` this is a complete history, so a job killed
        mid-training still leaves the epochs it managed to finish.
        """
        (self.dir / "history.json").write_text(
            json.dumps(self._history_payload(), indent=2) + "\n", encoding="utf-8"
        )

    # -- cleanup --------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._log_file.close()
        self._jsonl_file.close()

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


if __name__ == "__main__":
    # Smoke test: `python logging_utils.py` prints a fake three-epoch run.
    log = TrainingLogger(log_dir="/tmp/a2_logging_demo", run_name="demo")
    for e in range(1, 4):
        log.epoch(
            e,
            3,
            {"train_loss": 2.5 / e, "eval_loss": 2.2 / e, "macro_f1": 0.1 * e},
            best=e == 3,
            seconds=12.5,
        )
    log.checkpoint(0.3, "checkpoints/cnn_best.pt")
    log.finish({"best_macro_f1": 0.3})
    print("files:", sorted(p.name for p in Path("/tmp/a2_logging_demo/demo").iterdir()))
