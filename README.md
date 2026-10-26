# aist4010-hw2-kaggle

in this repo, we have the following files :

## config.py

this file contain the config of character -> integer mapping,
hyperparameters,
data preprocess config
training config,
etc. Any field can also be overridden on the command line
(`python train.py --help`).

## dataPreprocessor.py

read the FASTA files in ./data and map each amino acid into an integer id using
the `aa_alphabet` hashmap from config. Provides `Tokenizer`, `ProteinDataset`,
`TestDataset`, `collate` and `load_split`. (There is no class called
`DataPreprocessor`.)

## train.py

Here contain the training loop. Prints a progress line every 10% of an epoch
(`--log-every N` to change), one line per epoch, and one final JSON line.

## model.py

Here contain the absolute baseline model (a single-conv CNN + masked mean
pooling), plus `resolve_device()` which maps `--device auto` to cuda / mps / cpu.

## logging_utils.py

`TrainingLogger` writes one directory per run: `train.log`, `history.csv`,
`metrics.jsonl` and `summary.json`. It is deliberately **not** named
`logging.py` — that would shadow the standard library and make `import torch`
fail.

## gpu.slurm

CSE Slurm batch script that runs `train.py` on a GPU node inside the PyTorch
singularity image (below).

## predict.py

Loads the newest `checkpoints/*_best.pt`, runs it over `data/test.fasta` and
writes `submission.csv` in the same `id,label` format as
`data/sample_submission.csv`. Reuses the training tokeniser and the label space
stored in the checkpoint.

## submit.sh

One command that syncs this repo to the cluster and submits `gpu.slurm` (see
below). Run it locally, with the CSE VPN connected.

---

# how to submit the job to cse slurm

Everything below is run from your own machine unless marked **[cluster]**.
The repository directory on the cluster is `~/aist4010_hw2_kaggle-1`
(note the underscores).

## 0. one command (recommended)

```bash
./submit.sh                                   # config.py defaults
./submit.sh --epochs 10 --batch-size 64       # flags go to train.py
./submit.sh --follow --epochs 2 --limit 128   # wait, print output, fetch history
```

It rsyncs the code, submits `gpu.slurm`, and prints the job id plus the commands
to watch it. `--follow` waits for the job to finish, shows its tail, and copies
the run's logs back into `./logs`; `--fetch` does the copy on its own (useful
when the job already finished); `--no-sync` re-submits what is already on the
cluster. Overridable: `REMOTE` (default `linux1`),
`REMOTE_DIR` (`aist4010_hw2_kaggle-1`), `SSH_KEY` (`~/.ssh/id_ed25519`).

The sections below are the same steps by hand, plus the one-time image pull.

## 1. copy the code to the cse machine

```bash
KEY=~/Desktop/aist4010-hw2-kaggle/.ssh/id_ed25519     # your cluster key
rsync -az -e "ssh -i $KEY" \
  --exclude .git --exclude .ssh --exclude __pycache__ \
  --exclude checkpoints --exclude logs --exclude .DS_Store \
  ./ cyho2@linux1:aist4010_hw2_kaggle-1/
```

## 2. get into the cse linux login node

```bash
ssh -i "$KEY" cyho2@linux1        # from outside the campus network:
                                  # ssh cyho2@gw.cse.cuhk.edu.hk, then: ssh linux1
cd ~/aist4010_hw2_kaggle-1
```

## 3. pull the PyTorch image (once per GPU node)

The cluster has no PyTorch in `python3`, and the image (~3 GB) cannot live in
your home directory (quota is 2.4 GB), so it is cached in `/dev/shm` on one GPU
node and reused from there.

**[cluster]** — takes about 5 minutes; needs the CSE proxy, which is already
exported on the login node:

```bash
srun -w projgpu33 -p gpu_2h --qos=gpu --account=gpu --gres=gpu:1 -c 6 -t 00:40:00 bash -c '
  export SINGULARITY_CACHEDIR=/dev/shm/sgc SINGULARITY_TMPDIR=/dev/shm/stmp
  cd /dev/shm
  singularity pull --name pytorch-2.5.1.sif \
    docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime'
```

The image contains python 3.11.10, torch 2.5.1+cu121, numpy 2.1.2 and
scikit-learn 1.8.0, so `train.py` needs no extra `pip install`.

## 4. submit the job

**[cluster]** — skipped if you used `./submit.sh`.

```bash
sbatch gpu.slurm --epochs 10 --batch-size 64
# every flag of train.py is passed straight through, e.g.
sbatch gpu.slurm --epochs 10 --batch-size 64 --limit 512 --eval-limit 256
sbatch gpu.slurm --epochs 10 --batch-size 64 --tag v2
```

`gpu.slurm` pins `projgpu33` (that is where the image is), re-pulls the image if
it is missing, then runs:

```bash
singularity exec --nv /dev/shm/pytorch-2.5.1.sif python -u train.py --device cuda "$@"
```

## 5. watch the job

**[cluster]**

```bash
squeue -u cyho2                       # PD = queued, R = running
tail -f slurm-a2-<JOBID>.out          # sbatch -o slurm-%x-%j.out -> job name + id
sacct -j <JOBID> -o JobID,State,Elapsed,ExitCode   # after it finishes
```

The same lines are written into the run directory, so you can also watch from a
second shell while the job runs:

```bash
tail -f logs/cnn-*/train.log
```

## 6. where the logs are

Each run creates `logs/<arch>[-<tag>]-<YYYYmmdd-HHMMSS>/`:

| file | content |
| --- | --- |
| `train.log` | the console transcript with timestamps |
| `history.csv` | one row per epoch: `epoch,elapsed_s,train_loss,eval_loss,macro_f1,lr` |
| `history.json` | the whole history as one JSON document (rewritten every epoch) |
| `metrics.jsonl` | the same numbers as one JSON object per epoch |
| `summary.json` | run name, wall clock, `best_macro_f1`, last epoch |

Get the history back on your machine and read it as JSON:

```bash
./submit.sh --fetch                     # syncs <cluster>/logs/ into ./logs/

jq '.epochs' logs/cnn-*/history.json    # array of per-epoch objects
jq -r '.epochs[] | [.epoch, .train_loss, .macro_f1] | @tsv' logs/cnn-*/history.json
jq -r '.["best_macro_f1"]' logs/cnn-*/history.json
```

`history.json` is rewritten after every epoch, so a job that is cancelled or
hits the 2 h wall still leaves the epochs it finished. The fetch is a plain
`rsync` without `--delete`, so local runs are never removed by it.

The last line on stdout is still `{"best_macro_f1": ...}`, so the output stays
easy to parse. Checkpoints go to `checkpoints/<arch>[_tag]_best.pt`.

## 7. quick interactive test (optional)

For a one-off check it is fine to run in the foreground, but the job dies when
your ssh session closes:

```bash
srun -w projgpu33 -p gpu_2h --qos=gpu --account=gpu --gres=gpu:1 -c 6 -t 00:30:00 \
  singularity exec --nv /dev/shm/pytorch-2.5.1.sif \
  python -u train.py --device cuda --epochs 1 --batch-size 64 --limit 512 --eval-limit 256
```

## 8. things that go wrong

| message | cause | fix |
| --- | --- | --- |
| `py/bin/python3: No such file or directory` | `submit.slurm` / `job_gpu.slurm` expect a portable interpreter at `./py` that was never built | use `gpu.slurm` (they are obsolete) |
| `Requested node configuration is not available` | the GPU nodes report `RealMemory=1`, so any `--mem=` makes the job unschedulable | omit `--mem` |
| `CPU:GPU Ratio >6:1 ... state the features 'highcpucount'` | more than 6 CPUs per GPU | keep `--cpus-per-task<=6`, or add `#SBATCH --constraint=highcpucount` |
| `-w projgpu33` stays `PD` forever | that node is busy | drop `-w projgpu33`; the script re-pulls the image (~5 min) |
| `FATAL: container creation failed` / image not found | `/dev/shm` is per node, so the image exists only on the node that pulled it | re-run on `projgpu33`, or pull again |
| job dies instantly, empty `.out` | `cd` into the wrong directory (the name has underscores, not dashes) | `cd ~/aist4010_hw2_kaggle-1` |
| `quota exceeded` | home quota is 2.4 GB | `quota -s`, remove old `checkpoints/`, `logs/` |

## 9. inference -> submission.csv

The training job can run the predictor right after it saves the checkpoint:

```bash
./submit.sh --predict --follow --epochs 10 --batch-size 64
```

or, from the cluster:

```bash
PREDICT=1 sbatch gpu.slurm --epochs 10 --batch-size 64
```

`predict.py` picks the newest `checkpoints/*_best.pt` (so `--tag` variants work),
reads the preprocessing parameters back out of that checkpoint, and writes
`submission.csv` next to `train.py`. `./submit.sh --fetch` copies it back to your
machine.

Locally, without the cluster:

```bash
python predict.py --device cpu                        # newest checkpoint
python predict.py --checkpoint checkpoints/cnn_best.pt --out submission.csv
```

The output is `id,label` with integer class ids in `test.fasta` order, and the
script checks that order against `data/sample_submission.csv`:

```
wrote submission.csv (4469 rows)
  [ok] id order matches sample_submission.csv (4469 rows)
```
