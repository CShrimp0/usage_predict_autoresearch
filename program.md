# usage_predict autoresearch

This repo is a single-GPU autoresearch adaptation of the original `usage_predict` project.

## In-Scope Files

Read these files for context:

1. `README.md`
2. `prepare.py` — fixed harness, do not modify during experiments
3. `train.py` — the only file you edit

`prepare.py` owns:
- the fixed 5-minute training budget
- data loading and subject-level split logic
- the fixed evaluation harness
- output directory conventions

`train.py` owns:
- model architecture
- optimizer and scheduler
- regularization and EMA
- data augmentation knobs passed into the fixed harness
- training loop details

## Hardware Assumption

This machine has 48 GB of GPU VRAM available.

Use that headroom deliberately:
- explore larger batch sizes when they fit within the fixed 5-minute budget
- explore wider or deeper backbones when startup cost is acceptable
- explore richer multimodal fusion heads if they remain trainable and stable

Do not waste the extra VRAM, but do not accept unstable or overfit models just because they are larger.

## Setup

1. Work on a dedicated branch named `autoresearch/<tag>`.
2. Verify the data paths exist:
   - `/home/szdx/LNX/data/TA/Healthy/Images`
   - `/home/szdx/LNX/data/TA/characteristics.xlsx`
3. Initialize `results.tsv` if it does not exist yet with exactly this header:
   `commit	prediction_mae	memory_gb	status	description`

## Goal

The primary metric is `prediction_mae`, but model acceptance is not based on that number alone.

Each run trains for a fixed 5-minute wall-clock budget, then automatically evaluates the best checkpoint on the held-out prediction set and prints a compact summary.

Track both:
- `prediction_mae` as the main ranking metric
- `best_val_mae` as the generalization stability check

## Run Command

Every experiment runs on a single GPU:

```bash
conda activate us
CUDA_VISIBLE_DEVICES=0 /home/szdx/anaconda3/envs/us/bin/python train.py > run.log 2>&1
```

Key metrics can be extracted with:

```bash
grep "^prediction_mae:\|^best_val_mae:\|^peak_vram_mb:" run.log
```

## What You Can Change

- Anything in `train.py`

Typical changes:
- backbone choice
- regression head design
- auxiliary branch design
- optimizer or scheduler
- loss function
- dropout and regularization
- augmentation settings exposed through `ExperimentConfig`

## What You Cannot Change

- `prepare.py`
- the data split
- the evaluation metric
- external dependencies

## Reinforced Rules

### 1. Metric Weighting Rule

Never accept a model by looking at `prediction_mae` alone.

Acceptance policy:
- Double improvement: if `prediction_mae` and `best_val_mae` both improve, mark the experiment as `keep`.
- Overfit circuit breaker: if `prediction_mae` improves only slightly but `best_val_mae` degrades sharply, treat it as overfitting and mark it `discard`.
- Use a hard warning threshold of roughly 10% relative degradation in `best_val_mae` as an immediate discard signal, even if `prediction_mae` looks a bit better.
- Prefer robust models in the first tier of `prediction_mae` whose `best_val_mae` stays stable and does not diverge.

### 2. Crash Recovery Protocol

If a run crashes and `grep` returns no metrics:
- you must read the last 80 lines of `run.log`
- your next edit to `train.py` must focus only on fixing that concrete failure
- do not introduce a new modeling idea until the crash is resolved

Crash handling order:
1. inspect traceback
2. repair the exact failure
3. rerun
4. only then resume optimization

### 3. No Dependency Hallucination

Do not import or rely on libraries that are not already present in this environment.

Allowed ecosystem:
- Python standard library
- `torch`
- `torchvision`
- basic scientific stack already in the repo environment

Not allowed:
- `timm`
- `albumentations`
- `einops`
- any other new third-party package
- `pip install`

If you need a special block such as attention, a custom loss, or a fusion module, implement it directly inside `train.py`.

### 4. Strict TSV Formatting

When appending to `results.tsv`, write a pure tab-separated line only.

Never wrap the line in Markdown fences.
Never emit ```tsv or ``` around the record.

Required format:
`commit_hash	prediction_mae	memory_gb	status	description`

Example valid line:
`a1b2c3d	8.912340	12.5	keep	added spatial attention`

## Logging Results

Log every experiment to `results.tsv` as tab-separated values with these rules:
- use `0.000000` and `0.0` for crashes
- memory is `peak_vram_mb / 1024`, rounded to one decimal
- status is one of `keep`, `discard`, `crash`
- the description must be short, plain text, and contain no tabs
- append only a single raw TSV line per experiment result

## Experiment Loop

Loop forever:

1. Inspect the current branch and commit.
2. Change `train.py`.
3. Commit the change.
4. Run:

```bash
conda activate us
CUDA_VISIBLE_DEVICES=0 /home/szdx/anaconda3/envs/us/bin/python train.py > run.log 2>&1
```

5. Read the outcome:

```bash
grep "^prediction_mae:\|^best_val_mae:\|^peak_vram_mb:" run.log
```

6. If grep returns nothing, inspect the crash immediately:

```bash
tail -n 80 run.log
```

7. Append the result to `results.tsv` without committing that file.
8. Decide `keep` or `discard` using the Metric Weighting Rule, not `prediction_mae` alone.
9. If the run is `keep`, continue from that commit.
10. If the run is `discard` or `crash`, reset to the previous good commit after recording the result.

## Simplicity Rule

Small wins are only worth keeping if the code stays coherent.

Examples:
- Equal or better performance with simpler code: keep it.
- Tiny improvement with a lot of brittle logic: probably discard it.
- Worse performance with more complexity: discard it.

## Timeout Rule

- A normal run should finish a bit over 5 minutes because startup and final evaluation are outside the 5-minute training budget.
- If a run exceeds 10 minutes total, kill it and treat it as a failure.

## First Run

If the baseline is not yet recorded in `results.tsv`, the first run must be the current unmodified baseline.
