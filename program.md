# usage_predict autoresearch

This repo is a single-GPU autoresearch adaptation of the original `usage_predict` project.

## In-Scope Files

Read these files for context:

1. `README.md`
2. `prepare.py` — fixed harness, do not modify during experiments
3. `train.py` — the only file you edit during routine autoresearch
4. `evaluate.py` — fixed confirmation tool for explicit final test evaluation

`prepare.py` owns:
- the fixed 5-minute training budget
- data loading and subject-level split logic
- the fixed output directory conventions

`train.py` owns:
- model architecture
- optimizer and scheduler
- regularization and EMA
- data augmentation knobs passed into the fixed harness
- training loop details

## Hardware Assumption

This machine has 48 GB of GPU VRAM available.

Treat extra VRAM as headroom, not as the optimization target.

Rules:
- Do not optimize for GPU memory usage itself.
- Do not increase batch size just to fill memory.
- Larger batches, wider heads, or heavier backbones are only worthwhile if they improve validation performance in the 5-minute budget.

## Current Baseline (The Golden Anchor)

The default `train.py` baseline should match this anchor:

- Backbone: `resnet50`
- Pretrained: `True`
- Batch Size: `8`
- Optimizer: `adamw`
- LR: `3.893e-05`
- Weight Decay: `4.914e-05`
- Dropout: `0.4125`
- Scheduler: `plateau`
- LR Factor: `0.696`
- LR Patience: `4`
- LR Min: `3.36e-06`
- Warmup Epochs: `5`
- EMA Decay: `0.999`
- Loss: `mae`

Start from this anchor unless `train.py` has drifted and first needs to be brought back into alignment.

## Goal

Routine autoresearch is validation-driven.

Primary search metric:
- `best_val_mae`

Reserved confirmation metric:
- `prediction_mae` on the held-out test split

Test-set policy:
- Do not use the test split to rank routine experiments.
- Do not peek at the test split after every run.
- Only run final test evaluation for a small number of shortlisted commits.

Shortlisting guidance:
- the current best validation commit
- a clearly better new validation commit
- a tiny-gain candidate that survives seed confirmation

## Run Commands

Routine search run:

```bash
conda activate us
CUDA_VISIBLE_DEVICES=0 /home/szdx/anaconda3/envs/us/bin/python train.py > run.log 2>&1
```

Routine metrics:

```bash
grep "^run_mode:\|^best_val_mae:\|^peak_vram_mb:" run.log
```

Explicit final test confirmation after a training run:

```bash
conda activate us
RUN_FINAL_EVAL=1 CUDA_VISIBLE_DEVICES=0 /home/szdx/anaconda3/envs/us/bin/python train.py --final-eval > run.log 2>&1
```

Or evaluate an existing checkpoint directly:

```bash
conda activate us
/home/szdx/anaconda3/envs/us/bin/python evaluate.py --checkpoint outputs/autoresearch/run_xxx/best_model.pth
```

## What You Can Change

- Anything in `train.py`

Typical high-value directions:
- local tuning around the golden anchor
- lightweight regression-head improvements
- lightweight auxiliary-branch or fusion improvements
- simple hand-written gating / SE-style modules when justified
- regularization or optimizer refinements that remain easy to reason about

## What You Cannot Change

- `prepare.py`
- the data split
- the evaluation metric definitions
- external dependencies

## Search Policy

### 1. Primary Metric

Routine keep/discard decisions are driven by `best_val_mae`.

During the main loop:
- prefer lower `best_val_mae`
- use `best_val_rmse` and training stability only as supporting context
- do not treat `prediction_mae` as a routine ranking signal

### 2. Candidate Confirmation

Only run final test evaluation for shortlisted commits.

Use it for:
- the current best validation commit
- a new contender that is clearly better on validation
- a borderline improvement that survives seed confirmation

Do not repeatedly evaluate minor variants on the test split.

### 3. Mutation Discipline

One run should make a small, interpretable change.

Default rule:
- at most 2 numeric hyperparameter changes per run
- at most 1 structural idea per run

Examples of structural ideas:
- fusion change
- regression head redesign
- SE / gating / FiLM-like modulation

Do not bundle many structural changes into one experiment.

### 4. Search Order

Use this order unless there is a concrete reason to deviate:

(Briefly) Verify the golden anchor is stable.

(High Priority) Structural innovations: lightweight fusion, hand-written gating (SE/FiLM), or head improvements.

Repeated-seed confirmation for small structural gains.

Only then consider larger structural departures.

If the current direction produces several consecutive discards and you no longer have a precise new hypothesis inside that family:
- pivot to a different well-motivated direction
- do not keep repeating near-identical variants just because they are nearby

Note: Do not waste excessive runs on pure numerical hyperparameter tuning; the golden anchor is already near optimal numerically.

### 5. Stability Rule

Do not trust tiny improvements immediately.

If a new run improves `best_val_mae` by less than `0.05` absolute:
- treat it as provisional
- rerun with an additional seed before calling it a true `keep`
- if the confirmation is mixed or disappears, discard it

If the gain is clearly larger than that threshold and the run is stable, you may keep it without extra confirmation.

### 6. Simplicity Bias

Equal performance with simpler code should win.

Guidance:
- equal or better validation with simpler code: keep it
- tiny gain with much more brittle logic: usually reject it
- more complexity with worse validation: discard it

### 7. Crash Recovery Protocol

If a run crashes and grep returns no metrics:
- read the last 80 lines of `run.log`
- fix that exact crash first
- do not propose a new modeling idea until the crash is resolved

### 8. No Dependency Hallucination

Allowed ecosystem:
- Python standard library
- `torch`
- `torchvision`
- existing scientific stack already present in the environment

Not allowed:
- `timm`
- `albumentations`
- `einops`
- any new third-party package
- `pip install`

If you need a special block or loss, implement it directly inside `train.py`.

## Logging (The Reflection Memory)

Historical experiments live in `results_v2.tsv`. This file acts as your long-term memory. It must capture not just *what* happened, but *why* it happened.

Append new records with this exact header:
`commit	best_val_mae	test_mae	status	mutation_type	action	insight`

**Column Definitions:**
- `best_val_mae`: The primary validation metric.
- `test_mae`: Optional held-out test confirmation metric for shortlisted candidates. Leave it empty for routine validation-only runs.
- `status`: `keep`, `candidate`, `discard`, or `crash`.
- `mutation_type`: Categorize your change with full words (`Architecture`, `Loss`, `Optimization`, `Regularization`, `Data`).
- `action`: What exact structural or numeric change did you make? (e.g., "Added CBAM after layer4", "Lowered base LR to 1e-5").
- `insight`: **(Crucial)** Why did it succeed or fail? What is the takeaway for future runs? (e.g., "Crash: tensor shape mismatch in fusion head" or "Discard: validation degraded by 15%, CBAM without extra dropout causes immediate overfitting").

Logging strict rules:
- Format as a single raw TSV line. No Markdown fences (```tsv).
- Do not use tabs (`\t`) or newlines (`\n`) inside the text fields (`action` and `insight`), replace them with spaces to keep the TSV format unbroken.
- For crashes, `best_val_mae` is `0.000000`.
- Leave `test_mae` empty unless an explicit final confirmation was run.

Never wrap TSV output in Markdown fences.

## Experiment Loop

Loop carefully using the Scientific Method:

1. **Analyze:** Read `results_v2.tsv` to understand past failures and successes (the `insight` column is your guide).
2. **Hypothesize:** Based on the insights, decide on ONE logical mutation (`Architecture`, `Loss`, `Optimization`, `Regularization`, or `Data`).
3. **Execute:** Edit `train.py` to implement the mutation.
4. **Commit:** Commit the change.
5. **Train:** Run the validation-only training command.
6. **Observe:** Read the outcome from `run.log`. If grep returns nothing, inspect `tail -n 80 run.log` to find the crash traceback.
7. **Reflect & Log:** - Analyze the metric gap between training and validation. 
   - Formulate an `insight` about why this mutation behaved the way it did.
   - Append the strictly formatted single TSV line to `results_v2.tsv`. Do NOT commit this file.
8. **Decide:** Evaluate `keep`, `candidate`, or `discard` using the Metric Weighting and Simplicity rules.
9. **Confirm:** If a run is `candidate`, confirm it with an additional seed before upgrading it to `keep`.
10. **Revert:** If a run is `discard` or `crash`, you MUST reset the code to the previous good commit (`git checkout HEAD -- train.py`) before starting the next loop.

## Timeout Rule

- Training budget is fixed at 5 minutes inside `prepare.py`.
- Startup and optional final evaluation sit outside that training budget.
- A routine run should still finish well under 10 minutes.
- If a run exceeds 10 minutes total, kill it and treat it as a failure.

## First Run

If the golden baseline is not yet recorded in `results_v2.tsv`, the first run should be the unmodified aligned baseline.
