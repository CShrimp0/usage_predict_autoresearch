#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="${1:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}"
BRANCH="autoresearch/${TAG}"
MODEL="${MODEL:-gpt-5.2}"
CODEX_BIN="${CODEX_BIN:-}"
MAX_EXPERIMENTS="${MAX_EXPERIMENTS:-}"

cd "${ROOT_DIR}"

if [[ -z "${CODEX_BIN}" ]]; then
  if command -v codex >/dev/null 2>&1; then
    CODEX_BIN="$(command -v codex)"
  else
    CODEX_BIN="$(ls -d "${HOME}"/.vscode-server/extensions/openai.chatgpt-*/bin/linux-x86_64/codex 2>/dev/null | sort | tail -1 || true)"
  fi
fi

if [[ -z "${CODEX_BIN}" || ! -x "${CODEX_BIN}" ]]; then
  echo "Codex executable not found. Set CODEX_BIN=/full/path/to/codex and retry." >&2
  exit 1
fi

current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${current_branch}" != "${BRANCH}" ]]; then
  if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    git checkout "${BRANCH}"
  else
    git checkout -b "${BRANCH}"
  fi
fi

if [[ ! -f results.tsv ]]; then
  printf "commit\tprediction_mae\tmemory_gb\tstatus\tdescription\n" > results.tsv
fi

if [[ ! -f results_v2.tsv ]]; then
  printf "commit\tbest_val_mae\ttest_mae\tstatus\tmutation_type\taction\tinsight\n" > results_v2.tsv
fi

PROMPT_FILE="$(mktemp)"
cleanup() {
  rm -f "${PROMPT_FILE}"
}
trap cleanup EXIT

cat > "${PROMPT_FILE}" <<'EOF'
Read `program.md` and follow it exactly.

You are running an autonomous single-GPU autoresearch loop in this repository.

Constraints:
- Only modify `train.py`
- Do not modify `prepare.py`
- Use validation metrics for routine keep/discard decisions
- Avoid repeated test peeking during routine search
- Only run final test evaluation for shortlisted candidates
- Start from the true golden baseline if the code is not aligned yet
- Prefer small, interpretable, controlled edits
- Limit routine mutations to at most two numeric hyperparameter changes or one structural idea per run
- Assume 48 GB VRAM is available, but do not optimize for memory usage itself
- Use `conda activate us` semantics implicitly by invoking `/home/szdx/anaconda3/envs/us/bin/python`
- Run experiments as `CUDA_VISIBLE_DEVICES=0 /home/szdx/anaconda3/envs/us/bin/python train.py > run.log 2>&1`
- Reuse `evaluate.py` or `RUN_FINAL_EVAL=1` only when explicitly confirming shortlisted candidates
- Append experiment outcomes to `results_v2.tsv` without committing that file
- If a run crashes, read the last 80 lines of `run.log` and fix that exact crash before trying a new idea
- Do not introduce any new third-party dependencies
- When appending to `results_v2.tsv`, write a single raw TSV line with no Markdown fences
- Keep simple robust improvements; reject noisy tiny gains unless they are confirmed

Begin with the baseline run if it is not yet recorded in `results_v2.tsv`, then continue experimenting.
EOF

if [[ -n "${MAX_EXPERIMENTS}" ]]; then
  printf "\nStop after %s completed experiment loops.\n" "${MAX_EXPERIMENTS}" >> "${PROMPT_FILE}"
fi

"${CODEX_BIN}" exec \
  --dangerously-bypass-approvals-and-sandbox \
  -m "${MODEL}" \
  -C "${ROOT_DIR}" \
  "$(cat "${PROMPT_FILE}")"
