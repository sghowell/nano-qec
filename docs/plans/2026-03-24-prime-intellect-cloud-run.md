# Prime Intellect H100 Cloud Run Implementation Plan

> For Hermes: use this plan to add a reproducible cloud execution workflow without changing the protected NanoQEC v0 CLI/schema contracts.

Goal: run canonical `local-d5-v1` training on a single Prime Intellect H100 instance while keeping local Mac execution as the correctness and validation lane.

Architecture: keep `prepare.py`, `train.py`, `eval.py`, artifact schemas, and the `RESULT` line unchanged. Treat cloud as a different execution host only. Use the same repo commit, same `uv` environment, and the same manifest/checkpoint/results layout currently used locally.

Tech stack: Ubuntu 22.04+, CUDA-capable NVIDIA driver, Python 3.11, `uv`, PyTorch, `rsync`/`scp`, git.

---

## Constraints from repo authority

- `AGENTS.md` says the current implementation target is local-only profile training using `uv`.
- `AGENTS.md` also says cloud GPUs remain deferred until the local harness is stable.
- Protected boundaries must remain unchanged unless docs, tests, and code are updated together:
  - `prepare.py` CLI behavior and dataset manifest schema
  - `eval.py` CLI behavior and evaluation output schema
  - shared artifact schemas and the `RESULT` line contract
  - smoke tests and validation commands
  - experiment promotion policy
- Therefore this plan intentionally changes execution environment, not experiment protocol.

## Recommended first instance shape

Pick one standard H100 instance with:

- 1x H100 80 GB
- 24+ vCPU preferred
- 64+ GB RAM minimum, 128 GB preferred
- 500 GB+ local NVMe preferred
- Ubuntu 22.04 or similar standard Linux image
- direct SSH access

Avoid multi-GPU and unusual container-only setups for the first migration.

## Local/cloud lane split

### Local Mac lane

Use local for:

- `uv sync --all-extras`
- `uv run ruff check .`
- `uv run pytest`
- tiny smoke runs
- command verification
- artifact inspection and comparison
- branch management and merges

### Cloud H100 lane

Use cloud for:

- canonical `local-d5-v1` training
- repeated candidate reruns
- promotion-grade `val1024` benchmarking once the profile is formalized in repo docs
- long ablations that are too slow locally

---

## Task 1: Record the exact cloud bootstrap procedure

Objective: define a fresh-instance bootstrap that is reproducible and does not depend on hidden machine state.

Files:
- Create: `docs/plans/2026-03-24-prime-intellect-cloud-run.md`

Step 1: SSH into the fresh instance

```bash
ssh ubuntu@<PRIME_INTELLECT_HOST>
```

Step 2: Verify the GPU and driver stack

```bash
nvidia-smi
```

Expected: one H100 is listed and the command exits successfully.

Step 3: Install OS packages needed for git, Python build support, and artifact sync

```bash
sudo apt-get update
sudo apt-get install -y git curl build-essential rsync unzip
```

Step 4: Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

Expected: `uv` prints a version string.

Step 5: Clone the repo and check out the intended branch/commit

```bash
mkdir -p ~/src
cd ~/src
git clone git@github.com:<ORG_OR_USER>/nano-qec.git
cd nano-qec
git fetch --all --prune
git checkout <BRANCH_OR_COMMIT>
```

Step 6: Install the pinned project environment

```bash
source "$HOME/.local/bin/env"
uv python install 3.11
uv sync --all-extras
```

Step 7: Sanity-check that PyTorch sees CUDA

```bash
uv run python - <<'PY'
import torch
print({
    'torch_version': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'cuda_device_count': torch.cuda.device_count(),
    'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
})
PY
```

Expected: `cuda_available` is `True` and the device name identifies an H100.

Step 8: Run the repo validation commands before the first paid run

```bash
uv run ruff check .
uv run pytest
```

Expected: all checks pass.

---

## Task 2: Run a cheap cloud smoke test before a real d=5 job

Objective: prevent wasting H100 time on setup or path mistakes.

Files:
- Uses existing entrypoints only: `prepare.py`, `train.py`, `eval.py`

Step 1: Build a tiny deterministic `local-d5-v1` dataset

```bash
uv run prepare.py \
  --workspace . \
  --profile local-d5-v1 \
  --train-shots 64 \
  --val-shots 32
```

Step 2: Train a tiny smoke run on CUDA

```bash
uv run train.py \
  --workspace . \
  --dataset-manifest data/local-d5-v1-d5-r5-5rates-train64-val32/manifest.json \
  --device cuda \
  --duration-seconds 15 \
  --eval-interval-seconds 5 \
  --batch-size 32 \
  --skip-experiment-log \
  --hypothesis "cloud smoke test"
```

Expected:
- training exits zero
- `checkpoints/best.pt` exists
- `results/train/<run_id>.json` exists
- stdout contains exactly one `RESULT {...}` line

Step 3: Evaluate the smoke checkpoint

```bash
uv run eval.py \
  --workspace . \
  --dataset-manifest data/local-d5-v1-d5-r5-5rates-train64-val32/manifest.json \
  --checkpoint checkpoints/best.pt \
  --device cuda
```

Expected:
- evaluation exits zero
- `results/eval/best-eval.json` exists
- stdout prints one JSON summary

---

## Task 3: Run the first canonical cloud d=5 training job

Objective: execute a real single-GPU d=5 run using the existing harness.

Files:
- Uses existing entrypoints only: `prepare.py`, `train.py`, `eval.py`

Step 1: Materialize the canonical `local-d5-v1` dataset

```bash
uv run prepare.py --workspace . --profile local-d5-v1
```

Expected manifest path:

```text
data/local-d5-v1-d5-r5-5rates-train512-val256/manifest.json
```

Step 2: Launch the first serious training run

```bash
uv run train.py \
  --workspace . \
  --dataset-manifest data/local-d5-v1-d5-r5-5rates-train512-val256/manifest.json \
  --device cuda \
  --batch-size 64 \
  --duration-seconds 1800 \
  --eval-interval-seconds 120 \
  --hypothesis "first Prime Intellect H100 local-d5-v1 baseline"
```

Notes:
- adjust `--batch-size` upward only after observing actual H100 memory headroom
- keep all changes at the CLI/config layer; do not edit protected contracts just to fit cloud

Step 3: Evaluate the best checkpoint explicitly

```bash
uv run eval.py \
  --workspace . \
  --dataset-manifest data/local-d5-v1-d5-r5-5rates-train512-val256/manifest.json \
  --checkpoint checkpoints/best.pt \
  --device cuda
```

Expected artifacts:
- `checkpoints/<run_id>.pt`
- `checkpoints/latest.pt`
- `checkpoints/best.pt`
- `results/train/<run_id>.json`
- `results/eval/best-eval.json`
- optionally `results/experiments.jsonl`

---

## Task 4: Pull artifacts back to the Mac and compare locally

Objective: keep analysis and promotion decisions on the local control machine.

Files:
- no repo changes required

Step 1: Sync the cloud run artifacts back

```bash
rsync -avz \
  ubuntu@<PRIME_INTELLECT_HOST>:~/src/nano-qec/checkpoints/ \
  /Users/seanhowell/dev/nano-qec/checkpoints/

rsync -avz \
  ubuntu@<PRIME_INTELLECT_HOST>:~/src/nano-qec/results/ \
  /Users/seanhowell/dev/nano-qec/results/
```

Step 2: Inspect metrics locally

```bash
uv run python - <<'PY'
import json
from pathlib import Path
train_dir = Path('results/train')
latest = max(train_dir.glob('*.json'), key=lambda p: p.stat().st_mtime)
payload = json.loads(latest.read_text())
print({
    'run_id': payload['run_id'],
    'dataset_id': payload['dataset_id'],
    'aggregate_val_ler': payload['aggregate_val_ler'],
    'aggregate_mwpm_ratio': payload['aggregate_mwpm_ratio'],
    'throughput_samples_per_second': payload['throughput_samples_per_second'],
})
PY
```

Step 3: Decide keep/discard on the documented benchmark policy, not on cloud novelty.

---

## Task 5: Add formal cloud support only after the gate is satisfied

Objective: avoid silently drifting from the current local-only v0 authority.

Files:
- Modify later, only if the gate below is met:
  - `AGENTS.md`
  - `docs/implementation-v0.md`
  - `docs/hermes-ops.md`
  - tests covering any changed behavior

Gate to declare the harness cloud-ready:
- validation commands pass locally and on cloud
- `prepare.py`, `train.py`, and `eval.py` behave identically across hosts
- artifact schemas match exactly
- the `RESULT` line matches exactly
- at least 2 repeated d=5 cloud runs complete without ad hoc manual fixes
- checkpoint load/eval works without code edits between runs
- there is a written runbook for bootstrap, smoke, train, eval, and artifact sync

Only after those are true should NanoQEC promote cloud from “deferred” to “supported execution environment” in the authority docs.

---

## Known pitfalls

- `uv` is required; do not substitute `pip`.
- Keep runtime artifacts out of git.
- Do not make cloud-specific changes to the protected CLI/schema contracts.
- Always do a tiny smoke run before a full H100 job.
- Prefer one H100 over multi-GPU for now.
- Keep exact manifest paths scripted or copied from `prepare.py` output to avoid typos.

## Verification checklist

A first-pass cloud workflow is acceptable when all of the following are true:

- `nvidia-smi` succeeds on the instance
- `uv sync --all-extras` succeeds on the instance
- `uv run ruff check .` succeeds on the instance
- `uv run pytest` succeeds on the instance
- a tiny `local-d5-v1` smoke prepare/train/eval cycle succeeds on CUDA
- a full `local-d5-v1` run produces checkpoints, metrics, and a `RESULT` line
- artifacts sync back to the Mac cleanly
- the local machine can inspect and compare the synced metrics
