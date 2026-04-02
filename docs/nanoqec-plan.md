# NanoQEC: Autoresearch for Neural Quantum Error Correction

**March 2026**

---

## 0. Executive Summary

NanoQEC adapts the autoresearch pattern (Karpathy, 2026) to neural decoding for quantum error correction, but the implemented repository now does so as a harness-first local research stack. Repo-local docs, schemas, tests, and entrypoints are authoritative. Hermes is intended to be the primary operator inside that harness, not the source of truth for it. The current implemented target is a local single-host workflow using `uv`, deterministic profile preparation, a minimal AlphaQubit 2-style recurrent-transformer baseline, structured evaluation against MWPM, and machine-readable experiment logging.

The current decoder family is a deliberately simplified AQ2: detector-aware embeddings, interleaved gated temporal blocks and spatial transformer layers, temporal compression of syndrome frames, profile-aware conditioning, and calibrated logical readout. Training data is generated from Stim's `surface_code:rotated_memory_x` circuits with depolarizing noise. PyMatching provides the MWPM baseline. PyTorch is the training framework. The present research focus is to improve decoder quality on the local `d=3` and `d=5` profiles before expanding to mixed-distance training or cloud execution.

Hermes integration is now repo-local and adapter-based. `AGENTS.md`, `docs/implementation-v0.md`, and `docs/hermes-ops.md` are the operating authority. `SKILL.md` and `program.md` are thin pointers back to those docs. Automatic promotion, cron scheduling, Prime Intellect/cloud execution, and overnight autonomy remain future phases that are intentionally deferred until the local harness is stronger and Hermes runtime configuration is proven healthy.

### 0.1 Current Status Note

As of March 2026, the repository already includes:

- a `uv`-managed local package scaffold
- deterministic `local-d3-v1` and `local-d5-v1` profile preparation
- public `prepare.py`, `train.py`, and `eval.py` entrypoints
- a minimal AQ2-style baseline plus an alternate model path
- aggregate and per-slice evaluation, MWPM comparisons, and runtime experiment logs in `results/experiments.jsonl`
- repeated tuning utilities and CI

This document remains the long-horizon strategy and architecture note. For current operational behavior, follow `AGENTS.md`, `docs/implementation-v0.md`, and `docs/hermes-ops.md`.

---

## 1. System Architecture

### 1.1 Information Flow

The current system centers on repo-local authority docs, a human or Hermes
operator, stable public entrypoints, and runtime artifacts. Cloud and scheduler
layers remain optional future extensions.

```
┌─────────────────────────────────────────────────────────────────┐
│                   REPO-LOCAL AUTHORITY                          │
│  AGENTS.md | implementation-v0.md | hermes-ops.md | tests/CI   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                HERMES OR HUMAN OPERATOR                         │
│   SKILL.md and program.md are thin adapters into the repo docs  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
     ┌──────────────┐ ┌────────┐ ┌──────────┐
     │  prepare.py  │ │train.py│ │ eval.py  │
     │ (data/cache) │ │(model) │ │(metrics) │
     └───────┬──────┘ └───┬────┘ └────┬─────┘
             │             │           │
             └─────────────┴───────────┘
                           │
                           ▼
     ┌──────────────────────────────────────────────────────────┐
     │   data/ | checkpoints/ | results/experiments.jsonl      │
     │   PyTorch runtime on CPU/MPS today, cloud later         │
     └──────────────────────────────────────────────────────────┘
```

### 1.2 Autoresearch Loop

Each experiment cycle follows this sequence:

1. **Hermes or a human reads the repo-local authority docs** (`AGENTS.md`, `docs/implementation-v0.md`, `docs/hermes-ops.md`) and proposes a bounded change inside the mutable model/training surface.
2. **A short-lived git branch is created** and the diff is applied.
3. **`prepare.py`** loads or materializes deterministic profile data from Stim.
4. **`train.py`** trains the decoder for a fixed wall-clock time budget and emits a structured `RESULT` line.
5. **`eval.py`** evaluates aggregate and per-slice logical error rate (LER) against the fixed validation profile and MWPM baseline.
6. **The run is marked keep or discard** and a runtime record may be appended to `results/experiments.jsonl`.
7. **If a change is promoted**, it is validated, merged locally, revalidated on `main`, and then pushed. Auto-merge remains disabled in v0.
8. **Repeat**

### 1.3 Separation of Concerns

| File | Owner | Role |
|------|-------|------|
| `prepare.py` | Protected harness entrypoint | Deterministic data preparation and manifest generation |
| `train.py` | Stable public entrypoint | Training CLI that delegates to mutable internals under `src/nanoqec/` |
| `eval.py` | Protected harness entrypoint | Evaluation harness, metric computation, and MWPM comparison |
| `src/nanoqec/` | Mutable implementation surface | Model architecture, optimizer, training loop, layouts, and internal loaders |
| `AGENTS.md` + `docs/` | Human-maintained authority | Repo-local operating contracts, schemas, and mutation rules |
| `program.md` + `SKILL.md` | Thin adapters | Pointers back to the repo-local authority docs |

---

## 2. Decoder Architecture (Minimal AQ2)

### 2.1 Design Principles

The minimal AQ2 decoder retains the core architectural innovations of AlphaQubit 2 while dramatically reducing scale for single-GPU training:

- **Per-stabilizer representation**: Each of the `d^2 - 1` stabilizers gets a learned vector (embedding dimension 64, vs 128/256 in full AQ2)
- **Interleaved temporal and spatial layers**: Lightweight gated recurrence for temporal updates, transformer attention for spatial mixing
- **Temporal compression**: Group multiple syndrome cycles before processing (group size 2-4 for small distances)
- **Mean-pooled readout**: Simple mean pooling over stabilizer representations followed by an MLP head predicting logical error probability
- **Spatial RoPE**: Rotary position embeddings encoding stabilizer grid coordinates

### 2.2 Architecture Specification

```
Input: syndrome detection events  [batch, rounds, n_stabilizers]
       (optional: soft readouts)   [batch, rounds, n_stabilizers, 2]

Embedding:
  - Linear projection per stabilizer → [batch, rounds, n_stab, d_model=64]
  - Add learned stabilizer index embedding
  - Add spatial RoPE based on (row, col) coordinates

Temporal Compression:
  - Group consecutive rounds (group_size=2)
  - Concatenate + project: [batch, rounds//2, n_stab, d_model]

Core Network (repeat N_blocks=2 times):
  - GatedRecurrence layer (per-stabilizer, shared params)
      gate = sigmoid(W_g · [h_{t-1}, x_t] + b_g)
      h_t  = gate * h_{t-1} + (1 - gate) * tanh(W_h · [h_{t-1}, x_t] + b_h)
  - SpatialTransformer layers (repeat 2 per block)
      - Multi-head self-attention (4 heads, d_head=16) with spatial RoPE
      - LayerNorm + residual
      - FFN (d_model → 4*d_model → d_model, GELU)
      - LayerNorm + residual

Readout:
  - Take final time-step hidden states: [batch, n_stab, d_model]
  - Mean pool over stabilizers: [batch, d_model]
  - MLP: d_model → d_model → 1 (sigmoid)
  - Output: P(logical error)

Loss: Binary cross-entropy

Parameter count (d=5, d_model=64):
  Embedding: ~5K
  Core (2 blocks × (1 RNN + 2 Transformer)): ~200K
  Readout: ~4K
  Total: ~210K parameters
```

### 2.3 Scaling Knobs for the Agent

These are the primary variables the agent should explore:

| Knob | Default | Range | Notes |
|------|---------|-------|-------|
| `D_MODEL` | 64 | 32-256 | Embedding dimension per stabilizer |
| `N_BLOCKS` | 2 | 1-4 | Number of (RNN + Transformer) blocks |
| `N_HEADS` | 4 | 2-8 | Attention heads |
| `N_TRANSFORMER_PER_BLOCK` | 2 | 1-4 | Transformer layers per block |
| `GROUP_SIZE` | 2 | 1-4 | Temporal compression factor |
| `FFN_MULT` | 4 | 2-8 | FFN hidden dimension multiplier |
| `DROPOUT` | 0.0 | 0.0-0.2 | Regularization |
| `LR` | 3e-4 | 1e-5 to 1e-2 | Learning rate |
| `BATCH_SIZE` | 256 | 64-2048 | Batch size |
| `OPTIMIZER` | AdamW | AdamW, Muon, Lion | Optimizer choice |
| `WEIGHT_DECAY` | 0.01 | 0.0-0.1 | Regularization |

---

## 3. Data Pipeline

### 3.1 Syndrome Generation with Stim

All training data is generated synthetically using Google's Stim simulator. This is the same foundation used by the AlphaQubit papers (which used Stim circuits plus custom Pauli+ noise models). For our minimal setup, we use Stim's built-in depolarizing noise model.

```python
import stim
import numpy as np

def generate_syndrome_data(
    distance: int = 5,
    rounds: int = 5,
    p_error: float = 0.005,
    n_shots: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate syndrome detection events and logical observable labels."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p_error,
        before_round_data_depolarization=p_error,
        after_reset_flip_probability=p_error,
        before_measure_flip_probability=p_error,
    )
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(
        shots=n_shots, separate_observables=True
    )
    return detection_events.astype(np.float32), observable_flips.astype(np.float32)
```

### 3.2 Data Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Distances | 3, 5, (7) | d=3 for fast iteration, d=5 for meaningful signal, d=7 aspirational |
| Rounds | d (match distance) | Standard practice for memory experiments |
| Physical error rates | 0.001 to 0.01 (log-spaced, 8 values) | Covers sub-threshold to near-threshold regime |
| Training shots per config | 50K-200K | Enough for convergence at small d |
| Validation shots per config | 10K | Held-out evaluation |
| Total training samples | ~2-5M | Across all distances and error rates |

### 3.3 Data Loading Strategy

- **On-the-fly generation** for initial experiments (Stim is fast enough for d ≤ 7)
- **Cached to disk** as `.npz` files after first generation
- **Mixed-distance training**: Each batch samples uniformly across distances and error rates
- **Detection event reshaping**: Reshape flat detector array into `[rounds, n_stabilizers]` grid using Stim's coordinate metadata

### 3.4 Baseline: PyMatching

PyMatching (MWPM decoder) provides the comparison baseline:

```python
import pymatching

def mwpm_baseline(circuit, detection_events, observable_flips):
    """Compute MWPM logical error rate for comparison."""
    model = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)
    predictions = matching.decode_batch(detection_events)
    n_errors = np.sum(np.any(predictions != observable_flips, axis=1))
    return n_errors / len(detection_events)
```

---

## 4. Training Setup

### 4.1 Fixed Time Budget

Following the autoresearch pattern, each experiment trains for a fixed wall-clock time budget. This makes experiments comparable regardless of what the agent changes (model size, batch size, etc.).

| Platform | Time Budget | Expected Throughput |
|----------|-------------|---------------------|
| M4 Pro (MPS) | 3 minutes | ~50K-200K samples/min at d=5 |
| Prime Intellect A100 | 5 minutes | ~500K-1M samples/min at d=5 |
| Prime Intellect H100 | 5 minutes | ~1M-2M samples/min at d=5 |

### 4.2 Training Loop Skeleton

```python
import torch
import time

# Fixed constants (from prepare.py)
TIME_BUDGET_SECONDS = 300  # 5 minutes, or 180 for MPS
EVAL_INTERVAL = 30         # Evaluate every 30 seconds
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda"

# ===== AGENT MODIFIES BELOW =====
D_MODEL = 64
N_BLOCKS = 2
LR = 3e-4
BATCH_SIZE = 256
OPTIMIZER = "lion"
# ... (full model definition)
# ===== AGENT MODIFIES ABOVE =====

model = build_model(...)
optimizer = build_optimizer(model, lr=LR, kind=OPTIMIZER)

start_time = time.time()
step = 0
best_val_ler = float('inf')

while time.time() - start_time < TIME_BUDGET_SECONDS:
    batch = dataloader.next_batch(BATCH_SIZE)
    loss = train_step(model, optimizer, batch)
    step += 1

    if time.time() - start_time > next_eval_time:
        val_ler = evaluate(model, val_data)
        log_metrics(step, loss, val_ler)
        next_eval_time += EVAL_INTERVAL

# Final evaluation and threshold calibration
decision_threshold = calibrate_threshold(model, train_data)
final_val_ler = evaluate(model, val_data, threshold=decision_threshold)
print(
    'RESULT {"run_id":"...","val_ler":%.6f,"mwpm_ratio":1.0,"kept":false}'
    % final_val_ler
)
```

### 4.3 Evaluation Metric

The primary metric is **aggregate validation logical error rate (val_ler)** — the mean of the validation logical error rate across all slices in the prepared profile manifest. Lower is better. For each slice, LER is computed as:

```
val_ler = (1/N) * sum(I[decoder_prediction != true_observable])
```

where the decoder's sigmoid output is thresholded using the calibrated global
decision threshold stored in the checkpoint. If no threshold metadata is
available, the fallback threshold is 0.5.

Secondary metrics (logged but not used for keep/discard):
- val_ler per physical error rate slice
- Comparison ratio: aggregate val_ler / aggregate mwpm_ler (< 1.0 means beating MWPM)
- Training throughput (samples/second)
- Parameter count

---

## 5. Hermes Agent Integration

### 5.1 Integration Architecture

Hermes now integrates with NanoQEC through three repo-local layers:

1. **Authority docs**: `AGENTS.md`, `docs/implementation-v0.md`, and `docs/hermes-ops.md` define the contracts, protected boundaries, and workflow.
2. **Thin adapters**: `SKILL.md` and `program.md` point Hermes back to the repo-local instructions instead of carrying an independent protocol.
3. **Stable entrypoints**: Hermes operates through `uv` and the public `prepare.py`, `train.py`, and `eval.py` commands plus machine-readable artifacts.

Local single-host execution is the implemented mode today. Cron scheduling, overnight autonomy, and cloud backends remain deferred future phases.

### 5.2 Repo-Local Adapter Structure

```
nanoqec/
├── AGENTS.md
├── docs/
│   ├── implementation-v0.md
│   ├── hermes-ops.md
│   └── nanoqec-plan.md
├── SKILL.md
├── program.md
└── scripts/
    ├── check_improvement.py
    ├── plot_progress.py
    └── tune_profile.py
```

### 5.3 Adapter Content

`SKILL.md` and `program.md` are intentionally short. Their job is to redirect
Hermes into the repo-local authority docs, not to duplicate a second full
protocol. The practical experiment protocol is now:

1. Read `AGENTS.md`, `docs/implementation-v0.md`, and `docs/hermes-ops.md`.
2. Work on a short-lived branch.
3. Mutate only the allowed implementation surface behind the protected harness.
4. Run the documented `uv` validation and experiment commands.
5. Use `results/experiments.jsonl`, structured metrics JSON, and the final `RESULT` line for comparison.
6. Keep or discard the branch. In v0, merges remain explicit and local rather than automatic.

### 5.4 Future Autonomous Operation

Once the local baseline is meaningfully stronger and Hermes runtime/provider
health is stable, the same harness can be extended to:

- scheduled overnight runs
- cloud or SSH-backed execution
- stronger promotion rules
- eventually, guarded automation around promotion
- an optional richer Hermes skill bundle derived from the repo-local authority docs for convenience and distribution, but not as an independent source of truth

### 5.5 Hermes Memory Integration

After each experiment, Hermes's memory system will naturally accumulate:
- Which architectural changes helped and which didn't
- Platform-specific knowledge (MPS quirks, memory limits)
- Effective hyperparameter ranges
- Common failure modes and fixes

This accumulated knowledge can improve experiment quality over time, but the
repo-local docs and tests still remain authoritative.

---

## 6. Compute Strategy

### 6.1 Apple Silicon (M4 Pro, Primary Dev)

| Resource | Specification |
|----------|---------------|
| Device | M4 Pro (MPS backend) |
| Memory | 24-48 GB unified |
| Time budget | 3 minutes per experiment |
| Batch size | 128-256 (limited by MPS overhead) |
| Max distance | d=5 comfortably, d=7 tight |
| MPS caveats | No flash attention; use `torch.nn.functional.scaled_dot_product_attention` with `enable_math=True` fallback |

Key settings for MPS:
```python
if DEVICE == "mps":
    # MPS-specific optimizations
    torch.mps.set_per_process_memory_fraction(0.8)
    # Disable MPS graph capture for debugging
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
```

### 6.2 Cloud GPU: Prime Intellect (Future Scaling Platform)

Prime Intellect is a compute exchange that aggregates GPU resources from 12+ cloud providers into a unified marketplace with competitive, transparent pricing. Their mission to democratize and commoditize compute aligns with Zetetic Works' open-source neolab ethos. Prime Intellect also builds open-source distributed training infrastructure (PRIME-RL, OpenDiLoCo) and trains open models (INTELLECT-1/2/3), making them a natural partner for open AI-for-science work.

| GPU (via Prime Intellect) | VRAM | Cost/hr | NanoQEC Fit |
|---------------------------|------|---------|-------------|
| RTX 4090 | 24GB | ~$0.32 | Excellent for d=3,5 experiments |
| A6000 | 48GB | ~$0.41 | Good headroom for d=7 |
| A100 80GB | 80GB | ~$0.79 | Ideal for d=7+ and large batch sizes |
| H100 80GB | 80GB | ~$1.49 | Maximum throughput for burst experiments |

**Recommended GPU for NanoQEC**: The **A100 80GB at $0.79/hr** is the sweet spot. Our ~210K parameter decoder is tiny enough that the A100 is never compute-bottlenecked, and the 80GB VRAM allows very large batch sizes (2048+) and data caching entirely in GPU memory. For budget-conscious overnight runs, the RTX 4090 at $0.32/hr handles d=3,5 comfortably. For burst experiments pushing to d=7+, the H100 at $1.49/hr provides maximum throughput.

**Hermes Agent integration**: Hermes can eventually connect to Prime Intellect instances via an SSH terminal backend, but that is not part of the current local v0 workflow.

### 6.3 Recommended Workflow

1. **Develop and tune locally** on M4 Pro or CPU using the fixed `local-d3-v1` and `local-d5-v1` profiles
2. **Stabilize the baseline locally** until the aggregate `mwpm_ratio` is meaningfully below the current local baseline and trending toward 1.0
3. **Run a real Hermes dry-run locally** once the provider/runtime path is healthy
4. **Only then scale outward** to cloud GPUs, longer runs, or overnight execution

Cloud cost planning remains relevant for later scaling, but it is not part of the current implementation contract.

---

## 7. Project Structure

```
nanoqec/
├── AGENTS.md               # Operational authority
├── pyproject.toml          # Package metadata and dependencies
├── .python-version         # 3.11
├── uv.lock                 # Locked dependency graph
├── prepare.py              # Thin public prepare entrypoint
├── train.py                # Thin public train entrypoint
├── eval.py                 # Thin public eval entrypoint
├── README.md               # Quickstart and validation commands
├── SKILL.md                # Thin Hermes adapter
├── program.md              # Thin program pointer
├── docs/
│   ├── implementation-v0.md
│   ├── hermes-ops.md
│   └── nanoqec-plan.md
├── src/nanoqec/            # Mutable implementation surface
├── scripts/                # Research utilities and tuning harness
├── tests/                  # Smoke and contract coverage
├── .github/workflows/ci.yml
├── data/                   # Gitignored prepared profile artifacts
├── checkpoints/            # Gitignored saved model weights
└── results/                # Gitignored metrics, plots, and experiments.jsonl
```

### 7.1 Dependencies

```toml
[project]
name = "nanoqec"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "matplotlib>=3.8,<4",
    "numpy>=1.26,<3",
    "pymatching>=2.2,<3",
    "stim>=1.14,<2",
    "torch>=2.2,<3",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3,<9",
    "ruff>=0.11,<0.12",
]
```

---

## 8. Implementation Status And Next Phases

The original bootstrap plan is largely complete. The repo no longer needs a
from-scratch implementation pass; it needs decoder improvement and carefully
sequenced expansion.

### Phase 0: Harness And Repo Authority

- [x] Add repo-local authority docs and thin Hermes adapters
- [x] Standardize on `uv`, locked dependencies, and Python 3.11
- [x] Add CI, tests, and a package scaffold under `src/nanoqec/`

### Phase 1: Deterministic Local Data Pipeline

- [x] Implement profile-based `prepare.py` for `local-d3-v1` and `local-d5-v1`
- [x] Cache deterministic train/validation slices and manifests
- [x] Compute and store MWPM baselines per slice and in aggregate

### Phase 2: Baseline Decoder And Evaluation

- [x] Implement a minimal AQ2-style decoder with a stable train/eval contract
- [x] Save reloadable checkpoints with model-spec metadata
- [x] Report aggregate and per-slice metrics plus MWPM comparisons
- [x] Support at least one alternate model path to prove architecture replaceability

### Phase 3: Research Utilities And Tuning

- [x] Add experiment comparison and progress plotting utilities
- [x] Add a repeated tuning harness for fixed-profile config comparison
- [x] Promote the current stronger local baseline defaults based on repeated runs

### Phase 4: Remaining Near-Term Work

- [ ] Improve the local `d=3` baseline until aggregate `mwpm_ratio < 1.0`
- [ ] Tune and evaluate `local-d5-v1` using the same repeated-run discipline
- [ ] Run a real Hermes repo dry-run once provider/runtime configuration is healthy
- [ ] Harden promotion criteria once the decoder is materially stronger

### Phase 5: Deferred Expansion

- [ ] Mixed-distance training in a single run
- [ ] Cloud or Prime Intellect execution
- [ ] Cron or overnight autonomous operation
- [ ] Stronger guarded automation around promotion
- [ ] Larger-distance and broader-code exploration

---

## 9. Success Criteria

| Milestone | Criterion | Status |
|-----------|-----------|--------|
| M0: Harness Runs | `prepare.py`, `train.py`, and `eval.py` complete a local profile cycle without error | Complete |
| M1: Harness Is Reproducible | deterministic manifests, tests, CI, and runtime experiment logs exist | Complete |
| M2: Beats MWPM at d=3 | aggregate `mwpm_ratio < 1.0` on the local `d=3` profile | Open |
| M3: Hermes Dry-Run | Hermes completes a repo-local dry-run with healthy provider/runtime config | Open |
| M4: Beats MWPM at d=5 | aggregate `mwpm_ratio < 1.0` on the local `d=5` profile | Open |
| M5: Mixed Distance | one training run spans multiple distances under the same harness | Deferred |
| M6: Scaling | stable cloud or overnight operation under guarded promotion rules | Deferred |
| M7: Insight | the research loop discovers and validates a non-obvious improvement | Ongoing |

---

## 10. Open Source Stack Summary

| Component | Tool | License | Role |
|-----------|------|---------|------|
| Syndrome simulation | Stim (Google) | Apache 2.0 | Generate training data |
| Baseline decoder | PyMatching | Apache 2.0 | MWPM comparison |
| Training framework | PyTorch | BSD | Neural network training |
| Compute platform | Prime Intellect | Proprietary (open-source tooling) | Future GPU scaling and training infra |
| Agent framework | Hermes Agent (Nous) | Apache 2.0 | Autonomous experiment driver |
| Package manager | uv (Astral) | MIT | Dependency management |
| Version control | Git | GPL | Experiment branching |
| LLM backend | Any (via OpenRouter) | Various | Agent intelligence |

Current required infrastructure cost is local-only. Cloud cost planning remains a future scaling concern.

---

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| MPS backend instability | Training crashes | Catch exceptions, retry with reduced batch size; fall back to CPU for small models |
| Agent makes breaking changes | Lost progress | Git branching ensures `main` always has a working version |
| Overfitting at small d | Misleading metrics | Track train/val gap; use mixed-distance training |
| Stim data not representative | Poor generalization | Use multiple noise models; add noise augmentation |
| Hermes adapter/runbook too rigid | Agent wastes experiments | Keep repo authority in `AGENTS.md` and `docs/`; keep `SKILL.md` thin |
| Compute too limited for d=7 | Can't reach M5 | Focus on d=3,5 insights; scale to Prime Intellect A100/H100 for d=7 push |

---

## Appendix A: Key References

1. Bausch et al. "A scalable and real-time neural decoder for topological quantum codes" (AlphaQubit 2). arXiv:2512.07737, Dec 2025.
2. Bausch et al. "Learning high-accuracy error decoding for quantum processors" (AlphaQubit 1). Nature 635, 834-840, Nov 2024.
3. Gidney. "Stim: a fast stabilizer circuit simulator." Quantum 5, 497, Jul 2021.
4. Higgott & Gidney. "Sparse Blossom: correcting a million errors per core second with minimum-weight matching." (PyMatching). Quantum 9, 1600, 2025.
5. Lee, Hur & Park. "Scalable Neural Decoders for Practical Real-Time Quantum Error Correction." arXiv:2510.22724, Oct 2025.
6. Karpathy. "autoresearch." GitHub, Mar 2026.
7. Nous Research. "Hermes Agent." GitHub, Feb 2026.

## Appendix B: Glossary

- **Detection event**: The XOR of consecutive stabilizer measurements; 1 indicates a change (potential error)
- **Logical error rate (LER)**: Probability that the decoder incorrectly predicts the logical observable
- **MWPM**: Minimum-weight perfect matching, the standard classical decoder
- **Surface code**: A topological QEC code defined on a 2D grid of qubits
- **Syndrome**: The set of stabilizer measurement outcomes at a given time step
- **Distance (d)**: The code distance; a distance-d code can correct floor((d-1)/2) errors
- **Rotary Position Embedding (RoPE)**: Position encoding via rotation matrices applied to query/key vectors
- **Gated recurrence**: An RNN variant using element-wise gating (similar to GRU but lighter)
