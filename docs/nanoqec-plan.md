# NanoQEC: Autoresearch for Neural Quantum Error Correction

**Zetetic Works Research Corporation**
**March 2026**

---

## 0. Executive Summary

NanoQEC adapts the autoresearch pattern (Karpathy, 2026) to neural decoding for quantum error correction. An autonomous agent (Hermes Agent, Nous Research) iteratively modifies a minimal AlphaQubit 2-style recurrent-transformer decoder, trains it on Stim-generated surface code syndrome data under a fixed time budget, evaluates against a held-out validation set, and keeps or discards each change based on logical error rate improvement. The entire stack is open source, runs on a single GPU (M4 Pro MPS or a Prime Intellect cloud A100/H100), and produces a research log of experiments while you sleep.

The decoder architecture is a deliberately simplified AQ2: interleaved lightweight gated-recurrence layers and spatial transformer layers operating on per-stabilizer representations, with temporal compression of syndrome frames and mean-pooled readout. Training data is generated on-the-fly from Stim's `surface_code:rotated_memory_x` circuits with depolarizing noise. PyMatching provides the MWPM baseline. PyTorch is the training framework, with MPS backend for Apple Silicon and CUDA for cloud GPUs.

Hermes Agent drives the autonomous research loop via a custom `nanoqec` skill, replacing the `program.md` pattern from autoresearch with a richer skill document that includes syndrome-generation scripts, evaluation harness, and experiment logging. Hermes's persistent memory accumulates knowledge across experiment runs, and its cron scheduler enables overnight autonomous operation.

---

## 1. System Architecture

### 1.1 Information Flow

The system has four major subsystems connected by a simple data flow:

```
┌─────────────────────────────────────────────────────────────────┐
│                      HERMES AGENT                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Skill:   │  │ Memory   │  │ Cron     │  │ Experiment│       │
│  │ nanoqec  │  │ (FTS5)   │  │ Scheduler│  │ Log (JSON)│       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │              │             │              │              │
│       └──────────────┴─────────────┴──────────────┘              │
│                          │                                       │
└──────────────────────────┼───────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
     ┌──────────────┐ ┌────────┐ ┌──────────┐
     │  prepare.py   │ │train.py│ │ eval.py  │
     │ (Stim data)   │ │(model) │ │(metrics) │
     └───────┬──────┘ └───┬────┘ └────┬─────┘
             │             │           │
             ▼             ▼           ▼
     ┌──────────────────────────────────────┐
     │        PyTorch Runtime               │
     │   MPS (Apple Silicon) │ CUDA (Cloud) │
     └──────────────────────────────────────┘
```

### 1.2 Autoresearch Loop

Each experiment cycle follows this sequence:

1. **Hermes reads `train.py`** and proposes a modification (architecture, hyperparameters, optimizer, data augmentation, etc.)
2. **Hermes creates a git branch** and applies the diff
3. **`prepare.py`** generates fresh training and validation syndrome data from Stim (or loads cached data)
4. **`train.py`** trains the decoder for a fixed wall-clock time budget (configurable, default 5 minutes)
5. **`eval.py`** evaluates logical error rate (LER) on validation data across multiple physical error rates and distances
6. **Hermes compares** val_ler against the current best; if improved, the change is merged to `main`
7. **Hermes logs** the experiment (hypothesis, diff, metrics, outcome) to `experiments.jsonl`
8. **Repeat**

### 1.3 Separation of Concerns

| File | Owner | Role |
|------|-------|------|
| `prepare.py` | Human (fixed) | Stim circuit generation, data loading, evaluation utilities |
| `train.py` | Agent (modified) | Model architecture, optimizer, training loop, hyperparameters |
| `eval.py` | Human (fixed) | Evaluation harness, metric computation, baseline comparison |
| `program.md` | Human (iterated) | Autoresearch instructions for the agent |
| `SKILL.md` | Human (iterated) | Hermes Agent skill for the nanoqec research loop |

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
OPTIMIZER = "adamw"
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

# Final evaluation
final_val_ler = evaluate(model, val_data)
print(f"RESULT val_ler={final_val_ler:.6f}")
```

### 4.3 Evaluation Metric

The primary metric is **validation logical error rate (val_ler)** — the fraction of validation syndrome samples where the decoder incorrectly predicts the logical observable. Lower is better. This is computed as:

```
val_ler = (1/N) * sum(I[decoder_prediction != true_observable])
```

where the decoder's sigmoid output is thresholded at 0.5.

Secondary metrics (logged but not used for keep/discard):
- val_ler per distance (d=3, d=5 separately)
- val_ler per physical error rate
- Comparison ratio: val_ler / mwpm_ler (< 1.0 means beating MWPM)
- Training throughput (samples/second)
- Parameter count

---

## 5. Hermes Agent Integration

### 5.1 Integration Architecture

Hermes Agent drives NanoQEC through three integration points:

1. **Skill**: A `nanoqec` skill in `~/.hermes/skills/research/nanoqec/` provides the agent with domain knowledge about QEC, the AQ2 architecture, and the experiment protocol
2. **Terminal backend**: Hermes executes `train.py` and `eval.py` via its terminal tool (local, Docker, or SSH to a Prime Intellect pod)
3. **Cron scheduling**: Hermes runs experiments autonomously overnight via its built-in scheduler

### 5.2 Skill Structure

```
~/.hermes/skills/research/nanoqec/
├── SKILL.md                    # Main instructions (the "program.md" equivalent)
├── references/
│   ├── ARCHITECTURE.md         # AQ2 architecture details and design rationale
│   ├── QEC_PRIMER.md           # Surface code and decoding background
│   └── EXPERIMENT_PROTOCOL.md  # Step-by-step experiment execution guide
├── scripts/
│   ├── check_improvement.py    # Compare new results against current best
│   └── plot_progress.py        # Generate progress charts
└── assets/
    └── baseline_results.json   # Pre-computed MWPM baselines for comparison
```

### 5.3 SKILL.md Content

```yaml
---
name: nanoqec
description: >
  Autonomous neural QEC decoder research. Run experiments that modify
  train.py in a NanoQEC repo, train a minimal AlphaQubit 2-style decoder
  on Stim-generated surface code syndrome data, and evaluate logical error
  rate improvement. Use when the user mentions nanoqec, QEC experiments,
  decoder training, or autonomous research on quantum error correction.
version: 1.0.0
author: zetetic-works
license: Apache-2.0
platforms: [macos, linux]
metadata:
  hermes:
    tags: [research, quantum, ml, autonomous]
    category: research
    requires_toolsets: [terminal]
---

# NanoQEC: Autonomous Neural QEC Decoder Research

## When to Use
- User says "run a nanoqec experiment" or "start autonomous QEC research"
- User asks to improve the decoder or try a new architecture idea
- Scheduled cron job triggers an experiment batch

## Experiment Protocol

### Setup (first run only)
1. cd to the nanoqec repo directory
2. Run `uv sync` to install dependencies
3. Run `uv run prepare.py` to generate initial data caches
4. Run `uv run train.py` to verify baseline training works
5. Run `uv run eval.py` to compute MWPM baselines

### Each Experiment
1. Read `train.py` and `experiments.jsonl` to understand current state
2. Formulate a hypothesis (e.g., "increasing d_model from 64 to 96
   should improve capacity for d=5 decoding")
3. Create a git branch: `git checkout -b exp-{NNN}-{short-description}`
4. Modify ONLY `train.py` — do not touch prepare.py or eval.py
5. Run: `uv run train.py 2>&1 | tee /tmp/train_output.txt`
6. Parse the final RESULT line for val_ler
7. Compare against current best in experiments.jsonl
8. If improved: merge to main, update best
9. If not improved: record result, discard branch
10. Append experiment record to experiments.jsonl

### Experiment Record Format
```json
{
  "id": 42,
  "timestamp": "2026-03-21T04:23:00Z",
  "hypothesis": "Increase d_model from 64 to 96",
  "diff_summary": "+D_MODEL = 96  -D_MODEL = 64",
  "val_ler": 0.0423,
  "val_ler_d3": 0.0112,
  "val_ler_d5": 0.0734,
  "mwpm_ratio": 0.87,
  "params": 340000,
  "throughput_samples_sec": 125000,
  "kept": true,
  "wall_time_sec": 300
}
```

### Research Priorities (ordered)
1. Get a working baseline that trains and evaluates correctly
2. Beat MWPM at d=3 (mwpm_ratio < 1.0)
3. Beat MWPM at d=5
4. Explore architectural variants: attention patterns, normalization,
   activation functions, positional encodings
5. Explore optimizer variants: AdamW, Lion, schedule-free AdamW
6. Explore data efficiency: curriculum learning, noise scheduling
7. Explore temporal compression strategies
8. Push to d=7 if compute allows

### What NOT to Do
- Do not modify prepare.py or eval.py
- Do not change the evaluation metric or time budget
- Do not install additional pip packages without asking
- Do not try distributed training or multi-GPU setups
- Keep changes atomic: one hypothesis per experiment

## Pitfalls
- MPS backend does not support all CUDA operations; use
  `torch.backends.mps.is_available()` for device detection
- Stim data generation is CPU-bound; cache aggressively
- Small models at low d can overfit quickly; watch train vs val gap
- Binary cross-entropy requires careful numerical stability
  (use `torch.nn.functional.binary_cross_entropy_with_logits`)

## Verification
- Experiment succeeds if train.py runs without error and prints
  a RESULT line
- An experiment is "kept" if val_ler < current_best_val_ler
- Periodically run plot_progress.py to visualize the research arc
```

### 5.4 Autonomous Operation via Cron

Configure Hermes Agent to run experiments overnight:

```
# In Hermes CLI or messaging:
/cron "Every 10 minutes from 11pm to 7am, run a nanoqec experiment
       using the experiment protocol. Use the nanoqec skill."
```

Expected throughput: ~48 experiments per 8-hour overnight session (at 5-minute time budget + 5-minute overhead per experiment).

### 5.5 Hermes Memory Integration

After each experiment, Hermes's memory system will naturally accumulate:
- Which architectural changes helped and which didn't
- Platform-specific knowledge (MPS quirks, memory limits)
- Effective hyperparameter ranges
- Common failure modes and fixes

This accumulated knowledge improves experiment quality over time without explicit programming.

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

### 6.2 Cloud GPU: Prime Intellect (Primary Platform)

Prime Intellect is a compute exchange that aggregates GPU resources from 12+ cloud providers into a unified marketplace with competitive, transparent pricing. Their mission to democratize and commoditize compute aligns with Zetetic Works' open-source neolab ethos. Prime Intellect also builds open-source distributed training infrastructure (PRIME-RL, OpenDiLoCo) and trains open models (INTELLECT-1/2/3), making them a natural partner for open AI-for-science work.

| GPU (via Prime Intellect) | VRAM | Cost/hr | NanoQEC Fit |
|---------------------------|------|---------|-------------|
| RTX 4090 | 24GB | ~$0.32 | Excellent for d=3,5 experiments |
| A6000 | 48GB | ~$0.41 | Good headroom for d=7 |
| A100 80GB | 80GB | ~$0.79 | Ideal for d=7+ and large batch sizes |
| H100 80GB | 80GB | ~$1.49 | Maximum throughput for burst experiments |

**Recommended GPU for NanoQEC**: The **A100 80GB at $0.79/hr** is the sweet spot. Our ~210K parameter decoder is tiny enough that the A100 is never compute-bottlenecked, and the 80GB VRAM allows very large batch sizes (2048+) and data caching entirely in GPU memory. For budget-conscious overnight runs, the RTX 4090 at $0.32/hr handles d=3,5 comfortably. For burst experiments pushing to d=7+, the H100 at $1.49/hr provides maximum throughput.

**Hermes Agent integration**: Hermes Agent connects to Prime Intellect instances via its SSH terminal backend. Provision a pod on Prime Intellect's dashboard or CLI (`prime pods create --name nanoqec`), configure SSH credentials in Hermes, and the agent can execute training runs remotely while you sleep.

### 6.3 Recommended Workflow

1. **Develop and debug locally** on M4 Pro with d=3, 1-minute time budget
2. **Run overnight experiments locally** on M4 Pro with d=3,5, 3-minute time budget
3. **Scale to Prime Intellect A100** for d=5,7 experiments with 5-minute time budget
4. **Burst on Prime Intellect H100** when exploring a promising direction at d=7+

Total overnight cost: $0 (local M4 Pro) to ~$6 (8 hours on Prime Intellect A100 at $0.79/hr).

---

## 7. Project Structure

```
nanoqec/
├── pyproject.toml          # Dependencies: torch, stim, pymatching, numpy
├── .python-version         # 3.11
├── prepare.py              # Data generation + evaluation utilities (DO NOT MODIFY)
├── train.py                # Model + training loop (AGENT MODIFIES THIS)
├── eval.py                 # Evaluation harness (DO NOT MODIFY)
├── program.md              # Autoresearch instructions (human-iterated)
├── experiments.jsonl       # Experiment log (append-only)
├── data/                   # Cached syndrome data (.npz files)
│   ├── d3_r3_p0.001.npz
│   ├── d3_r3_p0.003.npz
│   └── ...
├── checkpoints/            # Saved model weights
│   ├── best_model.pt
│   └── latest_model.pt
├── results/                # Evaluation results and plots
│   ├── progress.png
│   └── ler_vs_p.png
└── .hermes/                # (Optional) local Hermes context files
    └── context.md          # Project-specific context for Hermes
```

### 7.1 Dependencies

```toml
[project]
name = "nanoqec"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "stim>=1.14",
    "pymatching>=2.2",
    "numpy>=1.26",
    "matplotlib>=3.8",
]
```

---

## 8. Implementation Plan

Working backward from the goal of autonomous overnight experiments:

### Phase 0: Environment Setup (1-2 hours)

- [ ] Install Hermes Agent (`curl -fsSL ... | bash`)
- [ ] Configure Hermes with preferred LLM backend (recommend Claude Sonnet 4.6 via OpenRouter or Nous Portal)
- [ ] Install Prime Intellect CLI: `uv tool install prime && prime login`
- [ ] Create `nanoqec/` repo with `uv init`
- [ ] Install dependencies: `uv add torch stim pymatching numpy matplotlib`
- [ ] Verify Stim and PyTorch work on target device (MPS locally; CUDA on Prime Intellect pod)
- [ ] Install the `nanoqec` skill to `~/.hermes/skills/research/nanoqec/`

### Phase 1: Data Pipeline (2-3 hours)

- [ ] Implement `prepare.py`:
  - Stim circuit generation for d=3,5 with configurable noise
  - Syndrome data generation and caching to `data/`
  - DataLoader class that yields batches of (detection_events, observable_flips)
  - Detector coordinate extraction for reshaping flat arrays to grid
  - MWPM baseline computation and caching
- [ ] Verify: generate 100K samples at d=3, p=0.005; confirm shapes and label distribution

### Phase 2: Baseline Decoder (3-4 hours)

- [ ] Implement initial `train.py`:
  - `SyndromeEmbedding` module (linear projection + index embedding + spatial RoPE)
  - `GatedRecurrence` module (element-wise gated RNN per stabilizer)
  - `SpatialTransformer` module (multi-head attention + FFN)
  - `TemporalCompression` module (concatenate + project)
  - `ReadoutHead` module (mean pool + MLP)
  - `NanoQECDecoder` model composing the above
  - Training loop with fixed time budget
  - Metric logging and RESULT output
- [ ] Verify: train on d=3 for 1 minute, confirm loss decreases and val_ler is reported

### Phase 3: Evaluation Harness (1-2 hours)

- [ ] Implement `eval.py`:
  - Load best model checkpoint
  - Evaluate across all distances and error rates
  - Compute MWPM comparison ratio
  - Generate LER vs physical error rate plots
  - Output structured JSON results
- [ ] Verify: eval produces results and plots; compare neural decoder to MWPM at d=3

### Phase 4: Autoresearch Loop (2-3 hours)

- [ ] Write `program.md` with experiment protocol
- [ ] Write full `SKILL.md` and supporting reference documents
- [ ] Implement `scripts/check_improvement.py` (parse RESULT, compare to best)
- [ ] Implement `scripts/plot_progress.py` (read experiments.jsonl, plot LER over time)
- [ ] Set up git repo with initial commit
- [ ] Test one full experiment cycle manually via Hermes

### Phase 5: Autonomous Operation (1 hour + overnight)

- [ ] Configure Hermes cron for overnight experiments
- [ ] Run first overnight batch
- [ ] Review experiment log and progress plot in the morning
- [ ] Iterate on `program.md` / `SKILL.md` based on agent behavior

### Phase 6: Scale and Explore (ongoing)

- [ ] Extend to d=7 on Prime Intellect A100/H100
- [ ] Explore color code circuits (AQ2's other target)
- [ ] Add soft readout support (analog syndrome values)
- [ ] Benchmark against published neural decoder results

---

## 9. Success Criteria

| Milestone | Criterion | Estimated Timeline |
|-----------|-----------|-------------------|
| M0: Runs | train.py completes one experiment without error | Day 1 |
| M1: Learns | Neural decoder val_ler improves with training (loss decreases) | Day 1 |
| M2: Beats MWPM at d=3 | mwpm_ratio < 1.0 at d=3, p=0.005 | Day 2-3 |
| M3: Autonomous | Hermes runs 10+ experiments without human intervention | Day 2-3 |
| M4: Beats MWPM at d=5 | mwpm_ratio < 1.0 at d=5, p=0.005 | Week 1-2 |
| M5: Scaling | Successful training and evaluation at d=7 | Week 2-4 |
| M6: Insight | Agent discovers a non-obvious architectural improvement | Ongoing |

---

## 10. Open Source Stack Summary

| Component | Tool | License | Role |
|-----------|------|---------|------|
| Syndrome simulation | Stim (Google) | Apache 2.0 | Generate training data |
| Baseline decoder | PyMatching | Apache 2.0 | MWPM comparison |
| Training framework | PyTorch | BSD | Neural network training |
| Compute platform | Prime Intellect | Proprietary (open-source tooling) | GPU marketplace and training infra |
| Agent framework | Hermes Agent (Nous) | Apache 2.0 | Autonomous experiment driver |
| Package manager | uv (Astral) | MIT | Dependency management |
| Version control | Git | GPL | Experiment branching |
| LLM backend | Any (via OpenRouter) | Various | Agent intelligence |

Total infrastructure cost for overnight research: $0 (local M4 Pro) to ~$6 (8 hours on Prime Intellect A100).

---

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| MPS backend instability | Training crashes | Catch exceptions, retry with reduced batch size; fall back to CPU for small models |
| Agent makes breaking changes | Lost progress | Git branching ensures `main` always has a working version |
| Overfitting at small d | Misleading metrics | Track train/val gap; use mixed-distance training |
| Stim data not representative | Poor generalization | Use multiple noise models; add noise augmentation |
| Hermes skill too rigid | Agent wastes experiments | Iterate on SKILL.md based on experiment logs; keep priorities loose |
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
