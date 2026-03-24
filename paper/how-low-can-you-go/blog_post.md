# How low can you go? Speedrunning AlphaQubit with Hermes Agent

We wanted to answer a simple question:

Can an autonomous coding agent take a local quantum error-correction decoder from “clearly worse than MWPM” to “actually beating MWPM,” using only a well-structured research repo and repeated overnight experimentation?

The answer, in this case, was yes.

This post tells the story of how Hermes Agent improved the local-d3-v1 decoder in NanoQEC from an AQ2-style baseline with aggregate MWPM ratio around **1.84** all the way down to a final promoted regime with repeated primary-benchmark aggregate MWPM ratio around **0.959**.

That means the learned decoder beat MWPM on repeated evidence on the project’s primary benchmark.

## The setup

NanoQEC is a harness-first repo for autonomous neural quantum error-correction experiments. The repository is designed so that the public contracts stay stable while the internals can evolve:

- `prepare.py` creates deterministic train/validation caches
- `train.py` trains a decoder and emits structured metrics
- `eval.py` evaluates a checkpoint against the fixed benchmark profile

The target profile in this work was `local-d3-v1`, a small rotated-memory surface-code setup with five physical error rates:

- 0.001
- 0.003
- 0.005
- 0.007
- 0.01

The operational goal was simple:

> drive the learned decoder’s aggregate logical error rate low enough that its aggregate MWPM ratio reaches 1.0 or below.

## The baseline: AQ2 is good, but not good enough

The starting point was an AQ2-style decoder family:

- detector-aware embeddings
- gated recurrence over time
- per-bucket spatial transformer

With some modest scheduling improvements, that baseline sat around **1.84x MWPM**.

It worked, but it was nowhere near parity.

## The first big lesson: data and time matter

The biggest early jump didn’t come from architecture. It came from scale:

- more train shots
- a longer fixed-budget training window

Moving to **4096 training shots** and **120 seconds** cut the aggregate ratio to roughly **1.33**.

That was an important result because it showed the baseline wasn’t just under-designed; it was under-trained.

Still, that only got us so far.

## The second big lesson: benchmark policy matters

For a while we were making decisions on a small validation benchmark (`val256`).

That turned out to be noisy enough to distort the search. Some slices looked much worse than they really were, and some apparently “good” moves turned out to be flukes.

So the project adopted a new policy:

- **primary benchmark:** train8192 / val1024
- **continuity benchmark:** train8192 / val256

That change mattered a lot. It made the search more stable, and it changed which hypotheses looked worth pursuing.

## The graph prior phase

Once the easy gains from data and training time were exhausted, the real progress came from introducing graph structure.

First we added lightweight local message passing only in the final AQ2 block. That helped.

Then we learned a very specific lesson:

> graph structure works best as a late, gentle prior rather than a heavy-handed early override.

That led to:

- final-block message passing
- a mild gated second pass
- a slightly wider local neighborhood (`k=3`)

Those variants pushed the primary benchmark into the **~1.26** range.

At that point, the model family was clearly stronger, but still not at parity.

## The breakthrough: stop treating graph structure as a correction

The real step change came when we stopped treating graph reasoning as a small correction to AQ2 and made it the backbone.

That produced a new model family: **SpacetimeGNN**.

Instead of letting recurrence and transformers do almost everything, then patching in local graph information at the end, the new model reasoned directly over detector nodes with:

- spatial graph structure
- temporal graph structure
- repeated typed graph updates

This is what finally moved the project from “good graph-augmented decoder” to “genuinely graph-native decoder.”

## The final promoted regime

The final winning setup was:

- model: `spacetime_gnn`
- graph blocks: 6
- feedforward multiplier: 6
- train shots: 8192
- training budget: 180 seconds

Repeated primary-benchmark results:

- seed 1: 0.942
- seed 2: 0.957
- seed 3: 0.978
- mean: **0.959**

That beat MWPM on repeated evidence on the primary benchmark.

On the smaller continuity benchmark, the final model was still slightly above parity on average, but close enough that the project’s benchmark-policy decision was clearly justified.

## What actually mattered

In hindsight, the major gains came in this order:

1. More data and more training time
2. Cleaner train-only weighting
3. Lightweight graph priors
4. Fuller graph-native architecture
5. Only then more capacity inside the graph-native model

One of the clearest lessons is that **capacity only started to pay off once the inductive bias was right**.

Bigger AQ2-style models were usually noisy or disappointing.
Bigger graph-native models actually helped.

## What didn’t work

A lot of things did not matter much:

- generic learning-rate fiddling
- simple width/depth sweeps in the old family
- naive dropout
- longer training without the right architecture
- larger datasets without enough convergence time
- overly aggressive graph bias

These experiments were still useful. They helped rule out dead ends and clarify what the bottleneck really was.

## The systems lesson

This project was not only about the final decoder. It was also about whether an autonomous agent could run a real open-ended ML improvement loop inside a constrained repo.

The answer depended heavily on the repo design.

What made it work:

- stable public contracts
- reproducible datasets and metrics
- explicit benchmark policy
- machine-readable outputs
- clear promotion rules

Without that structure, the agent would have had much more room to thrash or silently drift the experiment protocol.

## The science lesson

The deepest scientific lesson is simple:

> graph-native inductive bias was the key to crossing the MWPM threshold.

Not more generic transformer capacity.
Not another optimizer tweak.
Not more clever slice weighting alone.

The problem needed the model to reason about error structure in a way that better matched the geometry of the code.

Once the architecture started doing that, the remaining improvements became much more tractable.

## Where this leaves us

We reached the target on the primary benchmark, so the original speedrun objective was achieved.

But the work also points to a natural next phase.

If we wanted to keep going, the next serious directions would be more graph-neural still:

- learned edge-feature MLPs
- deeper edge/node update stacks
- explicit temporal graph edges across buckets
- hybrid graph encoder plus logical-readout head

That would be a new research phase, not just more tweaking.

## Final takeaway

Starting from an AQ2-style baseline at about **1.84x MWPM**, Hermes Agent eventually drove the primary-benchmark ratio down to **0.959x MWPM**.

The path there was not a straight line, and it was not just “search over hyperparameters until something works.”

It required:

- the right benchmark policy
- enough data and time
- multiple rounds of failed ideas
- and, ultimately, a real shift in inductive bias from sequence-centric decoding to graph-native decoding

That was the difference between getting closer to MWPM and actually beating it.

## Figures

Useful companion figures from the paper draft are available in:

- `figures/milestone_progress.pdf`
- `figures/final_slice_ratios.pdf`
- `figures/architecture_overview.pdf`
- `figures/benchmark_policy.pdf`
- appendix architecture diagrams in the same folder
