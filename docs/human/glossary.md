# Glossary

This glossary collects the recurring NanoQEC terms that appear across the human
docs. It is meant to be scanned quickly when a term is familiar enough to
recognize but not yet stable in memory.

## Implemented Today

As of April 9, 2026, these terms describe the repository’s documented local
research flow and the evidence language used in its human-facing results.

## Future Direction

If the project grows into broader execution or benchmark regimes, this glossary
should expand with the docs. It should not invent future terminology early.

## Terms

**Aggregate MWPM ratio**  
The aggregate learned-decoder logical error rate divided by the aggregate MWPM
logical error rate on the same benchmark.

**Authority docs**  
The repo-local documents that define the exact rules and protected interfaces:
`AGENTS.md`, `docs/implementation-v0.md`, and `docs/hermes-ops.md`.

**Checkpoint**  
A serialized saved model state produced by `train.py`, including metadata.

**Continuity benchmark**  
A secondary benchmark used to check whether performance trends remain sensible
outside the primary decision lane.

**Dataset manifest**  
The manifest JSON produced by `prepare.py` that describes a prepared dataset
profile and the files behind it.

**`local-d3-v1`**  
The main small local research profile and the source of the promoted MWPM
result documented in this repo.

**`local-d5-v1`**  
A larger supported local profile with the same five-rate sweep structure.

**MWPM**  
Minimum-weight perfect matching, the baseline decoder used for comparison.

**Primary benchmark**  
The benchmark lane used for research and promotion decisions.

**Profile**  
A named experiment configuration that fixes distance, rounds, and physical
error-rate slices.

**`RESULT` line**  
The final machine-parseable stdout line emitted by `train.py`.

**`spacetime_gnn`**  
The graph-native decoder family behind the promoted `local-d3-v1` result.

## Next Reads

- [Concepts](./concepts.md)
- [Overview](./overview.md)
- [FAQ](./faq.md)
