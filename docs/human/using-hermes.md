# Using Hermes

This page explains how a human collaborator can use Hermes productively inside
NanoQEC without letting the agent drift outside the repository’s documented
rules. The goal is not to duplicate the Hermes runbook, but to give humans safe
starting patterns and clear escalation points.

## Implemented Today

As of April 9, 2026, Hermes is treated as a primary operator for experiment
execution inside the NanoQEC harness, but the repository docs and tests remain
the authority for behavior.

## Future Direction

Autonomous promotion, cron scheduling, and broader unattended operation remain
future phases. Human oversight is still part of the documented v0 model.

## Safe Starting Pattern

Start Hermes from the repo root so it can see the authority docs and local
artifacts.

Recommended preflight:

```bash
hermes model
hermes status
```

Then start an interactive session:

```bash
hermes chat
```

## Prompt Pattern That Keeps Hermes Grounded

Good prompt shape:

1. tell Hermes which authority docs to read first
2. name the bounded goal
3. name the protected surfaces it must not change
4. require a validation path before merge or promotion

Examples:

```text
Read AGENTS.md, docs/implementation-v0.md, and docs/hermes-ops.md. Summarize the operating rules you will follow before making any code changes.
```

```text
Read AGENTS.md, docs/implementation-v0.md, and docs/hermes-ops.md. Then propose one bounded local-d3-v1 experiment to improve aggregate MWPM ratio without changing protected schemas or CLI behavior.
```

```text
Read the NanoQEC authority docs, inspect results/experiments.jsonl, and summarize the best supported local-d3-v1 result with exact benchmark names and dates.
```

## When Hermes Is a Good Fit

- bounded experiment proposals
- result summarization from existing artifacts
- implementation work inside mutable model/training surfaces
- repo-local documentation work that does not override authority docs

## When to Escalate to Human Judgment

- benchmark-policy changes
- contract or schema changes
- promotion decisions
- anything that touches the protected boundaries described in
  [../../AGENTS.md](../../AGENTS.md)

## Exact Runbook

For the actual Hermes operator rules, read
[../hermes-ops.md](../hermes-ops.md). That document is the authority. This page
is only a human-oriented introduction.

## Next Reads

- [Contributing](./contributing.md)
- [FAQ](./faq.md)
- [../hermes-ops.md](../hermes-ops.md)
