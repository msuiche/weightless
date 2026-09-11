# tools/

Shipped product components — each one is a capability a weightless user can
reach for, with an API or byte-format contract other code depends on, its own
tests, and its own README. A tool is versioned and expected to grow.

Contrast with `scripts/`: those are operational automation for our own rig
and repo workflows (dashboard, router, watchdog, probes). A script does a job
for the operator; a tool is part of the product surface.

One folder per tool:

- `captain-vector/` — derives projective control vectors
  (difference-of-means with null-calibrated gates) and writes the `glp.*`
  GGUF files the serving hotfixes in `patches/` consume. Also backs the
  `weightless.py validate` verb.

If it has a contract someone else builds against, it belongs here. If it
automates our own chores, it belongs in `scripts/`.
