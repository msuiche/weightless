# Patch layout

- `hotfix-<lane>-steering-projective.py` — the per-lane boot hotfixes that
  apply GLP projective steering inside each serving image. The filename is
  load-bearing: start scripts, `DEPLOY_MAP`, and fail-closed boot checks
  reference it, and several are linked from HF model cards and blog posts.
  Do not rename.
- `reference/` — vendored copies of the upstream model files the structure
  tests diff the hotfixes against (`tests/structure/test-*`).
- `vendor/` — third-party fixes we bind-mount over image files (provenance
  header inside each file).
- `0001-*.patch`, `0002-*.patch` — the retired v0.27.0 overlay and its test,
  kept as git patches for the fallback stack.
