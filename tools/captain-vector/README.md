<p align="center">
  <img src="logo.webp" alt="Captain Vector — the captain who steers your vectors" width="320">
</p>

# Captain Vector

Derives a projective control vector from a model plus a pair of prompt sets
(difference-of-means and friends, with a held-out-vs-null validation gate) and
writes it as a GGUF.

This is the **canonical producer** of the `glp.*` GGUF files that the serving
hotfixes in `weightless/patches/` consume. They gate on the `glp.*` metadata
keys, so the writer format here must not drift. `validate_gguf` is stdlib-only
on purpose — it runs where the file is served, no torch, no gguf package:

```sh
python3 ../../weightless.py validate some.gguf    # exit 1 on FAIL
```

Derivation itself needs `torch`, `transformers`, `safetensors`, `gguf`. The
full parameter reference and design rationale live in
`refusal-research/derivation/captain-vector/README.md`.

The derivation methodology gates — null calibration, prompt-driven capture,
ship gates — are specified in `refusal-research/METHODOLOGY.md` (§18 and the
sections it builds on) and are required practice, not suggestions.

- `captain_vector.py` — the library and its CLI (single file on purpose)
- `calibrate_null.py` — measures what held/null ratio pure noise reaches
- `test_captain_vector.py` — self-test script; also runs in `../../tests/`
- `examples/` — a form-matched, benign prompt-pair template
