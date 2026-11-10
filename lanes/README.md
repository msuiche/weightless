# lanes/

First-class lane files for third-party serving recipes — the primary path
for GLP deployment support. Each subdirectory is one lane:

```
lanes/<name>/weightless.json
```

Convention: [`docs/LANE.md`](../docs/LANE.md). Precedence: a lane file in
the recipe's own repo (`weightless.json` at its root) overrides the copy
here when both exist; most lanes will only ever exist here.

Adding a lane: one `weightless.json`, verified against the recipe at the
`last_verified` date stamped inside. Minimal fields: `schema`, `name`, one
`glp.placement`, `commands.up`. The launcher refuses unknown schema, vector
sha mismatches, and failed `check` commands — never falls back.

| lane | recipe | hook served |
|---|---|---|
| [spark-ds41f](spark-ds41f/) | [christopherowen/spark-ds41f](https://github.com/christopherowen/spark-ds41f) | `hc_fold_pre_engram`, `residual_stream_post_layer` |
| [ringside-glm53](ringside-glm53/) | [othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE](https://github.com/othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE) | `residual_stream_post_layer` (GLP-44, TP4) |
| [x00byte-dual-spark](x00byte-dual-spark/) | [x00byte/DeepSeek-V4.1-Flash-1017-Dual-Spark](https://github.com/x00byte/DeepSeek-V4.1-Flash-1017-Dual-Spark) (via the msuiche fork until #1 merges) | `hc_fold_pre_engram` |
