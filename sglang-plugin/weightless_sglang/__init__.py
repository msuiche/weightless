"""weightless-sglang: GLP projective activation steering as an SGLang plugin.

Applies  h <- h - alpha * (h . d_hat) d_hat  on the post-layer residual
stream of steered decoder layers (glp.hook_point =
residual_stream_post_layer; on DeepSeek-V4, the FFN write before the fold,
ffn_out_pre_residual; on DeepSeek-V4.1, the four hyper-connection copies after
the layer's post, with the coefficient of their mean), with per-layer unit
directions loaded from a GLP
GGUF control vector (spec: ../docs/spec/GLP.md). Same env contract, loader
and gates as the vLLM plugin (../vllm-plugin/weightless_steer); only the
wiring into the engine differs.
"""

__version__ = "0.1.0"
