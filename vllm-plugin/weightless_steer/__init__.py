"""weightless-steer — GLP projective activation steering as a vLLM plugin.

Applies  h <- h - alpha * (h . d_hat) d_hat  on the post-layer residual
stream of steered decoder layers, with per-layer unit directions loaded
from a GLP GGUF control vector (spec: ../spec/GLP.md). This package is the
plugin successor to the patches/hotfix-*-steering-projective.py fleet:
same loader gates, same CUDA-graph discipline, but installed through
vLLM's `vllm.general_plugins` entry point instead of rewriting model
files inside the container.
"""

__version__ = "0.1.0"
