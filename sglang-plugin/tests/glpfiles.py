"""Shared by the offline tests: the import paths, where the published GLP
files are, and the synthetic GLP GGUF writer.

The writer is the vLLM plugin's (../vllm-plugin/tests/glpfiles.py), loaded
by file path so that both suites write the same files.

The tests that read the published GLP files (for example
huggingface.co/msuiche/Qwen3.8-27B-abliterated-cyber-GLP-49) need
WEIGHTLESS_TEST_GLP_DIR: the folder that holds them, or a folder of such
folders (several may be given, separated by os.pathsep). A file is looked
up in each folder and in its direct subfolders. Without it those tests
skip.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.dirname(HERE)
REPO = os.path.dirname(PLUGIN)
for p in (PLUGIN, os.path.join(REPO, "vllm-plugin"), REPO,
          os.path.join(REPO, "tools", "captain-vector")):
    if p not in sys.path:
        sys.path.insert(0, p)

_spec = importlib.util.spec_from_file_location(
    "vllm_plugin_glpfiles", os.path.join(REPO, "vllm-plugin", "tests", "glpfiles.py"))
_writer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_writer)
write_gguf, good_meta, good_tensors = _writer.write_gguf, _writer.good_meta, _writer.good_tensors

GLP_DIRS = [d.strip() for d in os.environ.get("WEIGHTLESS_TEST_GLP_DIR", "").split(os.pathsep)
            if d.strip()]
GLP_DIR = GLP_DIRS[0] if GLP_DIRS else None


def find_glp(name):
    """The first path to a GLP file called ``name`` under GLP_DIRS, or None."""
    for d in GLP_DIRS:
        cands = [os.path.join(d, name)]
        if os.path.isdir(d):
            cands += [os.path.join(d, sub, name) for sub in sorted(os.listdir(d))]
        for c in cands:
            if os.path.isfile(c):
                return c
    return None


GLP49 = find_glp("Qwen3.8-27B-abliterated-cyber-GLP-49-L10-58-a1.gguf")
GLP63 = find_glp("Qwen3.8-27B-abliterated-cyber-GLP-63-L1-63-a1.gguf")
# huggingface.co/msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44
GLM44 = find_glp("GLM-5.3-Flash-abliterated-cyber-GLP-44-L1-44-a2.gguf")


def have(*paths):
    """True when every path is set and exists."""
    return all(p and os.path.exists(p) for p in paths)
