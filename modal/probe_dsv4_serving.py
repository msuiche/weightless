"""CPU-only probe: why did the day-0 image serve thinking despite
--chat-template with the </think> splice? Inspects the image's OpenAI
serving code and the checkpoint's tokenizer config. No GPU.

    modal run probe_dsv4_serving.py
"""
import modal

IMAGE = "vllm/vllm-openai:deepseekv4-flash-vision"
app = modal.App("weightless-dsv4-serving-probe")
vol = modal.Volume.from_name("dsv4-0731", create_if_missing=False)

image = modal.Image.from_registry(IMAGE, add_python="3.12",
                                  setup_dockerfile_commands=["ENTRYPOINT []"])


@app.function(image=image, volumes={"/data": vol}, timeout=900)
def probe():
    import glob
    import subprocess

    def sh(cmd):
        print(f"$ {cmd}", flush=True)
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(r.stdout, r.stderr, sep="", flush=True)

    sh("/usr/bin/python3.12 -c \"import vllm; print('vllm', vllm.__version__)\"")
    sh("sed -n '180,220p' /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py")
    sh("grep -n 'def resolve_chat_template_kwargs' -A 60 /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/chat_utils.py | head -80")
    snaps = glob.glob("/data/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731"
                      "/snapshots/*")
    print("snapshots:", snaps, flush=True)
    if snaps:
        sh(f"ls {snaps[0]}")
        sh(f"grep -c chat_template {snaps[0]}/tokenizer_config.json || true")
        sh(f"ls {snaps[0]}/chat_template.jinja 2>/dev/null || "
           "echo 'no chat_template.jinja file'")
