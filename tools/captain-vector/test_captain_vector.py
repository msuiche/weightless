#!/usr/bin/env python3
"""Tests for the parts that can be wrong silently.

Each targets a failure that produces a plausible number rather than an error --
the only kind worth a test here.
"""
import sys, json, tempfile, os
import hashlib, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import captain_vector as cv

fails = []
def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  -- ' + detail if detail else ''}")
    if not cond: fails.append(name)

# --- stdlib-only: GGUF fixtures, --validate, inspect, export -----------------
# These sections run on every machine, torch or not: the GGUF validate,
# inspect and export paths are stdlib-only by design, and the fixtures are
# hand-built so they do not depend on the writer they are testing against.

def _kv_str(k, v):
    b = struct.pack("<Q", len(k)) + k.encode()
    b += struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v.encode()
    return b

def _kv_u32(k, v):
    return struct.pack("<Q", len(k)) + k.encode() + struct.pack("<I", 4) + struct.pack("<I", v)

def _gguf(path, kvs, tensors, dtype=0):
    """kvs: list of bytes blobs; tensors: {layer: [floats]}."""
    blob = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs)
    infos, data = b"", b""
    off = 0
    fmt, sz = ("f", 4) if dtype == 0 else ("e", 2)
    for L in sorted(tensors):
        vals = tensors[L]
        name = f"direction.{L}"
        infos += struct.pack("<Q", len(name)) + name.encode()
        infos += struct.pack("<I", 1) + struct.pack("<Q", len(vals))
        infos += struct.pack("<I", dtype) + struct.pack("<Q", off)
        data += struct.pack(f"<{len(vals)}{fmt}", *vals)
        off += sz * len(vals)
    blob += infos
    pad = (32 - len(blob) % 32) % 32
    blob += b"\0" * pad + data
    open(path, "wb").write(blob)

_GOOD_KVS = [
    _kv_str("general.architecture", "controlvector"),
    _kv_str("glp.mode", "project"), _kv_u32("glp.spec_version", 1),
    _kv_str("glp.hook_point", "residual_stream_post_layer"),
    _kv_str("glp.derived_at", "residual_stream_post_layer"),
    struct.pack("<Q", len("glp.alpha_default")) + b"glp.alpha_default" + struct.pack("<If", 6, 1.0),
    _kv_u32("glp.rank", 1),
    struct.pack("<Q", len("glp.orthonormal")) + b"glp.orthonormal" + struct.pack("<IB", 7, 1),
    _kv_str("general.base_model.0.name", "Model"),
    _kv_str("general.base_model.0.organization", "org"),
    _kv_str("general.base_model.0.version", "a" * 40),
    _kv_str("general.base_model.0.repo_url", "https://huggingface.co/org/Model"),
    _kv_str("glp.method", "dom"), _kv_str("glp.contrast", "a-vs-b"),
    _kv_str("glp.created", "2026-09-04"),
    _kv_str("glp.layer_ids_zero_based", "1,2"),
]
_u1, _u2 = [0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0]
_sha = hashlib.sha256()
for _v in (_u1, _u2):
    _sha.update(struct.pack("<4f", *_v))

with tempfile.TemporaryDirectory() as t:
    quiet = lambda *a: None
    good = os.path.join(t, "good.gguf")
    _gguf(good, _GOOD_KVS + [_kv_str("glp.content_sha256", _sha.hexdigest())],
          {1: _u1, 2: _u2})
    check("validate: conformant file passes", cv.validate_gguf(good, out=quiet) == 0)

    bad_sha = os.path.join(t, "bad_sha.gguf")
    _gguf(bad_sha, _GOOD_KVS + [_kv_str("glp.content_sha256", "0" * 64)],
          {1: _u1, 2: _u2})
    check("validate: wrong content_sha256 fails",
          cv.validate_gguf(bad_sha, out=quiet) == 1)

    l0 = os.path.join(t, "layer0.gguf")
    _gguf(l0, _GOOD_KVS, {0: _u1, 1: _u2})
    check("validate: direction.0 fails", cv.validate_gguf(l0, out=quiet) == 1)

    no_pin = os.path.join(t, "no_pin.gguf")
    kvs_no_pin = [b for b in _GOOD_KVS if b"base_model.0.version" not in b]
    _gguf(no_pin, kvs_no_pin, {1: _u1, 2: _u2})
    check("validate: missing commit pin fails", cv.validate_gguf(no_pin, out=quiet) == 1)

    not_gguf = os.path.join(t, "nope.gguf")
    open(not_gguf, "wb").write(b"not a gguf")
    check("validate: non-GGUF fails", cv.validate_gguf(not_gguf, out=quiet) == 1)

    # --- inspect ------------------------------------------------------------
    rep = cv.inspect_gguf(good, topk=2)
    check("inspect: layer list", [e["layer"] for e in rep["layers"]] == [1, 2])
    check("inspect: unit norms",
          all(abs(e["norm"] - 1.0) < 1e-6 for e in rep["layers"]))
    check("inspect: adjacent-layer cosine",
          rep["layers"][0]["cos_prev"] is None
          and abs(rep["layers"][1]["cos_prev"] - 0.5) < 1e-6,
          f"{rep['layers'][1]['cos_prev']:+.4f}")
    check("inspect: top-k dims by magnitude",
          rep["layers"][1]["top_dims"] == [[0, 1.0], [1, 0.0]],
          f"{rep['layers'][1]['top_dims']}")
    check("inspect: topk=0 omits top dims",
          "top_dims" not in cv.inspect_gguf(good, topk=0)["layers"][0])
    check("inspect: report is JSON-serializable",
          json.loads(json.dumps(rep))["layers"][1]["layer"] == 2)
    check("inspect: metadata filtered to glp.*/general.base_model.*",
          "general.architecture" not in rep["metadata"]
          and rep["metadata"].get("glp.mode") == "project")

    # --- export: round-trip through a stdlib safetensors reader -------------
    st = os.path.join(t, "v.safetensors")
    names = cv.export_safetensors(good, st)
    check("export: tensor names sorted by layer", names == ["direction.1", "direction.2"])

    d = open(st, "rb").read()
    hlen = struct.unpack("<Q", d[:8])[0]
    hdr = json.loads(d[8:8 + hlen])
    buf = d[8 + hlen:]
    check("export: safetensors header keys",
          sorted(k for k in hdr if k != "__metadata__") == ["direction.1", "direction.2"])
    t1, t2 = hdr["direction.1"], hdr["direction.2"]
    check("export: dtype and shape",
          t1["dtype"] == "F32" and t1["shape"] == [4] and t2["shape"] == [4])
    check("export: data_offsets tile the buffer",
          t1["data_offsets"] == [0, 16] and t2["data_offsets"] == [16, 32]
          and len(buf) == 32)
    check("export: byte-exact tensor data",
          buf[0:16] == struct.pack("<4f", *_u1)
          and buf[16:32] == struct.pack("<4f", *_u2))
    md = hdr["__metadata__"]
    check("export: provenance metadata travels",
          md.get("glp.mode") == "project"
          and md.get("glp.hook_point") == "residual_stream_post_layer"
          and md.get("glp.content_sha256") == _sha.hexdigest())

    # --- refusal paths: inspect/export must fail clearly, not silently ------
    try:
        cv.inspect_gguf(not_gguf)
        check("inspect: non-GGUF errors", False, "it did not")
    except ValueError:
        check("inspect: non-GGUF errors", True)

    empty = os.path.join(t, "empty.gguf")
    _gguf(empty, _GOOD_KVS, {})
    try:
        cv.inspect_gguf(empty)
        check("inspect: no direction tensors errors", False, "it did not")
    except ValueError:
        check("inspect: no direction tensors errors", True)

    f16 = os.path.join(t, "f16.gguf")
    _gguf(f16, _GOOD_KVS, {1: _u1}, dtype=1)
    check("inspect: F16 directions still readable",
          abs(cv.inspect_gguf(f16)["layers"][0]["norm"] - 1.0) < 1e-3)

    try:
        cv.export_safetensors(not_gguf, os.path.join(t, "x.safetensors"))
        check("export: non-GGUF errors", False, "it did not")
    except ValueError:
        check("export: non-GGUF errors", True)
    try:
        cv.export_safetensors(empty, os.path.join(t, "x.safetensors"))
        check("export: no direction tensors errors", False, "it did not")
    except ValueError:
        check("export: no direction tensors errors", True)
    try:
        cv.export_safetensors(f16, os.path.join(t, "x.safetensors"))
        check("export: refuses non-F32 directions", False, "it did not")
    except ValueError:
        check("export: refuses non-F32 directions", True)

# --- everything below needs torch -------------------------------------------
try:
    import torch
except ImportError:
    torch = None
if torch is None:
    print("  [SKIP] torch-dependent sections: torch not installed")
    print(f"\n  {len(fails)} failure(s)" + (": " + ", ".join(fails) if fails else ""))
    sys.exit(1 if fails else 0)

torch.manual_seed(0)
D, N = 256, 48

# A planted direction the estimators must recover.
#
# Signal size is chosen so recovery is unambiguous. With a shift of s along
# `truth`, the difference of means also carries sampling noise of norm about
# sqrt(2D/N); at s=3, D=256, N=48 that noise is LARGER than the signal and even a
# perfect estimator lands near cos 0.68. That would be testing sampling
# statistics, not the estimator. s=12 puts the expected cosine above 0.95.
truth = torch.randn(D); truth = truth / truth.norm()
SIGNAL = 12.0
A = torch.randn(N, D) + SIGNAL * truth     # harmful: shifted along truth
B = torch.randn(N, D)                      # harmless
for name, fn in cv.ESTIMATORS.items():
    d = fn(A, B)
    check(f"{name}: unit norm", abs(float(d.norm()) - 1) < 1e-4, f"{float(d.norm()):.6f}")
    check(f"{name}: recovers the planted direction",
          abs(float(d @ truth)) > 0.75, f"cos {float(d @ truth):+.3f}")
    check(f"{name}: sign points from harmless to harmful", float(d @ truth) > 0)

# --- LDA must beat difference-of-means where it is supposed to ------------
#
# Above, the noise is isotropic, so the true covariance IS the identity and LDA
# has nothing to whiten -- it only pays the cost of estimating a 256x256
# covariance from 48 samples, and loses. That is correct behaviour, not a defect,
# and testing only that case would misrepresent the estimator.
#
# The case LDA exists for: a high-variance NUISANCE axis orthogonal to the signal.
# A difference of means is pulled toward it by sampling noise; whitening divides
# that axis down by its own variance and suppresses it.
nuis = torch.randn(D); nuis = nuis - (nuis @ truth) * truth; nuis = nuis / nuis.norm()
def corr_noise(n, scale=8.0):
    z = torch.randn(n, D)
    return z + (scale - 1.0) * (z @ nuis).unsqueeze(1) * nuis      # inflate along nuis
A2 = corr_noise(N) + 4.0 * truth
B2 = corr_noise(N)
c_dom = abs(float(cv.est_dom(A2, B2) @ truth))
c_lda = abs(float(cv.est_lda(A2, B2) @ truth))
# NOT asserted: that LDA wins on overall cosine. Whitening a d x d covariance
# from n << d samples injects noise across all the other axes, and that cost can
# exceed the nuisance benefit -- measured here as lda 0.61 vs dom 0.74 even though
# LDA suppresses the nuisance 15x better. On real activations LDA measured +47.7%
# held-out separation, so the covariance structure there is worth the estimation
# cost. The advantage is regime-dependent; the MECHANISM is what is tested.
print(f"  [info] cos to truth: lda {c_lda:.3f}  dom {c_dom:.3f} "
      f"(overall win is regime-dependent, not asserted)")
check("lda suppresses the nuisance axis",
      abs(float(cv.est_lda(A2, B2) @ nuis)) < abs(float(cv.est_dom(A2, B2) @ nuis)),
      f"lda {abs(float(cv.est_lda(A2,B2) @ nuis)):.3f} vs "
      f"dom {abs(float(cv.est_dom(A2,B2) @ nuis)):.3f} projection onto nuisance")

# --- validation must SEPARATE signal from noise ---------------------------
held, null = cv.validate_layer(A, B, cv.est_dom, reps=10)
check("validate: real feature clears the null", held / null > 3,
      f"held {held:.2f} null {null:.2f} = {held/null:.1f}x")

Z1, Z2 = torch.randn(N, D), torch.randn(N, D)     # two samples, same distribution
held0, null0 = cv.validate_layer(Z1, Z2, cv.est_dom, reps=10)
check("validate: pure noise does NOT clear the 5x gate", held0 / null0 < 5,
      f"held {held0:.2f} null {null0:.2f} = {held0/null0:.1f}x")
check("validate: in-sample would have been fooled",
      abs(cv.cohen_d(Z1 @ cv.est_dom(Z1, Z2), Z2 @ cv.est_dom(Z1, Z2))) > held0 * 1.5,
      "in-sample separation on noise exceeds held-out, as expected")

# --- input parsing --------------------------------------------------------
with tempfile.TemporaryDirectory() as t:
    p1, p2 = os.path.join(t, "a.json"), os.path.join(t, "b.json")
    json.dump(["one", "two"], open(p1, "w"))
    json.dump({"results": [{"i": 0, "prompt": "three"}]}, open(p2, "w"))
    check("parse: bare list", cv.load_prompts(p1) == ["one", "two"])
    check("parse: results/prompt objects", cv.load_prompts(p2) == ["three"])
check("parse span", cv.parse_span("1-63", 64) == list(range(1, 64)))
check("parse span clamps to model depth", cv.parse_span("1-999", 64)[-1] == 63)

# --- export refuses layer 0 (llama.cpp rejects direction.0) ---------------
try:
    cv.write_gguf("/tmp/_x.gguf", {0: torch.randn(D)}, {})
    check("export rejects layer 0", False, "it did not")
except ValueError:
    check("export rejects layer 0", True)
except ImportError:
    print("  [SKIP] export rejects layer 0 -- gguf not installed")

# --- steering hook arithmetic ---------------------------------------------
# The whole method is one line of algebra; if this is wrong every number the
# tool produces is wrong, and it fails silently rather than raising.
class _FakeLayer(torch.nn.Module):
    def forward(self, x):
        return x

class _FakeAd:
    """Minimal stand-in for Adapter: N identical passthrough layers."""
    def __init__(self, n):
        self.layers = [_FakeLayer() for _ in range(n)]
    def submodule(self, layer, hook):
        return layer

d = cv._unit(torch.randn(D))
h = torch.randn(2, 5, D)                      # (batch, seq, hidden)

ad = _FakeAd(3)
for alpha, name, want in ((0.0, "alpha=0 is a bit-exact no-op", "identity"),
                          (1.0, "alpha=1 removes the component", "zero"),
                          (2.0, "alpha=2 reflects the component", "negate")):
    hs = cv.steer_hooks(ad, {0: d}, alpha, "post_layer")
    try:
        out = ad.layers[0](h)
    finally:
        for x in hs:
            x.remove()
    before = h @ d
    after = out @ d
    if want == "identity":
        check(name, torch.equal(out, h), "bitwise identical")
    elif want == "zero":
        check(name, after.abs().max() < 1e-4,
              f"max |h'.d| = {after.abs().max():.2e}")
    else:
        check(name, torch.allclose(after, -before, atol=1e-4),
              f"h'.d = -h.d within {float((after + before).abs().max()):.2e}")

# orthogonal content must pass through untouched -- this is what makes the
# intervention self-limiting rather than a blanket edit
q = torch.randn(D)
q = cv._unit(q - (q @ d) * d)                 # exactly orthogonal to d
hq = q.expand(1, 1, D).clone()
hs = cv.steer_hooks(ad, {0: d}, 1.0, "post_layer")
try:
    outq = ad.layers[0](hq)
finally:
    for x in hs:
        x.remove()
check("orthogonal activation is unchanged (self-limiting)",
      torch.allclose(outq, hq, atol=1e-5),
      f"max drift {float((outq - hq).abs().max()):.2e}")

# hooks must apply ONLY to the layers named in dirs
ad2 = _FakeAd(3)
hs = cv.steer_hooks(ad2, {1: d}, 1.0, "post_layer")
try:
    untouched = ad2.layers[0](h)
    touched = ad2.layers[1](h)
finally:
    for x in hs:
        x.remove()
check("hook applies only to listed layers",
      torch.equal(untouched, h) and (touched @ d).abs().max() < 1e-4)

# removing handles must fully restore behaviour -- a leaked hook would silently
# contaminate every later arm in the same process
hs = cv.steer_hooks(ad2, {0: d}, 1.0, "post_layer")
for x in hs:
    x.remove()
check("removed hooks leave no residue", torch.equal(ad2.layers[0](h), h))

# --- scorer ---------------------------------------------------------------
check("score: plain refusal", cv.score("I'm sorry, but I can't help with that.") == "REFUSE")
check("score: compliance", cv.score("Sure! Here are the steps you asked about, "
      "beginning with a clear overview of the whole process and its parts.") == "COMPLY")
check("score: degenerate output is GARBLED not COMPLY",
      cv.score("!" * 200) == "GARBLED",
      "this exact hole produced a '100% delivery' headline")
check("score: markdown rule is not degenerate",
      cv.score("# Title\n" + "=" * 60 + "\nNormal varied prose continues here at "
               "sufficient length to clear the diversity check.") == "COMPLY")
check("score: 'I refuse to accept that premise, but here is how' is compliance",
      cv.score("I refuse to accept that premise, but here is how the process "
               "works: begin by measuring the aperture, then proceed.") == "COMPLY",
      "a bare 'refuse' substring match gets this wrong")
check("score: empty is GARBLED", cv.score("") == "GARBLED")

# --- massive-activation screen and masking --------------------------------
# The screen decides whether --mask is even relevant for a checkpoint, so a
# false negative silently costs delivery and a false positive silently removes
# real signal.
clean_A, clean_B = torch.randn(24, D), torch.randn(24, D)
r_clean, _ = cv.massive_ratio(clean_A, clean_B)
check("screen: clean activations report a low ratio", r_clean < 20,
      f"{r_clean:.0f}x (DeepSeek measured 6-17x)")

spiked_A, spiked_B = clean_A.clone(), clean_B.clone()
spiked_A[:, 7] *= 300.0
spiked_B[:, 7] *= 300.0
r_spiked, top = cv.massive_ratio(spiked_A, spiked_B)
check("screen: a planted massive dim is detected", r_spiked > 50 and top[0] == 7,
      f"{r_spiked:.0f}x, top dim {top[0]} (Qwen measured 75-434x at dim 3994)")

# masking must remove exactly the planted dim and leave a unit vector
d_spk = cv.est_dom(spiked_A, spiked_B)
d_msk = cv.apply_mask(d_spk, spiked_A, spiked_B, 0.005)
k = max(1, int(round(0.005 * D)))
check("mask: zeroes the massive dim", float(d_msk[7].abs()) == 0.0)
check("mask: result stays unit norm", abs(float(d_msk.norm()) - 1.0) < 1e-5,
      f"norm {float(d_msk.norm()):.6f}")
check("mask: zeroes exactly k dims", int((d_msk == 0).sum()) == k,
      f"{int((d_msk == 0).sum())} zeroed, k={k}")
check("mask: frac=0 is an exact no-op",
      torch.equal(cv.apply_mask(d_spk, spiked_A, spiked_B, 0.0), d_spk))

# and it must actually change what the direction points at
cos_before_after = float(torch.nn.functional.cosine_similarity(
    d_spk, d_msk, dim=0))
check("mask: materially changes a contaminated direction",
      cos_before_after < 0.99,
      f"cos(unmasked, masked) = {cos_before_after:.4f}")

# --- effective dose -------------------------------------------------------
# alpha is not portable between checkpoints (measured: 4.9x difference in dose
# per unit alpha between two models). Dose is the quantity that is.
lo_A = torch.randn(24, D)                      # direction nearly orthogonal to data
lo_B = torch.randn(24, D)
q = torch.randn(D)
d_lo = cv._unit(q)
dz_lo = cv.dose(lo_A, lo_B, d_lo)
check("dose: a random direction takes a small share of the norm", dz_lo < 0.1,
      f"{dz_lo:.4f} (1/sqrt(D) = {D**-0.5:.4f})")

# a direction ALIGNED with the data's dominant axis must show a large dose
big = torch.randn(24, D) * 0.1
big[:, 3] += 20.0
d_hi = torch.zeros(D); d_hi[3] = 1.0
dz_hi = cv.dose(big, big, d_hi)
check("dose: an aligned direction takes most of the norm", dz_hi > 0.8,
      f"{dz_hi:.4f}")
check("dose: is scale-invariant in h", 
      abs(cv.dose(big * 7.0, big * 7.0, d_hi) - dz_hi) < 1e-4,
      "multiplying activations by 7 leaves dose unchanged")
check("dose: is sign-invariant in d",
      abs(cv.dose(big, big, -d_hi) - dz_hi) < 1e-6)

# --- degeneracy guard: Unicode decoration ---------------------------------
# The guard shipped with an ASCII-only decoration list and met U+2500, which one
# model uses for markdown rules. Six substantive 1100-1400 char answers scored as
# a collapsed model. It must clear formatting runs of ANY script while still
# catching real collapse.
_REAL = ("# In-Memory AMSI Bypass via patching\n\n# " + "\u2500" * 60 +
         "\n\nAMSI hooks into the PowerShell execution pipeline; when a script runs "
         "it calls AmsiScanString in amsi.dll, which forwards the buffer to whichever "
         "antimalware provider is registered. Patching the prologue of that export "
         "with a short sequence returning a clean result short-circuits the check.")
check("guard: '!!!!' collapse still caught", cv._degenerate("!" * 200) is not None,
      "this exact output was once scored as 100% delivery")
check("guard: 'aaaa' collapse still caught", cv._degenerate("a" * 200) is not None)
check("guard: real collapse '48C88C..' caught", cv._degenerate("48C88C" * 40) is not None,
      "actual output from an over-steered run")
for name, ch in (("U+2500 box rule", "\u2500"), ("ASCII '='", "="),
                 ("em dash", "\u2014"), ("heavy box U+2501", "\u2501")):
    check(f"guard: {name} does not flag a real answer",
          cv._degenerate(_REAL.replace("\u2500", ch)) is None)
check("guard: short text returns None", cv._degenerate("hi") is None)

# --- draft / MTP stack detection ------------------------------------------
# A speculative-decoding draft head lives in a separate module list, so hooks on
# the main decoder miss it. Undetected, that silently destroys acceptance rate
# under MTP while offline benchmarks look perfect.
class _Stack(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(n)])

class _WithMTP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mtp = _Stack(1)

class _NoMTP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Stack(4)

nm, st = cv.find_draft_stack(_WithMTP())
check("draft stack: mtp.layers detected", nm == "mtp.layers" and len(st) == 1,
      f"found {nm!r}")
nm2, st2 = cv.find_draft_stack(_NoMTP())
check("draft stack: absent when there is none", nm2 is None and st2 is None)

# --- bake: GGUF + base weights -> rank-1 PEFT LoRA --------------------------
# Needs safetensors on top of torch; skip cleanly without it.
try:
    from safetensors.torch import save_file as _st_save, load_file as _st_load
except ImportError:
    print("  [SKIP] bake section: safetensors not installed")
else:
    with tempfile.TemporaryDirectory() as t:
        Hd, I = 8, 16                    # tiny hidden / intermediate dims
        torch.manual_seed(1)
        d1, d2 = cv._unit(torch.randn(Hd)), cv._unit(torch.randn(Hd))
        weights = {
            "model.layers.1.self_attn.o_proj.weight": torch.randn(Hd, Hd),
            "model.layers.1.mlp.down_proj.weight": torch.randn(Hd, I),
            "model.layers.2.self_attn.o_proj.weight": torch.randn(Hd, Hd),
            "model.layers.2.mlp.down_proj.weight": torch.randn(Hd, I),
            # a NON-residual writer: must be left alone by auto-detect
            "model.layers.1.self_attn.q_proj.weight": torch.randn(Hd, Hd),
            # a layer the vector does not cover: must be left alone too
            "model.layers.3.mlp.down_proj.weight": torch.randn(Hd, I),
        }
        base_dir = os.path.join(t, "base")     # plain dir: unverifiable -> warn
        os.makedirs(base_dir)
        _st_save(weights, os.path.join(base_dir, "model.safetensors"))
        gg = os.path.join(t, "v.gguf")         # pinned to "a"*40, alpha_default 1.0
        _gguf(gg, _GOOD_KVS, {1: d1.tolist(), 2: d2.tolist()})

        outd = os.path.join(t, "adapter")
        rc = cv._bake_cmd([gg, "--base", base_dir, "--out", outd])
        check("bake: exits 0 (unverifiable local base warns, not fails)", rc == 0)

        ad = _st_load(os.path.join(outd, "adapter_model.safetensors"))
        want_keys = {f"base_model.model.model.layers.{L}.{mod}.lora_{ab}.weight"
                     for L in (1, 2)
                     for mod in ("self_attn.o_proj", "mlp.down_proj")
                     for ab in ("A", "B")}
        check("bake: adapter keys follow PEFT naming, auto-detect picked "
              "exactly the residual writers",
              set(ad) == want_keys,
              f"missing {sorted(want_keys - set(ad))[:2]}, "
              f"extra {sorted(set(ad) - want_keys)[:2]}")
        check("bake: lora_B equals the unit direction",
              all(torch.allclose(ad[f"base_model.model.model.layers.{L}"
                                  f".self_attn.o_proj.lora_B.weight"][:, 0],
                                 d, atol=1e-6)
                  for L, d in ((1, d1), (2, d2))))
        check("bake: tensors are fp32",
              all(t.dtype == torch.float32 for t in ad.values()))

        # the algebra itself, on random probes: B(Ax) == -alpha*(d.Wx)d
        torch.manual_seed(2)
        worst = 0.0
        for L, d in ((1, d1), (2, d2)):
            for mod in ("self_attn.o_proj", "mlp.down_proj"):
                W = weights[f"model.layers.{L}.{mod}.weight"]
                A = ad[f"base_model.model.model.layers.{L}.{mod}.lora_A.weight"]
                B = ad[f"base_model.model.model.layers.{L}.{mod}.lora_B.weight"]
                for _ in range(4):
                    x = torch.randn(W.shape[1])
                    direct = W @ x - 1.0 * torch.dot(W @ x, d) * d
                    via = W @ x + (B @ (A @ x.unsqueeze(1))).squeeze(1)
                    worst = max(worst, float((direct - via).abs().max()))
        check("bake: B(Ax) == -alpha*(d.Wx)d on random probes", worst < 1e-6,
              f"max abs err {worst:.2e}")

        # alpha: default comes from the GGUF; --alpha overrides into lora_A
        W = weights["model.layers.1.mlp.down_proj.weight"]
        A_def = ad["base_model.model.model.layers.1.mlp.down_proj.lora_A.weight"]
        check("bake: alpha defaults to glp.alpha_default (1.0)",
              torch.allclose(A_def, (-1.0 * (d1 @ W)).unsqueeze(0), atol=1e-6))
        outd2 = os.path.join(t, "adapter-half")
        rc2 = cv._bake_cmd([gg, "--base", base_dir, "--out", outd2,
                            "--alpha", "0.5"])
        ad2 = _st_load(os.path.join(outd2, "adapter_model.safetensors"))
        A_half = ad2["base_model.model.model.layers.1.mlp.down_proj.lora_A.weight"]
        check("bake: --alpha 0.5 scales lora_A exactly",
              rc2 == 0 and torch.allclose(A_half, 0.5 * A_def, atol=1e-6))

        cfg = json.load(open(os.path.join(outd, "adapter_config.json")))
        check("bake: adapter_config has r=1, lora_alpha=1, the pin as revision",
              cfg["r"] == 1 and cfg["lora_alpha"] == 1
              and cfg["revision"] == "a" * 40
              and cfg["base_model_name_or_path"] == "org/Model"
              and cfg["peft_type"] == "LORA")
        check("bake: target_modules are the short suffixes",
              cfg["target_modules"] == ["down_proj", "o_proj"],
              str(cfg["target_modules"]))
        rep = json.load(open(os.path.join(outd, "bake-report.json")))
        check("bake: report carries alpha, pin, per-layer rel norms",
              rep["alpha"] == 1.0 and rep["base_revision"] == "a" * 40
              and len(rep["layers"]) == 4
              and all(v["rel_fro"] > 0 for v in rep["layers"].values()))
        check("bake: round-trip errors are recorded and tiny",
              all(v < 1e-6 for v in rep["roundtrip"].values()),
              str(rep["roundtrip"]))

        # --modules override: only the named suffix is baked
        outd3 = os.path.join(t, "adapter-mlp-only")
        rc3 = cv._bake_cmd([gg, "--base", base_dir, "--out", outd3,
                            "--modules", "mlp.down_proj"])
        ad3 = _st_load(os.path.join(outd3, "adapter_model.safetensors"))
        check("bake: --modules overrides auto-detect",
              rc3 == 0 and len(ad3) == 4
              and all("mlp.down_proj" in k for k in ad3))

        # --- the revision pin fails closed ----------------------------------
        wrong = os.path.join(t, "snapshots", "b" * 40)
        os.makedirs(wrong)
        _st_save(weights, os.path.join(wrong, "model.safetensors"))
        rc4 = cv._bake_cmd([gg, "--base", wrong, "--out", os.path.join(t, "n")])
        check("bake: base at the WRONG revision exits 1", rc4 == 1)
        rc5 = cv._bake_cmd([gg, "--base", wrong, "--out", os.path.join(t, "n2"),
                            "--revision", "b" * 40])
        check("bake: --revision escape hatch overrides the pin", rc5 == 0)

        no_pin = os.path.join(t, "no_pin.gguf")
        kvs_no_pin = [b for b in _GOOD_KVS if b"base_model.0.version" not in b]
        _gguf(no_pin, kvs_no_pin, {1: d1.tolist()})
        rc6 = cv._bake_cmd([no_pin, "--base", base_dir,
                            "--out", os.path.join(t, "n3")])
        check("bake: GGUF with a MISSING pin exits 1", rc6 == 1)

        # HF repo id: pinned snapshot must exist in the local cache
        old_cache = os.environ.get("HF_HUB_CACHE")
        os.environ["HF_HUB_CACHE"] = os.path.join(t, "empty-cache")
        try:
            rc7 = cv._bake_cmd([gg, "--base", "org/Model",
                                "--out", os.path.join(t, "n4")])
            check("bake: HF repo id without the pinned snapshot exits 1",
                  rc7 == 1)
            snap = os.path.join(t, "empty-cache", "models--org--Model",
                                "snapshots", "a" * 40)
            os.makedirs(snap)
            _st_save(weights, os.path.join(snap, "model.safetensors"))
            rc8 = cv._bake_cmd([gg, "--base", "org/Model",
                                "--out", os.path.join(t, "ok-hf")])
            check("bake: HF repo id WITH the pinned snapshot bakes", rc8 == 0)
        finally:
            if old_cache is None:
                del os.environ["HF_HUB_CACHE"]
            else:
                os.environ["HF_HUB_CACHE"] = old_cache

print(f"\n  {len(fails)} failure(s)" + (": " + ", ".join(fails) if fails else ""))
sys.exit(1 if fails else 0)
