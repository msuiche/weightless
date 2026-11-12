#!/usr/bin/env python3
"""weightless lane — deploy a serving recipe with a GLP vector.

    python3 scripts/lane.py list
    python3 scripts/lane.py up <name> [--ref R] [--vector PATH] [--alpha A] [--dry-run]
    python3 scripts/lane.py check <name>

Reads lanes/<name>/weightless.json (schema weightless/lane/1, docs/LANE.md),
clones the recipe at its pin, resolves and verifies the vector (sha256,
fail-closed), places it per the lane's glp.placement, runs the recipe's own
check and up commands, and reports whether the boot line shows steering
armed or stock. Stdlib only.
"""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LANES_DIR = os.path.join(ROOT, "lanes")
WORK = os.path.join(os.path.expanduser("~"), ".cache", "weightless-lanes")

SCHEMA = "weightless/lane/1"


def load_lane(name):
    path = os.path.join(LANES_DIR, name, "weightless.json")
    if not os.path.isfile(path):
        raise SystemExit(f"lane {name!r}: no {path}")
    lane = json.load(open(path))
    if lane.get("schema") != SCHEMA:
        raise SystemExit(f"lane {name!r}: schema {lane.get('schema')!r} "
                         f"!= {SCHEMA!r} (refusing)")
    lane["_dir"] = os.path.join(LANES_DIR, name)
    return lane


def list_lanes():
    names = sorted(d for d in os.listdir(LANES_DIR)
                   if os.path.isfile(os.path.join(LANES_DIR, d, "weightless.json")))
    rows = []
    for n in names:
        lane = load_lane(n)
        rows.append((n, lane.get("pin", {}).get("ref", "?"),
                     lane.get("glp", {}).get("placement", "?"),
                     lane.get("last_verified", "?"),
                     lane.get("homepage", "")))
    for n, ref, place, verified, home in rows:
        print(f"  {n:<22} pin={ref:<10} {place:<7} verified={verified}  {home}")
    return 0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(argv, cwd=None, env=None, check=True, capture=True):
    print("  $ " + shlex.join(argv))
    r = subprocess.run(argv, cwd=cwd, env=env,
                       capture_output=capture, text=True)
    if capture and r.stdout:
        sys.stdout.write(r.stdout[-4000:] if len(r.stdout) > 4000 else r.stdout)
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {shlex.join(argv)}\n"
                         f"{(r.stderr or '')[-800:]}")
    return r


def clone_at_pin(lane, ref, dry):
    """Resolve the lane's working checkout. A lane with a `checkout` field
    uses that existing deployment directory when present (fetch + checkout
    the pin) instead of cloning fresh: recipes like spark-ds41f deploy from
    a canonical on-node path, and a side cache clone is not the deployment."""
    home = lane["homepage"]
    dest = os.path.expanduser(lane.get("checkout") or os.path.join(WORK, lane["name"]))
    if os.path.isdir(os.path.join(dest, ".git")):
        if dry:
            print(f"  $ git -C {dest} fetch origin {ref} && git checkout <pin> (existing checkout)")
            return dest
        run(["git", "-C", dest, "fetch", "--depth", "200", "origin", ref])
        run(["git", "-C", dest, "checkout", "--detach", "FETCH_HEAD"])
        rev = run(["git", "-C", dest, "rev-parse", "HEAD"]).stdout.strip()
        print(f"  checkout {dest} pinned: {rev[:12]}")
        return dest
    dest = os.path.join(WORK, lane["name"])
    if dry:
        print(f"  $ git clone --depth 50 {home} {dest} (or fetch + checkout {ref})")
        return dest
    os.makedirs(WORK, exist_ok=True)
    run(["git", "clone", "--depth", "50", "--branch", ref, home, dest])
    rev = run(["git", "-C", dest, "rev-parse", "HEAD"]).stdout.strip()
    print(f"  pinned: {rev[:12]}")
    return dest


def lane_config_vector(repo_dir, lane):
    """(repo, revision, file, sha256) from the recipe's own config, or None."""
    cfg = lane.get("glp", {}).get("config", {})
    path = os.path.join(repo_dir, cfg.get("path", ""))
    key = cfg.get("key")
    fields = cfg.get("fields", {})
    if not os.path.isfile(path) or not key:
        return None
    try:
        block = json.load(open(path)).get(key, {})
    except json.JSONDecodeError:
        return None
    want = ("repo", "revision", "file", "sha256")
    if all(block.get(fields.get(w, w)) for w in want):
        return tuple(block[fields.get(w, w)] for w in want)
    return None


def fetch_vector(repo, revision, fname, sha, work):
    dest = os.path.join(work, "vectors", fname)
    if os.path.isfile(dest) and sha256_file(dest) == sha:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    cli = shutil_which("hf")
    if cli:
        run([cli, "download", repo, fname, "--revision", revision,
             "--local-dir", os.path.dirname(dest)])
    else:
        raise SystemExit(f"vector missing at {dest} and no `hf` CLI to fetch it. "
                         f"Get it manually: hf download {repo} {fname} --revision {revision}")
    got = sha256_file(dest)
    if got != sha:
        raise SystemExit(f"vector sha256 mismatch: file {got[:16]}… != pinned {sha[:16]}… "
                         f"(refusing)")
    return dest


def shutil_which(prog):
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = os.path.join(d, prog)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def verify_vector(path, expected_sha):
    got = sha256_file(path)
    if expected_sha and got != expected_sha:
        raise SystemExit(f"vector sha256 mismatch: {got[:16]}… != {expected_sha[:16]}… (refusing)")
    print(f"  vector sha256 ok ({got[:16]}…)")
    return got


def render_config(repo_dir, lane, vector_path, alpha, dry):
    """Verify (and when asked, update) the recipe's GLP config block. The
    served_model_name field is preserved; only repo/revision/file/sha256
    move."""
    cfg = lane["glp"]["config"]
    path = os.path.join(repo_dir, cfg["path"])
    key = cfg["key"]
    fields = cfg["fields"]
    if dry:
        print(f"  $ verify {cfg['path']}#{key} has fields {sorted(fields.values())}"
              + ("; render file+sha from the vector" if vector_path else ""))
        return
    if not os.path.isfile(path):
        raise SystemExit(f"lane config missing: {path}")
    doc = json.load(open(path))
    if key not in doc:
        raise SystemExit(f"{path} has no {key!r} block (refusing)")
    if vector_path is None and alpha is None:
        cur = lane_config_vector(repo_dir, lane)
        if cur is None:
            raise SystemExit(f"{path}: glp block incomplete (need repo/revision/file/sha256)")
        print(f"  config glp: {cur[2]} @ {cur[1][:12]} (recipe's own pin)")
        return
    if vector_path is None:
        raise SystemExit("--alpha without --vector is refused: a new alpha belongs "
                         "to a new vector (alpha lives in the file's alpha_default)")
    sha = sha256_file(vector_path)
    print("  NOTE: config render records the vector path + sha; repo/revision "
          "must be supplied by the vector's publisher fields, kept as-is")
    if not dry:
        block = doc[key]
        block[fields.get("file", "file")] = os.path.basename(vector_path)
        block[fields.get("sha256", "sha256")] = sha
        json.dump(doc, open(path, "w"), indent=2)
        open(path, "a").write("\n")
        print(f"  rendered {cfg['path']}#{key}: file={os.path.basename(vector_path)} sha={sha[:12]}…")


def env_for(lane, vector_path, alpha):
    e = lane.get("glp", {}).get("env", {})
    out = {}
    if e.get("path"):
        out[e["path"]] = vector_path or ""
    if alpha is not None and e.get("alpha"):
        out[e["alpha"]] = str(alpha)
    for k, v in (e.get("extra") or {}).items():
        out[k] = v
    return out


def cmd_up(name, ref, vector_path, alpha, dry):
    lane = load_lane(name)
    ref = ref or lane.get("pin", {}).get("ref", "main")
    print(f"lane up: {name} @ {ref}" + (" [dry-run]" if dry else ""))
    repo_dir = clone_at_pin(lane, ref, dry)
    place = lane.get("glp", {}).get("placement")
    env = os.environ.copy()
    if place == "config":
        if vector_path:
            verify_vector(vector_path, None)
        render_config(repo_dir, lane, vector_path, alpha, dry)
    elif place == "env":
        if vector_path:
            verify_vector(vector_path, None)
        env.update(env_for(lane, vector_path, alpha))
        print(f"  env: {sorted(k for k in env_for(lane, vector_path, alpha))}")
    elif place == "self":
        prep = lane["glp"]["self"].get("prepare")
        if prep:
            vdir = os.path.join(repo_dir, ".weightless")
            argv = [a.replace("{vector_dir}", vdir) for a in prep]
            if not dry:
                os.makedirs(vdir, exist_ok=True)
            run(argv, cwd=repo_dir, env=env, check=True)
        else:
            cur = lane_config_vector(repo_dir, lane)
            if cur:
                fetch_vector(cur[0], cur[1], cur[2], cur[3], repo_dir) if not dry else None
    else:
        raise SystemExit(f"lane {name}: unknown placement {place!r}")
    cmds = lane.get("commands", {})
    if cmds.get("check"):
        if dry:
            print("  $ " + shlex.join(cmds["check"]))
        else:
            run(cmds["check"], cwd=repo_dir, env=env, check=True)
    if not cmds.get("up"):
        raise SystemExit(f"lane {name}: no commands.up declared")
    if dry:
        print("  $ " + shlex.join(cmds["up"]))
        print("dry-run: nothing executed")
        return 0
    r = run(cmds["up"], cwd=repo_dir, env=env, check=True)
    boot = lane.get("verify", {}).get("boot_line")
    if boot:
        text = (r.stdout or "") + (r.stderr or "")
        print(f"  verify: boot line {boot!r} "
              + ("FOUND — steering armed" if boot in text else "ABSENT — the boot is stock"))
    return 0


def cmd_check(name):
    lane = load_lane(name)
    dest = os.path.join(WORK, name)
    if not os.path.isdir(dest):
        raise SystemExit(f"lane {name}: not cloned yet (run `lane up --dry-run` first "
                         f"or clone into {WORK})")
    cmds = lane.get("commands", {})
    if cmds.get("check"):
        run(cmds["check"], cwd=dest, check=True)
    else:
        print(f"lane {name}: no check command")
    return 0


def main():
    p = argparse.ArgumentParser(prog="weightless lane",
                                description=__doc__.strip().split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    up = sub.add_parser("up")
    up.add_argument("name")
    up.add_argument("--ref", default="")
    up.add_argument("--vector", default="")
    up.add_argument("--alpha", type=float, default=None)
    up.add_argument("--dry-run", action="store_true")
    ck = sub.add_parser("check")
    ck.add_argument("name")
    a = p.parse_args()
    if a.cmd == "list":
        return list_lanes()
    if a.cmd == "up":
        return cmd_up(a.name, a.ref, a.vector or None, a.alpha, a.dry_run)
    if a.cmd == "check":
        return cmd_check(a.name)


if __name__ == "__main__":
    sys.exit(main())
