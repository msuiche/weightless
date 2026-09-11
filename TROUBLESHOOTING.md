# Troubleshooting

Generic failures across lanes and nodes. Lane-specific gotchas stay in each
recipe's README. First stop for any "the model is down": the wizard's
diagnose (`python3 weightless.py` → 5), which walks DNS → TCP → HTTP →
generation, then checks the node over ssh (containers, restart counts,
fabric, peers, GPU) and interprets what it finds.

## The model is down / endpoint dead

Symptom → likely cause, in the order to check:

1. **Router answers but serves zero models** — the serving engine behind it
   is down, the router is fine. Diagnose falls through to the node check
   automatically.
2. **Container restart count in the hundreds+** — crash loop. Classic cause:
   vLLM exits fast when its fabric peer is gone and the restart policy
   re-runs the whole entrypoint every few seconds (2026-09-10: 2,825
   restarts in 9h). The peer gate in `docker-compose.dsv4.yml` waits for
   the peer instead — redeploy the lane if the deployed copy predates it.
3. **Node has no 192.168.100.x fabric address / peers ping DOWN** — the far
   Spark is off or wedged (NO-CARRIER class). Physical check: power LED,
   power brick, QSFP cable. No restart policy fixes a dead peer.

## Spark wedges (silent freeze)

The known GB10 failure class: the journal stops mid-entry with no panic, no
OOM, no GPU error. The OS hangs so hard the NICs die, which is why the peer
sees NO-CARRIER and mDNS/ssh go silent at the same moment.

Self-healing armed on every node the wizard deploys to (idempotent, runs as
a deploy step; `harden_steps` in setup.py):

- **Hardware watchdog** — `/etc/systemd/system.conf.d/watchdog.conf` sets
  `RuntimeWatchdogSec=60`; systemd pings the SBSA watchdog at
  `/dev/watchdog0`, and a full OS freeze hard-resets the board within 60s.
- **Panic-on-hang sysctls** — `/etc/sysctl.d/99-wedge-heal.conf`:
  `softlockup_panic=1`, `hung_task_panic=1`, `panic=30`. Kernel lockups and
  hung tasks (the GPU-driver D-state class) panic and reboot instead of
  sitting frozen.
- **Peer gate** — the surviving node's boot waits for the peer and proceeds
  by itself when it returns.
- **Crash dumps** — `nvidia-kdump-config enable-vmcore-dump`: real
  `crashkernel=` reservation (DGX OS ships `1G-:0M`, i.e. zero), USE_KDUMP=1,
  and a wider panic net (hardlockup, oops, RCU stall, NMI panics) on top of
  the wedge-heal sysctls. A hang that panics now leaves a vmcore in
  `/var/crash` instead of just rebooting blind. Active only after the
  node's first reboot following the hardening run.

Net effect: a wedge becomes a ~2-minute blip instead of a human power-cycle,
and a panicking wedge leaves a post-mortem. First live validation: 2026-09-11,
spark-5bc3 wedged at 06:35 and was reset by the watchdog, back at 06:36:53 —
112s, no human involvement.

Known gaps: the watchdog has not been fired deliberately (needs a reboot
window). A hard wedge that never panics still leaves no dump — the watchdog
resets the board without running the crash kernel; `efi_pstore` is the only
channel there, check `/sys/fs/pstore` after the next event. Check NVIDIA
for DGX OS updates; the wedge class is theirs to fix.

## Boot fails after the assets synced

- **`.local` hostnames do not resolve inside host-networked containers** —
  fabric addresses in the env must be literal IPs (`192.168.100.x`). The
  wizard hard-errors on `.local` fabric addresses.
- **Steering hook mismatch** — the vector's `glp.hook_point` must match the
  lane's enforced site. The wizard reads it on the node pre-deploy; do not
  bypass by editing the env.
- **Stale container from a previous lane** — park it first (the wizard's
  `park_other_lanes` does this; by hand: `docker rm -f <old>`).

## Slow serving

Run the dashboard (`python3 weightless.py dash`). Read KV pressure and queue
depth first. On the big MoE lanes, 17–30 tok/s single-stream is the design
point, not a regression. Page-cache pressure stalls weight loads on boot —
the deploy chain drops caches on all nodes before booting.
