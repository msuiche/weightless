"""What held/null ratio does PURE NOISE produce? The gate must clear that.

--min-null-ratio defaults to 2.0. If noise routinely reaches 2x, the gate lets
noise through and the tool's central safety claim is false. Measured rather than
assumed, across seeds and sample sizes.
"""
import statistics, torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import captain_vector as cv

print(f"  {'n per class':>12} {'d':>6} {'median':>8} {'p90':>7} {'p99':>7} {'max':>7}  over 40 seeds")
for N, D in ((16, 256), (32, 256), (48, 256), (32, 5120), (16, 5120)):
    rs = []
    for s in range(40):
        torch.manual_seed(1000 + s)
        Z1, Z2 = torch.randn(N, D), torch.randn(N, D)     # same distribution
        h, nl = cv.validate_layer(Z1, Z2, cv.est_dom, reps=10, seed=s)
        rs.append(h / max(nl, 1e-9))
    rs.sort()
    print(f"  {N:>12} {D:>6} {statistics.median(rs):>8.2f} {rs[int(.9*len(rs))]:>7.2f} "
          f"{rs[int(.99*len(rs))]:>7.2f} {max(rs):>7.2f}")
print()
print("  A real feature in this project measured 55x. Anything a noise sample can")
print("  reach is not a useful floor.")
