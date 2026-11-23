"""Run calib.py (its asserts are the checks), draw the solution set, write generated/values.tex."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import calib

ROOT = calib.ROOT
GEN, FIG = os.path.join(ROOT, "generated"), os.path.join(ROOT, "figures")
os.makedirs(GEN, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
r = calib.run()
f = r["family"]
assert abs(r["Pmax"] - 17.5) < 1e-9 and abs(r["Pmin"] - 12.5) < 1e-9 and abs(r["muA"] - 5) < 1e-9
assert abs(r["grid_range"][0] - 12.5) < 0.05 and abs(r["grid_range"][1] - 17.5) < 0.05

m = {}
put = lambda k, v, fmt="{:.4f}": m.__setitem__(k, v if isinstance(v, str) else fmt.format(v))
put("GridDist", r["grid_maxdist"], "{:.4f}")
put("GridLo", r["grid_range"][0], "{:.2f}"); put("GridHi", r["grid_range"][1], "{:.2f}")
put("AvgArc", r["Pavg_arc"], "{:.2f}"); put("RangeOverAvg", 100 * r["muA"] / r["Pavg_arc"], "{:.0f}")
rows = []
for k, (lo, hi) in r["strikes"].items():
    rows.append(f"{k} & {lo:.2f} & {hi:.2f} & {hi - lo:.2f}\\\\")
with open(os.path.join(GEN, "values.tex"), "w") as fh:
    for k, v in m.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
with open(os.path.join(GEN, "strikes.tex"), "w") as fh:
    fh.write("\\begin{tabular}{rrrr}\\toprule $K$ & min & max & $\\mu_{\\mathcal A}$\\\\\\midrule\n")
    fh.write("\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")

# ---- figure: solution set coloured by the price of (S2 - 90)^+
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(6.4, 5.0))
pts, P = r["pts"], r["P"]
segs = np.stack([pts[:-1], pts[1:]], axis=1)
keep = np.hypot(*(pts[1:] - pts[:-1]).T) < 0.01        # drop the jumps between segments
lc = LineCollection(segs[keep], cmap="viridis", linewidths=4, capstyle="round")
lc.set_array(0.5 * (P[:-1] + P[1:])[keep]); lc.set_clim(12.5, 17.5)
ax.add_collection(lc)
cb = fig.colorbar(lc, ax=ax, pad=0.02); cb.set_label(r"price of $(S_2-90)^+$", color=INK)
for (x, y), lab, dx, dy in (((0, 1/3), r"$(0,1/3)$", 0.01, 0.025), ((1/11, 1/3), r"$(1/11,1/3)$", 0.008, 0.025),
                            ((3/11, 1/9), r"$(3/11,1/9)$", 0.012, 0.01), ((3/11, 0), r"$(3/11,0)$", 0.012, 0.012),
                            ((2/11, 2/9), r"$(2/11,2/9)$", 0.012, 0.012)):
    ax.plot(x, y, "o", color=INK, ms=4, zorder=5); ax.annotate(lab, (x, y), (x + dx, y + dy), color=INK, fontsize=10)
ax.set_xlim(-0.01, 0.36); ax.set_ylim(-0.02, 0.42); ax.set_aspect("equal")
ax.set_xlabel(r"$\sigma_1^u$", color=INK); ax.set_ylabel(r"$\sigma_1^d$", color=INK)
ax.grid(color="#e6e5e0", lw=0.7); ax.set_axisbelow(True)
ax.set_title(r"Calibrated models, $\lambda_1=.2$, $\lambda_2=.4$", loc="left", color=INK)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "solution_set.pdf")); fig.savefig(os.path.join(FIG, "solution_set.png"), dpi=170)
print(m)
