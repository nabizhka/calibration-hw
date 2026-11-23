"""Two-period binomial calibration and model risk (MF731 HW 13, 13_calibration_pub)."""
import os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
L1, L2, S0, K = 0.2, 0.4, 100.0, 90.0


def leaves(s0, su, sd, S=S0):
    return np.array([S * (1 + s0) * (1 + su), S * (1 + s0) * (1 - su),
                     S * (1 - s0) * (1 + sd), S * (1 - s0) * (1 - sd)])


def price(payoff, s0, su, sd, S=S0):
    """r = 0, risk-neutral probabilities 1/2 on each branch -> 1/4 per leaf."""
    return payoff(leaves(s0, su, sd, S)).mean()


def call(k):
    return lambda s: np.maximum(s - k, 0.0)


def rhs_q2(su, sd, l1=L1):
    sp, sm = 1 + l1 / 2, 1 - l1 / 2
    return l1 / 2 + su * sp + max(l1 / 2 - su * sp, 0) + max(-l1 / 2 + sd * sm, 0)


def family(l1=L1, l2=L2, n=2001):
    """Closed-form solution set of Q4: three segments."""
    sp, sm, c = 1 + l1 / 2, 1 - l1 / 2, l1 / 2
    su_k, sd_k = c / sp, c / sm                     # kinks 1/11, 1/9
    sd_h = (l2 - c) / sm                            # horizontal piece: 1/3
    su_v = (l2 - c) / sp                            # vertical piece: 3/11
    seg1 = np.column_stack([np.linspace(0, su_k, n), np.full(n, sd_h)])
    su = np.linspace(su_k, su_v, n)
    seg2 = np.column_stack([su, (l2 - sp * su) / sm])           # sp su + sm sd = l2
    seg3 = np.column_stack([np.full(n, su_v), np.linspace(sd_k, 0, n)])
    return dict(seg=(seg1, seg2, seg3), su_k=su_k, sd_k=sd_k, sd_h=sd_h, su_v=su_v)


def run():
    s0 = L1 / 2
    out = dict(s0=s0)
    # Q1: one-period call
    out["c1"] = 0.5 * (max(S0 * (1 + s0) - S0, 0) + max(S0 * (1 - s0) - S0, 0))
    assert abs(out["c1"] - L1 * S0 / 4) < 1e-12
    # Q2: identity on a random grid
    rng = np.random.default_rng(731)
    for su, sd in rng.uniform(0, 1, size=(20_000, 2)):
        assert abs(price(call(S0), s0, su, sd) / (S0 / 4) - rhs_q2(su, sd)) < 1e-12
    # Q4: closed form lies on the calibration constraint ...
    f = family()
    pts = np.vstack(f["seg"])
    for su, sd in pts:
        assert abs(price(call(S0), s0, su, sd) - L2 * S0 / 4) < 1e-9
    # ... and a brute-force grid finds nothing off it
    g = np.linspace(0, 1, 1601)
    SU, SD = np.meshgrid(g, g)
    lv = [S0 * (1 + s0) * (1 + SU), S0 * (1 + s0) * (1 - SU), S0 * (1 - s0) * (1 + SD), S0 * (1 - s0) * (1 - SD)]
    C2 = sum(np.maximum(v - S0, 0) for v in lv) / 4
    near = np.abs(C2 - L2 * S0 / 4) < 0.02
    d = np.array([np.min(np.hypot(pts[:, 0] - a, pts[:, 1] - b)) for a, b in zip(SU[near], SD[near])])
    out["grid_maxdist"] = d.max()
    assert d.max() < 5e-3
    # Q3 on the family
    sp, sm = 1 + L1 / 2, 1 - L1 / 2
    out["q3_min"] = min(2 * max(a * sp, b * sm) for a, b in pts)
    assert out["q3_min"] > L1
    # Q5: price of (S2 - 90)^+ over the family
    P = np.array([price(call(K), s0, a, b) for a, b in pts])
    out["P"], out["pts"] = P, pts
    out["Pmax"], out["Pmin"] = P.max(), P.min()
    out["argmax"], out["argmin"] = pts[P.argmax()], pts[P.argmin()]
    out["muA"] = P.max() - P.min()
    # average over models parametrised by arc length (one reasonable "average model")
    seglen = [np.sum(np.hypot(*np.diff(s, axis=0).T)) for s in f["seg"]]
    out["seglen"] = seglen
    Pseg = [np.array([price(call(K), s0, a, b) for a, b in s]) for s in f["seg"]]
    out["Pavg_arc"] = sum(L * p.mean() for L, p in zip(seglen, Pseg)) / sum(seglen)
    # brute-force range on the grid points that calibrate (tolerance band)
    lvK = sum(np.maximum(v - K, 0) for v in lv) / 4
    out["grid_range"] = (lvK[near].min(), lvK[near].max())
    # other strikes for the table
    out["strikes"] = {}
    for k in (80, 90, 100, 110, 120):
        Pk = np.array([price(call(k), s0, a, b) for a, b in pts])
        out["strikes"][k] = (Pk.min(), Pk.max())
    out["family"] = f
    return out


if __name__ == "__main__":
    r = run()
    f = r["family"]
    print("s0", r["s0"], "kinks", f["su_k"], f["sd_k"], "sd_h", f["sd_h"], "su_v", f["su_v"])
    print("grid maxdist", r["grid_maxdist"], "q3 min", r["q3_min"])
    print("Pmax", r["Pmax"], r["argmax"], "Pmin", r["Pmin"], r["argmin"], "muA", r["muA"])
    print("grid range", r["grid_range"], "arc avg", r["Pavg_arc"], "seglen", r["seglen"])
    for k, v in r["strikes"].items(): print(k, v)
