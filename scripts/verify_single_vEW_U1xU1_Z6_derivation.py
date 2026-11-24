#!/usr/bin/env python3
"""5-Second Decisive Verification of the Exact Derived U(1)_A x U(1)_B x Z_6
Single-Scale (v_EW = 100 GeV, lambda = 1/4) 4-Simplex Spin(10) Model!

ALL 12 integer exponents derived strictly from:
  - Generation charges: a = (-4, -1, 0),  b = (-1, 0, 0)
  - Higgs charges:      Q_{Hu} = (0, 0),  Q_{Hd} = (-1, +2),  Q_F = (-4, 0)
  - Exact rule:         n^(H)_ij = |a_i + a_j + A_H| + |b_i + b_j + B_H|
  - Exact base:         lambda = 1/4 (common eigenvalue of 4-simplex K_5 & 24-cell P_24!)
  - ONE SINGLE SCALE:   v_EW = 100 GeV (NO separate v_d vs v_u hierarchy input — m_b/m_t ~ (1/4)^3 = 1/64 is DERIVED!)

Predicted exact exponents relative to v_EW = 100 GeV:
  H_u: u33=0, u23=1, u13=5, u12=6, u11=10   (u22=0 by Z_6 subset 2T!)
  H_d: h33=3, h23=4, h12=7, h11=9           (h13=h22=0 by Z_6 subset 2T!)
  F:   f3=4,  f2=6,  f1=14                  (F_12=F_13=0 by Z_2 subset Z_6 subset 2T!)

Checks whether ALL 18 observables fit to chi^2 < 3 with ALL dimensionless O(1)
coefficients strictly bounded in [0.25, 4.0] (and tests stricter [0.33, 3.0]!).
"""

import json
import math
import numpy as np
from scipy.optimize import least_squares

from verify_3gen_spin10_2T_global_fit import TARGETS
from verify_predictive_2T_reduction_models import evaluate_model_D

OBS_KEYS = list(TARGETS.keys())


def verify_single_vEW_derived_exponents(c_max: float = 3.0):
    lam = 0.25
    vEW = 100.0  # ONE SINGLE ELECTROWEAK SCALE (GeV)!
    a = (-4, -1, 0)
    b = (-1, 0, 0)
    Qu, Qd, Qf = (0, 0), (-1, 2), (-4, 0)

    def n_exp(i, j, Q):
        return abs(a[i] + a[j] + Q[0]) + abs(b[i] + b[j] + Q[1])

    exps = dict(
        u33=n_exp(2, 2, Qu), u23=n_exp(1, 2, Qu), u13=n_exp(0, 2, Qu), u12=n_exp(0, 1, Qu), u11=n_exp(0, 0, Qu),
        h33=n_exp(2, 2, Qd), h23=n_exp(1, 2, Qd), h12=n_exp(0, 1, Qd), h11=n_exp(0, 0, Qd),
        f3=n_exp(2, 2, Qf),  f2=n_exp(1, 1, Qf),  f1=n_exp(0, 0, Qf),
    )
    assert exps == dict(u33=0, u23=1, u13=5, u12=6, u11=10, h33=3, h23=4, h12=7, h11=9, f3=4, f2=6, f1=14)

    # 13 dimensionless O(1) coeffs (ALL relative to ONE SINGLE v_EW = 100 GeV!):
    #   [cf1, cf2, cf3, ch11, ch12, ch23, ch33, cu11, cu12, cu13, cu23, cu33, s_norm, p1, p3]
    # Note: in M_u = H_u + s F, since H_u ~ v_EW * (1/4)^0 and F ~ v_EW * (1/4)^4,
    # natural s_eff has charge Q_{Hd} - Q_{Hu} -> (1/4)^{-3} = 64! So s_eff = s_norm * 4^3!
    o1_idx = list(range(13))
    c_min = 1.0 / c_max
    lb = [-c_max] * 13 + [-2 * math.pi, -2 * math.pi]
    ub = [c_max] * 13 + [4 * math.pi, 4 * math.pi]

    def build(x):
        cf1, cf2, cf3, ch11, ch12, ch23, ch33, cu11, cu12, cu13, cu23, cu33, s_norm, p1, p3 = x
        f1 = vEW * cf1 * (lam ** exps["f1"])
        f2 = vEW * cf2 * (lam ** exps["f2"])
        f3 = vEW * cf3 * (lam ** exps["f3"])
        h11 = vEW * ch11 * (lam ** exps["h11"])
        h12 = vEW * ch12 * (lam ** exps["h12"])
        h23 = vEW * ch23 * (lam ** exps["h23"])
        h33 = vEW * ch33 * (lam ** exps["h33"])
        u11 = vEW * cu11 * (lam ** exps["u11"])
        u12 = vEW * cu12 * (lam ** exps["u12"])
        u13 = vEW * cu13 * (lam ** exps["u13"])
        u23 = vEW * cu23 * (lam ** exps["u23"])
        u33 = vEW * cu33 * (lam ** exps["u33"])
        s_eff = s_norm * (4.0 ** 3)  # exact (1/4)^{-(Q_Hd)} = 64!
        F = np.diag([f1, f2, f3]).astype(complex)
        Hd = np.array([
            [h11, h12 * np.exp(1j * p1), 0.0],
            [h12 * np.exp(1j * p1), 0.0, h23 * np.exp(1j * p3)],
            [0.0, h23 * np.exp(1j * p3), h33],
        ], dtype=complex)
        Hu = np.array([
            [u11, u12, u13],
            [u12, 0.0, u23],
            [u13, u23, u33],
        ], dtype=complex)
        return Hd, Hu, F, complex(s_eff, 0.0)

    def resid(x):
        Hd, Hu, F, s_c = build(x)
        ev = evaluate_model_D(Hd, Hu, F, s_c)
        if ev is None:
            return np.full(18 + len(o1_idx), 1e3)
        r_obs = np.array([ev["pulls"][k] for k in OBS_KEYS], dtype=float)
        r_nat = np.array([3.0 * max(0.0, math.log(c_min / max(1e-12, abs(x[i])))) for i in o1_idx])
        return np.concatenate([r_obs, r_nat])

    # Initial guess mapped directly from M16-U11:
    m16_raw = dict(
        f1=10**(-7.4485486), f2=0.0255106, f3=-0.1688889,
        h11=0.0002634, h12=-0.0048183, h23=-0.2140703, h33=1.1822015,
        u11=0.0014171, u12=-0.0071763, u13=-0.0884768, u23=17.7313749, u33=105.0740843,
        p1=2.9703110, p3=3.3419018, s=172.8337099,
    )
    keys_order = ["f1", "f2", "f3", "h11", "h12", "h23", "h33", "u11", "u12", "u13", "u23", "u33"]
    x_init = np.array(
        [np.clip(m16_raw[k] / (vEW * lam ** exps[k]), -c_max, c_max) for k in keys_order]
        + [np.clip(m16_raw["s"] / 64.0, -c_max, c_max), m16_raw["p1"], m16_raw["p3"]]
    )

    best = None
    rng = np.random.default_rng(20261124 + int(c_max * 10))
    for t in range(40):
        x0 = x_init.copy() if t == 0 else np.clip(x_init + rng.normal(0.0, 0.45, len(x_init)), lb, ub)
        res = least_squares(resid, x0, bounds=(lb, ub), method="trf", x_scale="jac", ftol=1e-13, xtol=1e-13, max_nfev=3500)
        Hd, Hu, F, s_c = build(res.x)
        ev = evaluate_model_D(Hd, Hu, F, s_c)
        if ev is None:
            continue
        names = [f"c_{k}" for k in keys_order] + ["c_s"]
        coeffs = {n: round(float(res.x[i]), 3) for i, n in enumerate(names)}
        mags = [abs(v) for v in coeffs.values()]
        score = ev["chi2"] + 40.0 * sum(max(0.0, c_min - m) ** 2 for m in mags)
        if best is None or score < best[0]:
            ev["v_EW_single_scale_GeV"] = vEW
            ev["lambda"] = lam
            ev["derived_exponents"] = exps
            ev["O1_coeffs"] = coeffs
            ev["min_abs_coeff"] = round(min(mags), 3)
            ev["max_abs_coeff"] = round(max(mags), 3)
            ev["all_strictly_natural"] = bool(min(mags) >= c_min - 1e-3 and max(mags) <= c_max + 1e-3)
            best = (score, ev)
    return best[1]


def main():
    r3 = verify_single_vEW_derived_exponents(c_max=3.0)
    r4 = verify_single_vEW_derived_exponents(c_max=4.0)
    print(json.dumps(dict(
        strict_bound_033_to_3=dict(
            chi2=r3["chi2"], max_pull=r3["max_abs_pull"],
            min_c=r3["min_abs_coeff"], max_c=r3["max_abs_coeff"],
            all_strictly_natural=r3["all_strictly_natural"],
            derived_exponents=r3["derived_exponents"],
            O1_coeffs=r3["O1_coeffs"],
            preds=r3["preds"],
        ),
        bound_025_to_4=dict(
            chi2=r4["chi2"], max_pull=r4["max_abs_pull"],
            min_c=r4["min_abs_coeff"], max_c=r4["max_abs_coeff"],
            all_strictly_natural=r4["all_strictly_natural"],
            derived_exponents=r4["derived_exponents"],
            O1_coeffs=r4["O1_coeffs"],
            preds=r4["preds"],
        ),
    ), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
