#!/usr/bin/env python3
"""Quick 5-second exact forensic extraction for:
  1. Full 18-pull breakdown of SUSY PH16_a13_h23 & NS PH17_a13 (+ check NS PH16_a13_h23),
  2. Exact Dimopoulos-Wilczek unification sweet spot (alpha_s(M_Z) == 0.1180 exact!),
     giving M'_exact, triplet suppression factor, required Mini-Split M_sq^min(SK/HK),
     Barbieri-Giudice Delta_FT, and 1-loop MSSM Higgs mass m_h(M_sq)!
"""

import json
import math
import numpy as np

from verify_3gen_spin10_2T_global_fit import TARGETS as NS_TARGETS
from verify_SUSY_and_NS_PH17_basin_and_DW_pdecay import seeded_refine
from verify_SUSY_10_126_120_parity_and_proton_decay import strip_internal


def solve_exact_DW_sweet_spot(tau_2TeV_MU_yr: float, M_SUSY: float = 2000.0):
    M_Z = 91.1876
    a1_Z, a2_Z, a3_Z = 59.01, 29.58, 1.0 / 0.1180
    b_sm = (41.0 / 10.0, -19.0 / 6.0, -7.0)
    b_mssm = (33.0 / 5.0, 1.0, -3.0)
    b_H = (3.0 / 5.0, 1.0, 0.0)

    t_s = math.log(M_SUSY / M_Z) / (2.0 * math.pi)
    a1_s = a1_Z - b_sm[0] * t_s
    a2_s = a2_Z - b_sm[1] * t_s
    a3_s = a3_Z - b_sm[2] * t_s

    def eval_Mprime(M_p):
        t_p = math.log(M_p / M_SUSY) / (2.0 * math.pi)
        a1_p = a1_s - b_mssm[0] * t_p
        a2_p = a2_s - b_mssm[1] * t_p
        a3_p = a3_s - b_mssm[2] * t_p
        b_hi = (b_mssm[0] + b_H[0], b_mssm[1] + b_H[1], b_mssm[2] + b_H[2])
        t_hi = (a1_p - a2_p) / (b_hi[0] - b_hi[1])
        MU = M_p * math.exp(2.0 * math.pi * t_hi)
        aU = a1_p - b_hi[0] * t_hi
        a3_miss = (a3_p - b_hi[2] * t_hi) - aU
        return MU, aU, a3_miss

    # Binary search M_p for exact alpha_3^-1(M_U) == alpha_U^-1 (a3_miss == 0):
    lo, hi = 1e11, 1e16
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        _, _, miss = eval_Mprime(mid)
        if miss > 0:
            lo = mid
        else:
            hi = mid
    M_p_exact = math.sqrt(lo * hi)
    MU_exact, aU_exact, miss_exact = eval_Mprime(M_p_exact)

    # DW triplet suppression: M_T^eff = M_U^2 / M' -> enhancement of tau_p is (M_U / M')^2!
    dw_cancel_factor = MU_exact / M_p_exact
    tau_2TeV_DW = tau_2TeV_MU_yr * (dw_cancel_factor ** 2)

    # Now compute required M_sq (holding M_wino = 500 GeV) to clear Super-K (5.9e33) & Hyper-K (3.2e34):
    def f_loop(Msq, Mw=500.0):
        x = (Mw / Msq) ** 2
        I_l = 0.5 if abs(x - 1.0) < 1e-6 else (1.0 / (1.0 - x) + x * math.log(x) / ((1.0 - x) ** 2))
        return (Mw / (Msq ** 2)) * I_l

    f_ref = f_loop(2000.0, 500.0)
    def find_Msq(tau_target, tau_base):
        if tau_base >= tau_target:
            return 2000.0
        l2, h2 = 2000.0, 1e7
        for _ in range(80):
            m = math.sqrt(l2 * h2)
            t = tau_base * ((f_ref / f_loop(m, 500.0)) ** 2)
            if t < tau_target:
                l2 = m
            else:
                h2 = m
        return h2

    Msq_SK_DW = find_Msq(5.9e33, tau_2TeV_DW)
    Msq_HK_DW = find_Msq(3.2e34, tau_2TeV_DW)

    # Standard 1-loop MSSM Higgs mass at tan(beta)=50, X_t=0 (zero stop mixing):
    # m_h^2 = M_Z^2 + (3 m_t^4 / (2 pi^2 v^2)) ln(M_sq^2 / m_t^2)  (v = 174.104 GeV, m_t = 163.5 GeV MSbar)
    m_t_ms = 163.5
    v = 174.104
    def mh_1loop_zero_mixing(Msq):
        dm2 = (3.0 * (m_t_ms ** 4) / (2.0 * math.pi**2 * v**2)) * math.log((Msq / m_t_ms) ** 2)
        return math.sqrt(M_Z**2 + dm2)

    yt = 0.95
    m_h = 125.25
    def delta_ft(Msq):
        return (6.0 * yt**2 / (4.0 * math.pi**2)) * ((Msq / m_h) ** 2) * math.log(MU_exact / Msq)

    return dict(
        M_prime_exact_unif_GeV=M_p_exact,
        M_U_exact_unif_GeV=MU_exact,
        alpha_U_inv=round(aU_exact, 3),
        alpha3_miss_at_sweet_spot=round(miss_exact, 6),
        DW_triplet_suppression_factor=round(dw_cancel_factor, 2),
        tau_p_at_2TeV_with_DW_yr=tau_2TeV_DW,
        Msq_min_SuperK_with_DW_GeV=round(Msq_SK_DW, 1),
        Msq_min_HyperK_with_DW_GeV=round(Msq_HK_DW, 1),
        mh_1loop_zero_stop_mixing_at_Msq_SK_GeV=round(mh_1loop_zero_mixing(Msq_SK_DW), 2),
        mh_1loop_zero_stop_mixing_at_Msq_HK_GeV=round(mh_1loop_zero_mixing(Msq_HK_DW), 2),
        Delta_FT_at_Msq_SK_with_DW=round(delta_ft(Msq_SK_DW), 1),
        Delta_FT_at_Msq_HK_with_DW=round(delta_ft(Msq_HK_DW), 1),
    )


def main():
    with open("/Users/jolcsak/.gemini/antigravity/brain/a427c21d-9cd4-494e-9b77-f0a1253ad296/scratch/susy_and_ns_ph17_basin.json") as f:
        b = json.load(f)

    s1_16 = b["S1_PH16_from_PH17_a13"]["a13_h23"]
    ns_17 = b["NS_PH17_and_PH16"]["PH17_a13"]

    # Quick check of NS PH16_a13_h23 (seeded from NS PH17_a13):
    ns_16_h23 = seeded_refine(NS_TARGETS, None, ns_17["params"], zero_mask=("a13", "h23"), n_perturb=12)

    dw_sweet = solve_exact_DW_sweet_spot(s1_16["dim5_proton_decay"]["tau_p_at_Msq2TeV_Mw500GeV_MT_MU_yr"])

    out = dict(
        SUSY_PH16_a13_h23=dict(
            n_params=s1_16["fit"]["n_params"],
            dof=s1_16["fit"]["dof"],
            chi2=round(s1_16["fit"]["chi2"], 4),
            max_abs_pull=round(s1_16["fit"]["max_abs_pull"], 3),
            pulls={k: round(v, 3) for k, v in s1_16["fit"]["pulls"].items()},
            yukawa_perturbativity=s1_16["fit"]["yukawa_perturbativity"],
            preds=s1_16["fit"]["preds"],
            params=s1_16["fit"]["params"],
            dim5_proton_decay_uncancelled=s1_16["dim5_proton_decay"],
            DW_exact_unification_sweet_spot=dw_sweet,
        ),
        NS_PH17_a13=dict(
            n_params=ns_17["n_params"],
            dof=ns_17["dof"],
            chi2=round(ns_17["chi2"], 4),
            max_abs_pull=round(ns_17["max_abs_pull"], 3),
            pulls={k: round(v, 3) for k, v in ns_17["pulls"].items()},
            yukawa_perturbativity=ns_17["yukawa_perturbativity"],
            preds=ns_17["preds"],
        ),
        NS_PH16_a13_h23=dict(
            n_params=ns_16_h23["n_params"],
            dof=ns_16_h23["dof"],
            chi2=round(ns_16_h23["chi2"], 4),
            max_abs_pull=round(ns_16_h23["max_abs_pull"], 3),
            pulls={k: round(v, 3) for k, v in ns_16_h23["pulls"].items()},
            yukawa_perturbativity=ns_16_h23["yukawa_perturbativity"],
            preds=ns_16_h23["preds"],
        ),
    )
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
