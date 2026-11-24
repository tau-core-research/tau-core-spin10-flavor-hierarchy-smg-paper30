#!/usr/bin/env python3
"""Basin-seeded refinement & exact forensic audit of:
  1. Perturbative SUSY PH18 / PH17_a13 / PH16 submodels (seeded from S1_PH17_a13 & NS_PH18),
  2. Exact pulls, Yukawa norms, M_R spectrum, and falsifiable neutrino predictions,
  3. Exact Dimopoulos-Wilczek (missing-VEV) triplet suppression vs 1-loop gauge unification
     shift from the extra H'_u, H'_d doublet mass M' (no handwaving!).
"""

import json
import math
import sys
import time
import numpy as np
from scipy.optimize import least_squares

from verify_3gen_spin10_2T_global_fit import TARGETS as NS_TARGETS
from verify_3gen_spin10_120H_parity_and_nogo import diag_hermitian
from verify_SUSY_holomorphic_10_126_flavor import run_mssm_targets
from verify_SUSY_10_126_120_parity_and_proton_decay import (
    eval_parity_hermitian,
    audit_spin10_proton_decay,
    strip_internal,
)


def params_to_vec(p, active):
    F = p["F_diag"]
    H = p["H"]
    A = p["A_12_13_23"]
    d = dict(
        f1=F[0], f2=F[1], f3=F[2],
        h11=H[0][0], h22=H[1][1], h33=H[2][2],
        h12=H[0][1], h13=H[0][2], h23=H[1][2],
        a12=A[0], a13=A[1], a23=A[2],
        s=p["s"], c_e=p["c_e"], c_u=p["c_u"], c_nu=p["c_nu"],
    )
    return np.array([d[n] for n in active], dtype=float)


def seeded_refine(targets, tanb, seed_params, zero_mask=(), n_perturb=12, pert_cap=2.0):
    obs_keys = list(targets.keys())
    mt_t = targets["m_t"][0]
    all_names = [
        "f1", "f2", "f3",
        "h11", "h22", "h33", "h12", "h13", "h23",
        "a12", "a13", "a23",
        "s", "c_e", "c_u", "c_nu",
    ]
    active = [n for n in all_names if n not in zero_mask]
    idx_map = {n: i for i, n in enumerate(active)}

    def unpack_eval(x):
        def get(name):
            return float(x[idx_map[name]]) if name in idx_map else 0.0
        F = np.diag([get("f1"), get("f2"), get("f3")])
        H = np.array([
            [get("h11"), get("h12"), get("h13")],
            [get("h12"), get("h22"), get("h23")],
            [get("h13"), get("h23"), get("h33")],
        ], dtype=float)
        A = np.array([
            [0.0, get("a12"), get("a13")],
            [-get("a12"), 0.0, get("a23")],
            [-get("a13"), -get("a23"), 0.0],
        ], dtype=float)
        s, c_e, c_u, c_nu = get("s"), get("c_e"), get("c_u"), get("c_nu")
        Mu_u = H + s * F + 1j * c_u * A
        su_u, _, _ = diag_hermitian(Mu_u)
        if su_u[2] < 1e-12:
            return None
        r_mag = mt_t / su_u[2]
        return eval_parity_hermitian(H, F, A, r_mag, s, c_e, c_u, c_nu, targets, tanb, beta_II=0.0)

    def resid(x):
        ev = unpack_eval(x)
        if ev is None:
            return np.full(19, 1e3)
        pvec = [ev["pulls"][k] for k in obs_keys]
        ymax = ev["yukawa_perturbativity"]["max_Y_all"]
        pvec.append(0.0 if ymax <= pert_cap else 10.0 * (ymax - pert_cap))
        return np.array(pvec, dtype=float)

    x_base = params_to_vec(seed_params, active)
    rng = np.random.default_rng(99101 + len(zero_mask) * 37)
    best = None
    for j in range(n_perturb):
        x0 = x_base.copy() if j == 0 else x_base * (1.0 + rng.normal(0.0, 0.08 * j / n_perturb, size=len(active)))
        res = least_squares(resid, x0, method="trf", x_scale="jac", ftol=1e-13, xtol=1e-13, gtol=1e-13, max_nfev=3000)
        ev = unpack_eval(res.x)
        if ev is not None and ev["yukawa_perturbativity"]["max_Y_all"] <= pert_cap + 0.05:
            if best is None or ev["chi2"] < best["chi2"]:
                best = ev
    best["n_params"] = len(active) + 2
    best["dof"] = 18 - best["n_params"]
    best["zero_mask"] = list(zero_mask)
    return best


def audit_dimopoulos_wilczek_unification(cancel_req_SK: float, M_U: float = 7.97e15):
    """Exact 1-loop gauge-unification audit of Dimopoulos-Wilczek missing-VEV suppression!
    W_DT = lambda * 10_H * 45_{B-L} * 10'_H + M' * 10'_H * 10'_H.
      - Triplet mass matrix: [[0, M_T], [-M_T, M']], M_T = lambda v_{B-L} ~ M_U
        -> (M_T^-1)_{11} = M' / M_T^2  (suppression factor = M' / M_T = 1 / cancel_req_SK!)
      - Doublet mass matrix: [[0, 0], [0, M']] -> one massless MSSM pair (H_u, H_d) and
        one extra heavy doublet pair (H'_u, H'_d) of mass M'!
      - Check exact 1-loop MSSM beta-function shift from the extra doublet pair (H'_u, H'_d)
        running from M' to M_U:
          Delta b_1 = +3/5,  Delta b_2 = +1,  Delta b_3 = 0!
      - Solve exact 1-loop unification with (H'_u, H'_d) at M' = M_U / cancel_req_SK
        vs M_SUSY = 2 TeV!
    """
    M_Z, M_SUSY = 91.1876, 2000.0
    a1_Z, a2_Z, a3_Z = 59.01, 29.58, 1.0 / 0.1180  # SU(5)-normalized alpha_1^-1 = 3/5 * 98.35 = 59.01
    b_sm = (41.0 / 10.0, -19.0 / 6.0, -7.0)
    b_mssm = (33.0 / 5.0, 1.0, -3.0)
    b_extra_H = (3.0 / 5.0, 1.0, 0.0)

    t_s = math.log(M_SUSY / M_Z) / (2.0 * math.pi)
    a1_s = a1_Z - b_sm[0] * t_s
    a2_s = a2_Z - b_sm[1] * t_s
    a3_s = a3_Z - b_sm[2] * t_s

    # Baseline MSSM (no extra doublet, M' = M_U):
    t_u0 = (a1_s - a2_s) / (b_mssm[0] - b_mssm[1])
    MU0 = M_SUSY * math.exp(2.0 * math.pi * t_u0)
    aU0 = a1_s - b_mssm[0] * t_u0
    a3_miss0 = (a3_s - b_mssm[2] * t_u0) - aU0

    # With extra doublet pair at M' = M_T / cancel_req_SK (for M_T = MU0):
    M_prime = MU0 / cancel_req_SK
    t_p = math.log(M_prime / M_SUSY) / (2.0 * math.pi)
    a1_p = a1_s - b_mssm[0] * t_p
    a2_p = a2_s - b_mssm[1] * t_p
    a3_p = a3_s - b_mssm[2] * t_p

    b_hi = (b_mssm[0] + b_extra_H[0], b_mssm[1] + b_extra_H[1], b_mssm[2] + b_extra_H[2])
    t_hi = (a1_p - a2_p) / (b_hi[0] - b_hi[1])
    MU_dw = M_prime * math.exp(2.0 * math.pi * t_hi)
    aU_dw = a1_p - b_hi[0] * t_hi
    a3_miss_dw = (a3_p - b_hi[2] * t_hi) - aU_dw
    # Also compute implied alpha_s(M_Z) if exact unification alpha_3(M_U) = alpha_U holds:
    a3_s_pred = aU_dw + b_hi[2] * t_hi + b_mssm[2] * t_p
    a3_Z_pred = a3_s_pred + b_sm[2] * t_s
    alpha_s_pred_dw = 1.0 / a3_Z_pred

    a3_Z_pred0 = (aU0 + b_mssm[2] * t_u0) + b_sm[2] * t_s
    alpha_s_pred0 = 1.0 / a3_Z_pred0

    return dict(
        cancel_req_SK=round(cancel_req_SK, 1),
        M_prime_GeV=M_prime,
        baseline_MSSM=dict(M_U_GeV=MU0, alpha_U_inv=round(aU0, 3), alpha3_inv_miss=round(a3_miss0, 3), alpha_s_MZ_pred=round(alpha_s_pred0, 4)),
        with_DW_extra_doublet=dict(M_U_GeV=MU_dw, alpha_U_inv=round(aU_dw, 3), alpha3_inv_miss=round(a3_miss_dw, 3), alpha_s_MZ_pred=round(alpha_s_pred_dw, 4)),
        experimental_alpha_s_MZ="0.1180 +/- 0.0009",
    )


def main():
    t0 = time.time()
    from pathlib import Path
    data_dir = Path(__file__).resolve().parent.parent / "data"
    with open(data_dir / "susy_ph_10_126_120_results.json") as f:
        data = json.load(f)

    targs_S1, _ = run_mssm_targets(delta_b=0.0, tanb=50.0)
    seed_S1 = data["step2_predictive_texture_reduction"]["S1_SUSY_tb50_db0"]["PH17_a13"]["fit"]["params"]
    seed_NS = data["step1_full_PH18_benchmarks"]["NS_NonSUSY_SM_RG_check"]["PH18_fit"]["params"]

    out = {}

    # 1. Polish S1 PH18 and test ALL 1-zero extensions of S1 PH17_a13 (16p, dof=2):
    s1_ph18 = seeded_refine(targs_S1, 50.0, seed_S1, zero_mask=(), n_perturb=10)
    s1_ph17 = seeded_refine(targs_S1, 50.0, seed_S1, zero_mask=("a13",), n_perturb=12)
    pdec_s1_17 = audit_spin10_proton_decay(s1_ph17, tanb=50.0)
    dw_audit = audit_dimopoulos_wilczek_unification(pdec_s1_17["triplet_cancellation_factor_needed_at_2TeV_MU"])
    sys.stderr.write(
        f"[S1 PH18 seeded] chi2={s1_ph18['chi2']:.6f}, max_pull={s1_ph18['max_abs_pull']:.3f}s, max|Y|={s1_ph18['yukawa_perturbativity']['max_Y_all']}\n"
    )
    sys.stderr.write(
        f"[S1 PH17_a13 polished] chi2={s1_ph17['chi2']:.4f}, max_pull={s1_ph17['max_abs_pull']:.2f}s, max|Y|={s1_ph17['yukawa_perturbativity']['max_Y_all']}, "
        f"tau2T={pdec_s1_17['tau_p_at_Msq2TeV_Mw500GeV_MT_MU_yr']:.2e} yr, cancel={pdec_s1_17['triplet_cancellation_factor_needed_at_2TeV_MU']}x\n"
    )
    sys.stderr.write(
        f"[DW Unification Audit] M'={dw_audit['M_prime_GeV']:.2e} GeV: "
        f"alpha3_miss {dw_audit['baseline_MSSM']['alpha3_inv_miss']} -> {dw_audit['with_DW_extra_doublet']['alpha3_inv_miss']}, "
        f"alpha_s(M_Z) {dw_audit['baseline_MSSM']['alpha_s_MZ_pred']} -> {dw_audit['with_DW_extra_doublet']['alpha_s_MZ_pred']}\n"
    )

    out["S1_PH18_seeded"] = strip_internal(s1_ph18)
    out["S1_PH17_a13_polished"] = dict(fit=strip_internal(s1_ph17), dim5_proton_decay=pdec_s1_17, DW_unification_audit=dw_audit)

    # Test 16-param submodels seeded from S1 PH17_a13:
    out["S1_PH16_from_PH17_a13"] = {}
    for extra_z in ("h11", "h22", "h13", "h12", "h23", "a12", "a23", "s", "c_u", "c_e", "c_nu"):
        r16 = seeded_refine(targs_S1, 50.0, s1_ph17["params"], zero_mask=("a13", extra_z), n_perturb=12)
        pdec16 = audit_spin10_proton_decay(r16, tanb=50.0)
        sys.stderr.write(
            f"[S1 PH16_a13_{extra_z}] chi2={r16['chi2']:.4f}, max_pull={r16['max_abs_pull']:.2f}s, "
            f"max|Y|={r16['yukawa_perturbativity']['max_Y_all']}, s23^2={r16['obs']['l_s23_sq']:.4f}, "
            f"dCP={r16['preds']['delta_CP_lepton_deg']} deg, tau2T={pdec16['tau_p_at_Msq2TeV_Mw500GeV_MT_MU_yr']:.2e} yr\n"
        )
        out["S1_PH16_from_PH17_a13"][f"a13_{extra_z}"] = dict(fit=strip_internal(r16), dim5_proton_decay=pdec16)

    # 2. Test Non-SUSY Parity-Hermitian predictive 17p & 16p models (ZERO dim-5 proton decay + theta_QCD == 0!):
    out["NS_PH17_and_PH16"] = {}
    for zname, zmask in [
        ("PH17_a13", ("a13",)),
        ("PH17_h11", ("h11",)),
        ("PH17_h22", ("h22",)),
        ("PH17_h13", ("h13",)),
        ("PH16_a13_h11", ("a13", "h11")),
        ("PH16_a13_h22", ("a13", "h22")),
        ("PH16_a13_h13", ("a13", "h13")),
        ("PH16_h11_h13", ("h11", "h13")),
        ("PH16_h22_h13_Z6", ("h22", "h13")),
    ]:
        rns = seeded_refine(NS_TARGETS, None, seed_NS, zero_mask=zmask, n_perturb=14)
        sys.stderr.write(
            f"[NS {zname} ({rns['n_params']}p, dof={rns['dof']})] chi2={rns['chi2']:.4f}, max_pull={rns['max_abs_pull']:.2f}s, "
            f"max|Y|={rns['yukawa_perturbativity']['max_Y_all']}, s23^2={rns['obs']['l_s23_sq']:.4f}, "
            f"dCP={rns['preds']['delta_CP_lepton_deg']} deg, sum_mnu={rns['preds']['sum_m_nu_eV']}\n"
        )
        out["NS_PH17_and_PH16"][zname] = strip_internal(rns)

    sys.stderr.write(f"[Basin refinement & DW unification audit done in {time.time()-t0:.1f}s]\n")
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
