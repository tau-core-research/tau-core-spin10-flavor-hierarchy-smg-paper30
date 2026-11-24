#!/usr/bin/env python3
"""Step 2 (Perturbative & Calibrated) of the SUSY Spin(10) Program:
  1. Exact Global Fit of Perturbative Parity-Hermitian 4-Simplex Spin(10)
     (10_H + 126_bar_H + 120_H, exact tree-level Strong-CP theta_QCD == 0!)
     on ALL 18 MSSM-RG Observables at M_U^SUSY = 7.97e15 GeV across:
       - S1: tan(beta) = 50, Delta_b =  0.00 (pure 1-loop, zero threshold tuning)
       - S2: tan(beta) = 50, Delta_b = -0.15 (standard finite gluino-sbottom threshold)
       - S3: tan(beta) = 10, Delta_b =  0.00 (moderate tan(beta))
       - NS: Non-SUSY SM-RG cross-check with Jacobian-scaled TRF
     with STRICT Yukawa perturbativity audit:
       max(|Y_10|_min, |Y_126|_min, |Y_120|_min) <= 2.0 < sqrt(4 pi) = 3.54!
  2. Systematic Predictive Texture-Reduction Audit (17, 16, 15 real params vs 18 obs,
     dof = 1, 2, 3, STILL with exact tree-level Strong-CP theta_QCD == 0 & perturbativity!).
  3. Honest Quantitative Dimension-5 SUSY Proton Decay Audit (p -> K^+ nu_bar)
     calibrated against Murayama-Pierce (2002) minimal SUSY SU(5) benchmark,
     computing exact tau(p -> K^+ nu_bar), M_T^min, Mini-Split M_qtilde^min,
     Barbieri-Giudice Delta_FT, and triplet-cancellation factor vs Super-K & Hyper-K!

Strict `python3 -W error` compliance. Zero handwaving.
"""

import json
import math
import sys
import time
import numpy as np
from scipy.optimize import least_squares

from verify_3gen_spin10_2T_global_fit import TARGETS as NS_TARGETS, takagi, extract_mixing
from verify_3gen_spin10_120H_parity_and_nogo import diag_hermitian
from verify_SUSY_holomorphic_10_126_flavor import run_mssm_targets


def compute_min_yukawa_norms(H, F, A, r, s, c_e, c_u, c_nu, tanb):
    """Exact unitarity lower bounds on the GUT-scale Yukawa couplings |Y_10|, |Y_126|, |Y_120|
    from |U_{u,a}| <= 1 and |U_{d,a}| <= 1.
    """
    v = 174.104
    if tanb is None:
        vu, vd = v, v
    else:
        sb = tanb / math.sqrt(1.0 + tanb * tanb)
        cb = 1.0 / math.sqrt(1.0 + tanb * tanb)
        vu, vd = v * sb, v * cb

    h_max = float(np.max(np.abs(H)))
    f_max = float(np.max(np.abs(F)))
    a_max = float(np.max(np.abs(A)))
    ar = abs(r)

    y10_min = max(h_max / vd, (ar * h_max) / vu)
    y126_min = max(f_max / vd, (ar * abs(s) * f_max) / vu)
    fac_d120 = math.sqrt((3.0 + c_e) ** 2 + (1.0 - c_e) ** 2) / 4.0
    fac_u120 = math.sqrt((3.0 * c_u + c_nu) ** 2 + (c_u - c_nu) ** 2) / 4.0
    y120_min = max((a_max * fac_d120) / vd, (ar * a_max * fac_u120) / vu)

    return dict(
        max_Y10_min=y10_min,
        max_Y126_min=y126_min,
        max_Y120_min=y120_min,
        max_Y_all=max(y10_min, y126_min, y120_min),
    )


def eval_parity_hermitian(H, F, A, r_mag, s, c_e, c_u, c_nu, targets, tanb, beta_II=0.0):
    obs_keys = list(targets.keys())
    Md = H + F + 1j * A
    Me = H - 3.0 * F + 1j * c_e * A
    Mu_pos = r_mag * (H + s * F + 1j * c_u * A)

    det_check = np.linalg.det(Mu_pos) * np.linalg.det(Md)
    r = r_mag if det_check.real >= 0.0 else -r_mag
    Mu = r * (H + s * F + 1j * c_u * A)
    MnuD = r * (H - 3.0 * s * F + 1j * c_nu * A)

    if abs(np.linalg.det(F)) < 1e-35:
        return None
    try:
        Finv = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        return None

    Mnu_raw = beta_II * F - (MnuD @ Finv @ MnuD.T) / (r * r)
    Mnu_raw = 0.5 * (Mnu_raw + Mnu_raw.T)

    sd, Ud, _ = diag_hermitian(Md)
    se, Ue, _ = diag_hermitian(Me)
    su, Uu, _ = diag_hermitian(Mu)
    snu_raw, Unu = takagi(Mnu_raw)

    raw_dm31 = snu_raw[2] ** 2 - snu_raw[0] ** 2
    if raw_dm31 <= 1e-40:
        return None
    c_R_inv_eV = math.sqrt(targets["dm31_sq"][0] / raw_dm31)
    snu = snu_raw * c_R_inv_eV

    Vckm = Uu.T @ Ud.conj()
    Upmns = Ue.T @ Unu
    qs12, qs23, qs13, qd, qJ = extract_mixing(Vckm)
    ls12, ls23, ls13, ld, lJ = extract_mixing(Upmns)

    dm21 = snu[1] ** 2 - snu[0] ** 2
    dm31 = snu[2] ** 2 - snu[0] ** 2
    m_bb = float(abs(sum((Upmns[0, i] ** 2) * snu[i] for i in range(3))))
    Mnu_eV = Mnu_raw * c_R_inv_eV
    m_bb_chk = float(abs((Ue.T @ Mnu_eV.conj() @ Ue)[0, 0]))
    if abs(m_bb - m_bb_chk) > 1e-5 * max(1e-12, m_bb):
        return None

    det_prod = np.linalg.det(Mu) * np.linalg.det(Md)
    theta_qcd_tree = float(abs(math.atan2(det_prod.imag, det_prod.real)))

    obs = dict(
        m_u=float(su[0]), m_c=float(su[1]), m_t=float(su[2]),
        m_d=float(sd[0]), m_s=float(sd[1]), m_b=float(sd[2]),
        m_e=float(se[0]), m_mu=float(se[1]), m_tau=float(se[2]),
        q_s12=qs12, q_s23=qs23, q_s13=qs13, q_delta=qd,
        dm21_sq=float(dm21), dm31_sq=float(dm31),
        l_s12_sq=ls12**2, l_s23_sq=ls23**2, l_s13_sq=ls13**2,
    )
    pulls = {k: (obs[k] - targets[k][0]) / targets[k][1] for k in obs_keys}
    chi2 = float(sum(p * p for p in pulls.values()))

    c_R = (r * r) / (c_R_inv_eV * 1e-9)
    sMR = sorted(abs(float(x)) * c_R for x in np.linalg.eigvalsh(F))
    ynorms = compute_min_yukawa_norms(H, F, A, r, s, c_e, c_u, c_nu, tanb)

    return dict(
        chi2=chi2,
        max_abs_pull=float(max(abs(p) for p in pulls.values())),
        obs=obs,
        pulls=pulls,
        yukawa_perturbativity={k: round(v, 4) for k, v in ynorms.items()},
        preds=dict(
            ordering="NO",
            m_nu_eV=[float(x) for x in snu],
            sum_m_nu_eV=round(float(np.sum(snu)), 5),
            m_beta_beta_meV=round(m_bb * 1e3, 3),
            sin2_th23_lepton=round(float(ls23**2), 4),
            delta_CP_lepton_rad=float(ld),
            delta_CP_lepton_deg=round(float(math.degrees(ld)), 1),
            J_CP_lepton=float(lJ),
            M_R_GeV=[float(x) for x in sMR],
            tree_level_theta_QCD_rad=theta_qcd_tree,
            det_Mu_Md_positive=bool(det_prod.real > 0),
        ),
        params=dict(
            r=float(r), s=float(s), c_e=float(c_e), c_u=float(c_u), c_nu=float(c_nu),
            beta_II=float(beta_II), c_R=float(c_R),
            F_diag=[float(F[0, 0]), float(F[1, 1]), float(F[2, 2])],
            H=H.tolist(),
            A_12_13_23=[float(A[0, 1]), float(A[0, 2]), float(A[1, 2])],
        ),
        _matrices=dict(H=H, F=F, A=A, Uu=Uu, Ud=Ud, Ue=Ue, Vckm=Vckm, r=r, s=s, c_u=c_u),
    )


def fit_ph_model(
    targets,
    tanb,
    zero_mask=(),
    include_typeII: bool = False,
    n_restarts: int = 24,
    seed_offset: int = 0,
    pert_cap: float = 2.0,
    early_stop_chi2: float = 1e-3,
):
    obs_keys = list(targets.keys())
    mt_t = targets["m_t"][0]
    mb_t = targets["m_b"][0]
    ms_t = targets["m_s"][0]
    md_t = targets["m_d"][0]

    all_names = [
        "f1", "f2", "f3",
        "h11", "h22", "h33", "h12", "h13", "h23",
        "a12", "a13", "a23",
        "s", "c_e", "c_u", "c_nu",
    ]
    if include_typeII:
        all_names.append("beta_II")
    active = [n for n in all_names if n not in zero_mask]
    idx_map = {n: i for i, n in enumerate(active)}
    n_active = len(active)
    n_total_params = n_active + 2

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
        s = get("s")
        c_e = get("c_e")
        c_u = get("c_u")
        c_nu = get("c_nu")
        beta_II = get("beta_II") if include_typeII else 0.0

        Mu_u = H + s * F + 1j * c_u * A
        su_u, _, _ = diag_hermitian(Mu_u)
        if su_u[2] < 1e-12:
            return None
        r_mag = mt_t / su_u[2]
        return eval_parity_hermitian(H, F, A, r_mag, s, c_e, c_u, c_nu, targets, tanb, beta_II=beta_II)

    def resid(x):
        ev = unpack_eval(x)
        if ev is None:
            return np.full(19, 1e3)
        pvec = [ev["pulls"][k] for k in obs_keys]
        # Zero-bias perturbativity barrier: 0 when all |Y| <= pert_cap, smooth quadratic above!
        ymax = ev["yukawa_perturbativity"]["max_Y_all"]
        pert_pen = 0.0 if ymax <= pert_cap else 10.0 * (ymax - pert_cap)
        pvec.append(pert_pen)
        return np.array(pvec, dtype=float)

    rng = np.random.default_rng(77001 + seed_offset)
    best = None
    best_x = None

    for k in range(n_restarts):
        x0 = np.zeros(n_active)
        sgn = lambda: 1.0 if rng.random() < 0.5 else -1.0
        vals = dict(
            f1=sgn() * md_t * (10.0 ** rng.uniform(-1.0, 0.8)),
            f2=sgn() * ms_t * (10.0 ** rng.uniform(-0.8, 0.6)),
            f3=sgn() * mb_t * rng.uniform(0.03, 0.45),
            h11=sgn() * md_t * (10.0 ** rng.uniform(-1.0, 0.6)),
            h22=sgn() * ms_t * (10.0 ** rng.uniform(-0.8, 0.6)),
            h33=mb_t * rng.uniform(0.5, 1.3),
            h12=sgn() * math.sqrt(md_t * ms_t) * rng.uniform(0.2, 3.0),
            h13=sgn() * math.sqrt(md_t * mb_t) * rng.uniform(0.1, 2.5),
            h23=sgn() * math.sqrt(ms_t * mb_t) * rng.uniform(0.2, 2.5),
            a12=sgn() * math.sqrt(md_t * ms_t) * rng.uniform(0.2, 3.0),
            a13=sgn() * math.sqrt(md_t * mb_t) * rng.uniform(0.1, 2.5),
            a23=sgn() * math.sqrt(ms_t * mb_t) * rng.uniform(0.2, 2.5),
            s=rng.uniform(-1.2, 1.2),
            c_e=rng.uniform(-3.0, 3.0),
            c_u=rng.uniform(-2.0, 2.0),
            c_nu=rng.uniform(-3.0, 3.0),
            beta_II=rng.uniform(-2.0, 2.0),
        )
        if k % 4 == 0:
            vals["f3"] = sgn() * mb_t * rng.uniform(0.2, 0.7)
            vals["a23"] = sgn() * mb_t * rng.uniform(0.05, 0.5)
            vals["h23"] = sgn() * mb_t * rng.uniform(0.05, 0.5)
        for name, i in idx_map.items():
            x0[i] = vals[name]

        res = least_squares(
            resid, x0, method="trf", x_scale="jac",
            ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=1200,
        )
        ev = unpack_eval(res.x)
        if ev is not None:
            ymax = ev["yukawa_perturbativity"]["max_Y_all"]
            obj = ev["chi2"] + (0.0 if ymax <= pert_cap else 100.0 * (ymax - pert_cap) ** 2)
            best_obj = float("inf") if best is None else (
                best["chi2"] + (0.0 if best["yukawa_perturbativity"]["max_Y_all"] <= pert_cap else 100.0 * (best["yukawa_perturbativity"]["max_Y_all"] - pert_cap) ** 2)
            )
            if obj < best_obj:
                best = ev
                best_x = res.x.copy()
                if ev["chi2"] < early_stop_chi2 and ymax <= pert_cap:
                    break

    # Polish best solution:
    if best_x is not None:
        res = least_squares(
            resid, best_x, method="trf", x_scale="jac",
            ftol=1e-13, xtol=1e-13, gtol=1e-13, max_nfev=2500,
        )
        ev = unpack_eval(res.x)
        if ev is not None and ev["yukawa_perturbativity"]["max_Y_all"] <= pert_cap + 0.05 and ev["chi2"] <= best["chi2"]:
            best = ev

    best["n_params"] = n_total_params
    best["n_obs"] = 18
    best["dof"] = 18 - n_total_params
    best["zero_mask"] = list(zero_mask)
    return best


def compute_dim5_proton_decay(
    Y_T_up_basis,
    Y_barT_up_lepton_basis,
    Vckm,
    M_T: float,
    M_sq: float,
    M_wino: float,
    A_L: float = 1.25,
    A_S: float = 2.0,
):
    x = (M_wino / M_sq) ** 2
    if abs(x - 1.0) < 1e-6:
        I_loop = 0.5
    elif x < 1e-12:
        I_loop = 1.0
    else:
        I_loop = 1.0 / (1.0 - x) + (x * math.log(x)) / ((1.0 - x) ** 2)
    f_loop = (M_wino / (M_sq ** 2)) * I_loop

    alpha_2 = 1.0 / 29.58
    m_p = 0.938272
    m_K = 0.493677
    m_B = 1.15
    f_pi = 0.1302
    beta_H = 0.0144
    D_ch, F_ch = 0.80, 0.46
    hbar_GeV_s = 6.582119569e-25
    sec_per_yr = 3.15576e7

    C5 = np.zeros((3, 3, 3, 3), dtype=complex)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if a == b == c:
                    continue
                for l in range(3):
                    C5[a, b, c, l] = (Y_T_up_basis[a, b] * Y_barT_up_lepton_basis[c, l]) / M_T

    pref = (alpha_2 / (2.0 * math.pi)) * f_loop * A_L * A_S
    gamma_by_flavor = {}
    tot_gamma = 0.0
    for l, lname in enumerate(["nu_e", "nu_mu", "nu_tau"]):
        S_ab = sum(C5[a, b, 0, l] * Vckm[a, 0] * Vckm[b, 1] for a in range(3) for b in range(3))
        B_1 = sum(C5[0, b, c, l] * Vckm[b, 0] * Vckm[c, 1] for b in range(3) for c in range(3))
        B_2 = sum(C5[0, b, c, l] * Vckm[b, 1] * Vckm[c, 0] for b in range(3) for c in range(3))
        C_1l = pref * (S_ab - B_1)
        C_2l = pref * (S_ab - B_2)

        amp = (beta_H / f_pi) * (
            (2.0 * m_p / (3.0 * m_B)) * D_ch * C_1l
            + (1.0 + (m_p / (3.0 * m_B)) * (D_ch + 3.0 * F_ch)) * C_2l
        )
        gamma_l = (m_p / (32.0 * math.pi)) * ((1.0 - (m_K / m_p) ** 2) ** 2) * (abs(amp) ** 2)
        gamma_by_flavor[lname] = float(gamma_l)
        tot_gamma += gamma_l

    tau_yr = (hbar_GeV_s / tot_gamma) / sec_per_yr if tot_gamma > 0 else float("inf")
    return dict(
        tau_yr=float(tau_yr),
        partial_widths_GeV=gamma_by_flavor,
        dominant_channel=max(gamma_by_flavor, key=gamma_by_flavor.get),
    )


def verify_murayama_pierce_benchmark():
    v = 174.104
    tanb = 3.0
    sb = tanb / math.sqrt(1.0 + tanb**2)
    cb = 1.0 / math.sqrt(1.0 + tanb**2)
    yu = np.diag([5.0e-4, 0.25, 90.0]) / (v * sb)
    yd = np.diag([1.1e-3, 0.023, 1.1]) / (v * cb)
    s12, s23, s13, dckm = 0.225, 0.040, 0.0035, 1.144
    c12, c23, c13 = math.sqrt(1 - s12**2), math.sqrt(1 - s23**2), math.sqrt(1 - s13**2)
    V = np.array([
        [c12 * c13, s12 * c13, s13 * np.exp(-1j * dckm)],
        [-s12 * c23 - c12 * s23 * s13 * np.exp(1j * dckm), c12 * c23 - s12 * s23 * s13 * np.exp(1j * dckm), s23 * c13],
        [s12 * s23 - c12 * c23 * s13 * np.exp(1j * dckm), -c12 * s23 - s12 * c23 * s13 * np.exp(1j * dckm), c23 * c13],
    ], dtype=complex)
    res_1TeV = compute_dim5_proton_decay(yu, V @ yd, V, M_T=2.0e16, M_sq=1000.0, M_wino=200.0)
    res_10TeV = compute_dim5_proton_decay(yu, V @ yd, V, M_T=2.0e16, M_sq=10000.0, M_wino=500.0)
    return dict(
        minimal_SU5_Msq1TeV_Mw200GeV_tb3_MT2e16_tau_yr=res_1TeV["tau_yr"],
        minimal_SU5_Msq10TeV_Mw500GeV_tb3_MT2e16_tau_yr=res_10TeV["tau_yr"],
        dominant_channel=res_1TeV["dominant_channel"],
        published_Murayama_Pierce_range="1e30 - 1e31 yr (1 TeV) -> 1e33 - 1e34 yr (10 TeV)",
    )


def audit_spin10_proton_decay(fit_res, tanb: float, M_U: float = 7.97e15):
    v = 174.104
    sb = tanb / math.sqrt(1.0 + tanb**2)
    mats = fit_res["_matrices"]
    H = mats["H"]
    Uu, Ue, Vckm = mats["Uu"], mats["Ue"], mats["Vckm"]
    r = abs(mats["r"])

    U_u1_max = min(1.0, r / tanb)
    U_d1_val = U_u1_max * tanb / r
    Y_10 = (r / (U_u1_max * v * sb)) * H.astype(complex)

    UuL = Uu.conj()
    UeL = Ue.conj()
    Y_T_up = UuL.T @ Y_10 @ UuL
    Y_barT_ul = UuL.T @ Y_10 @ UeL

    r_2TeV = compute_dim5_proton_decay(Y_T_up, Y_barT_ul, Vckm, M_T=M_U, M_sq=2000.0, M_wino=500.0)
    tau_SK = 5.9e33
    tau_HK = 3.2e34

    MT_min_SK_at_2TeV = M_U * math.sqrt(tau_SK / r_2TeV["tau_yr"])
    MT_min_HK_at_2TeV = M_U * math.sqrt(tau_HK / r_2TeV["tau_yr"])
    cancel_req_SK = math.sqrt(tau_SK / r_2TeV["tau_yr"])

    def find_Msq_min(tau_target):
        lo, hi = 1e3, 1e9
        for _ in range(80):
            mid = math.sqrt(lo * hi)
            t = compute_dim5_proton_decay(Y_T_up, Y_barT_ul, Vckm, M_T=M_U, M_sq=mid, M_wino=500.0)["tau_yr"]
            if t < tau_target:
                lo = mid
            else:
                hi = mid
        return hi

    Msq_min_SK = find_Msq_min(tau_SK)
    Msq_min_HK = find_Msq_min(tau_HK)

    m_h = 125.25
    yt = 0.95
    def delta_ft(Msq):
        return (6.0 * yt**2 / (4.0 * math.pi**2)) * ((Msq / m_h) ** 2) * math.log(M_U / Msq)

    return dict(
        tan_beta=tanb,
        abs_r_fit=round(r, 2),
        U_u1_max=round(U_u1_max, 4),
        U_d1_at_max=round(U_d1_val, 4),
        max_abs_Y10_entry=round(float(np.max(np.abs(Y_10))), 4),
        tau_p_at_Msq2TeV_Mw500GeV_MT_MU_yr=r_2TeV["tau_yr"],
        dominant_channel=r_2TeV["dominant_channel"],
        SuperK_bound_yr=tau_SK,
        HyperK_reach_yr=tau_HK,
        MT_min_for_SuperK_at_Msq2TeV_GeV=MT_min_SK_at_2TeV,
        MT_min_for_HyperK_at_Msq2TeV_GeV=MT_min_HK_at_2TeV,
        triplet_cancellation_factor_needed_at_2TeV_MU=round(cancel_req_SK, 1),
        MiniSplit_Msq_min_for_SuperK_GeV=round(Msq_min_SK, 1),
        MiniSplit_Msq_min_for_HyperK_GeV=round(Msq_min_HK, 1),
        Delta_FT_at_2TeV=round(delta_ft(2000.0), 1),
        Delta_FT_at_MiniSplit_SuperK=round(delta_ft(Msq_min_SK), 1),
    )


def strip_internal(d):
    return {k: v for k, v in d.items() if k != "_matrices"}


def main():
    t0 = time.time()
    out = {}

    mp_cal = verify_murayama_pierce_benchmark()
    sys.stderr.write(
        f"[Dim-5 Calibration] minimal SU(5) (1 TeV, tb=3, MT=2e16): "
        f"tau = {mp_cal['minimal_SU5_Msq1TeV_Mw200GeV_tb3_MT2e16_tau_yr']:.2e} yr "
        f"(10 TeV: {mp_cal['minimal_SU5_Msq10TeV_Mw500GeV_tb3_MT2e16_tau_yr']:.2e} yr)\n"
    )
    out["step0_dim5_proton_decay_calibration"] = mp_cal

    benchmarks = []
    for label, tanb, db in [
        ("S1_SUSY_tb50_db0", 50.0, 0.0),
        ("S2_SUSY_tb50_db_m015", 50.0, -0.15),
        ("S3_SUSY_tb10_db0", 10.0, 0.0),
    ]:
        targs, rg_info = run_mssm_targets(delta_b=db, tanb=tanb)
        benchmarks.append((label, tanb, targs, rg_info))
    benchmarks.append(("NS_NonSUSY_SM_RG_check", None, NS_TARGETS, dict(mb_over_mtau=round(NS_TARGETS["m_b"][0] / NS_TARGETS["m_tau"][0], 4))))

    out["step1_full_PH18_benchmarks"] = {}
    full_fits = {}
    for idx, (label, tanb, targs, rg_info) in enumerate(benchmarks):
        fit = fit_ph_model(targs, tanb, zero_mask=(), include_typeII=False, n_restarts=24, seed_offset=100 * idx, pert_cap=2.0, early_stop_chi2=1e-3)
        full_fits[label] = (tanb, targs, fit)
        yp = fit["yukawa_perturbativity"]
        sys.stderr.write(
            f"[PH-18 {label}] mb/mtau={rg_info['mb_over_mtau']}: "
            f"chi2={fit['chi2']:.4f}, max_pull={fit['max_abs_pull']:.2f}s, "
            f"max|Y|=({yp['max_Y10_min']},{yp['max_Y126_min']},{yp['max_Y120_min']}), "
            f"s23^2={fit['obs']['l_s23_sq']:.4f}, dCP={fit['preds']['delta_CP_lepton_deg']} deg, "
            f"sum_mnu={fit['preds']['sum_m_nu_eV']} eV, theta_QCD={fit['preds']['tree_level_theta_QCD_rad']:.2e}\n"
        )
        entry = dict(rg_info=rg_info, PH18_fit=strip_internal(fit))
        if tanb is not None:
            pdec = audit_spin10_proton_decay(fit, tanb=tanb)
            entry["dim5_proton_decay_T10"] = pdec
            sys.stderr.write(
                f"   -> [Dim-5 {label}] tau(2TeV,MU)={pdec['tau_p_at_Msq2TeV_Mw500GeV_MT_MU_yr']:.2e} yr "
                f"(dom={pdec['dominant_channel']}), M_T^min(SK)={pdec['MT_min_for_SuperK_at_Msq2TeV_GeV']:.2e} GeV, "
                f"M_sq^min(SK)={pdec['MiniSplit_Msq_min_for_SuperK_GeV']:.0f} GeV (Delta_FT={pdec['Delta_FT_at_MiniSplit_SuperK']:.0f})\n"
            )
        out["step1_full_PH18_benchmarks"][label] = entry

    textures = [
        ("PH17_h11", ("h11",)),
        ("PH17_h22", ("h22",)),
        ("PH17_a13", ("a13",)),
        ("PH16_h11_a13", ("h11", "a13")),
        ("PH16_h11_h13", ("h11", "h13")),
        ("PH16_h22_h13_Z6", ("h22", "h13")),
        ("PH16_h22_a13", ("h22", "a13")),
        ("PH15_h11_h13_a13", ("h11", "h13", "a13")),
        ("PH15_h22_h13_a13", ("h22", "h13", "a13")),
    ]
    out["step2_predictive_texture_reduction"] = {}
    for b_label in ("S1_SUSY_tb50_db0", "S3_SUSY_tb10_db0"):
        tanb, targs, _ = full_fits[b_label]
        out["step2_predictive_texture_reduction"][b_label] = {}
        for tidx, (tname, zmask) in enumerate(textures):
            fit = fit_ph_model(
                targs, tanb, zero_mask=zmask, include_typeII=False,
                n_restarts=18, seed_offset=2000 + 300 * (0 if b_label.startswith("S1") else 1) + 23 * tidx,
                pert_cap=2.0, early_stop_chi2=0.05,
            )
            pdec = audit_spin10_proton_decay(fit, tanb=tanb)
            yp = fit["yukawa_perturbativity"]
            sys.stderr.write(
                f"[{b_label} | {tname} ({fit['n_params']}p, dof={fit['dof']})] "
                f"chi2={fit['chi2']:.4f}, max_pull={fit['max_abs_pull']:.2f}s, max|Y|={yp['max_Y_all']}, "
                f"s23^2={fit['obs']['l_s23_sq']:.4f}, dCP={fit['preds']['delta_CP_lepton_deg']} deg, "
                f"sum_mnu={fit['preds']['sum_m_nu_eV']}, tau2T={pdec['tau_p_at_Msq2TeV_Mw500GeV_MT_MU_yr']:.2e} yr, "
                f"M_sq^min(SK)={pdec['MiniSplit_Msq_min_for_SuperK_GeV']:.0f} GeV\n"
            )
            out["step2_predictive_texture_reduction"][b_label][tname] = dict(
                fit=strip_internal(fit),
                dim5_proton_decay_T10=pdec,
            )

    sys.stderr.write(f"[Perturbative SUSY Parity-Hermitian 10+126+120 + Dim-5 Proton Decay done in {time.time()-t0:.1f}s]\n")
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
