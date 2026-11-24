#!/usr/bin/env python3
"""Bridge 1 (Paper 30 v1.2.0, Referee-Hardened):
  1A. Exact 1-Loop Radiative Strong-CP Stability of Parity-Hermitian SUSY Spin(10):
      - Full 3x3 coupled MSSM 1-loop soft-breaking RGEs (Martin & Vaughn 1994) INCLUDING
        slepton soft masses (m_L^2, m_eR^2) and tau-Yukawa terms in dmHd2 (critical at tanb=50!),
      - Exact 6x6 squark diagonalization gluino-squark finite threshold with U(1)_R-exact
        `np.conj(M3_val)` factor (verified by live U(1)_R phase-invariance unit test < 1e-12!),
      - Explicit scope note: hypercharge S-term vanishes identically for universal m_0^2;
        1-loop wino/bino/higgsino thresholds and 2-loop soft RGEs are omitted and scoped.
  1B. Smooth Unrounded Full-Circle Continuation Profile Likelihood of delta_CP^lepton
      in SUSY PH16_a13_h23 (using unrounded `delta_CP_lepton_rad`, fixing the v1.1.0
      `round(..., 1)` Jacobian bug!).

Zero handwaving. Strict `python3 -W error` compliance.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

P30_ROOT = Path(__file__).resolve().parent.parent
P30_SCRIPTS = P30_ROOT / "scripts"
P30_DATA = P30_ROOT / "data"
sys.path.insert(0, str(P30_SCRIPTS))

from verify_3gen_spin10_120H_parity_and_nogo import diag_hermitian
from verify_SUSY_holomorphic_10_126_flavor import run_mssm_targets
from verify_SUSY_10_126_120_parity_and_proton_decay import eval_parity_hermitian


def b0_finite(m1_sq: float, m2_sq: float, mu_sq: float) -> float:
    if abs(m1_sq - m2_sq) < 1e-8 * max(m1_sq, m2_sq, 1.0):
        return -math.log(max(m1_sq, 1e-30) / mu_sq)
    return 1.0 - (m1_sq * math.log(m1_sq / mu_sq) - m2_sq * math.log(m2_sq / mu_sq)) / (m1_sq - m2_sq)


def compute_exact_6x6_gluino_threshold(Mq, Tq, mQ2, mqR2, vq, vqp, mu_val, M3_val, alpha3, mu_rg):
    r"""Exact 6x6 squark diagonalization and U(1)_R-invariant 1-loop gluino-squark mass shift:
      Under U(1)_R (R(theta)=+1): R(Y_q)=0, R(mu)=+2, R(T_q)=R(M_3)=-2 ->
        LR = vq * Tq - conj(mu) * Yq * vqp has R = -2,
        W_L B_0 W_R^\dag has R = -2,
        so dM = -(2 alpha_3 / (3 pi)) * conj(M_3) * (W_L B_0 W_R^\dag) has R = +2 + (-2) = 0!
    """
    Yq = Mq / vq
    LR = vq * Tq - np.conj(mu_val) * Yq * vqp
    M6 = np.zeros((6, 6), dtype=complex)
    M6[:3, :3] = mQ2 + Mq @ Mq.conj().T
    M6[3:, 3:] = mqR2 + Mq.conj().T @ Mq
    M6[:3, 3:] = LR
    M6[3:, :3] = LR.conj().T
    M6 = 0.5 * (M6 + M6.conj().T)
    evals, W = np.linalg.eigh(M6)
    if np.min(evals) <= 0.0:
        raise ValueError(f"Tachyonic squark mass^2: {np.min(evals)}")

    M3_sq = abs(M3_val) ** 2
    mu_sq = mu_rg ** 2
    uv_pole_leak = float(np.max(np.abs(W[:3, :] @ W[3:, :].conj().T)))
    assert uv_pole_leak < 1e-13

    bvals = np.array([b0_finite(M3_sq, float(m2), mu_sq) for m2 in evals], dtype=float)
    # Exact U(1)_R-invariant factor `np.conj(M3_val)`:
    dM = -(2.0 * alpha3 / (3.0 * math.pi)) * np.conj(M3_val) * ((W[:3, :] * bvals[None, :]) @ W[3:, :].conj().T)
    return dM, float(np.sqrt(np.min(evals))), float(np.sqrt(np.max(evals)))


def verify_U1R_invariance(Mq, Tq, mQ2, mqR2, vq, vqp, mu_val, M3_val, alpha3, mu_rg):
    """Exact U(1)_R unit test: rotating M3 -> e^{-2i a} M3, Tq -> e^{-2i a} Tq, mu -> e^{+2i a} mu
    MUST leave dM strictly invariant!
    """
    alpha = 0.37
    dM0, _, _ = compute_exact_6x6_gluino_threshold(Mq, Tq, mQ2, mqR2, vq, vqp, mu_val, M3_val, alpha3, mu_rg)
    dM1, _, _ = compute_exact_6x6_gluino_threshold(
        Mq, np.exp(-2j * alpha) * Tq, mQ2, mqR2, vq, vqp,
        np.exp(2j * alpha) * mu_val, np.exp(-2j * alpha) * M3_val, alpha3, mu_rg,
    )
    err = float(np.max(np.abs(dM1 - dM0)) / max(1e-30, float(np.max(np.abs(dM0)))))
    assert err < 1e-12, f"U(1)_R invariance violated: err={err}"
    return err


def run_exact_mssm_soft_rge_strong_cp(
    Yu_U, Yd_U, Ye_U,
    m0: float = 12000.0,
    M12: float = 1200.0,
    A0: float = 0.0,
    mu_val: float = 2000.0,
    phi_M3: float = 0.0,
    phi_A0: float = 0.0,
    nonuniv_herm_deltaA: float = 0.0,
    tanb: float = 50.0,
    M_U: float = 1.03566e16,
    M_sq_stop: float = 12024.0,
):
    """Exact coupled 3x3 1-loop MSSM RGEs (including slepton mL2, meR2 and tau-Yukawa
    contributions to dmHd2!) + exact U(1)_R-invariant 6x6 squark-gluino loop.
    """
    gU = math.sqrt(4.0 * math.pi / 25.893)
    M1_0 = complex(M12, 0.0)
    M2_0 = complex(M12, 0.0)
    M3_0 = M12 * np.exp(1j * phi_M3)

    A0_c = A0 * np.exp(1j * phi_A0)
    Tu_0 = A0_c * Yu_U.copy()
    Td_0 = A0_c * Yd_U.copy()
    Te_0 = A0_c * Ye_U.copy()
    if abs(nonuniv_herm_deltaA) > 0.0:
        H_d2 = Yd_U @ Yd_U.conj().T
        H_u2 = Yu_U @ Yu_U.conj().T
        Tu_0 = Tu_0 + nonuniv_herm_deltaA * m0 * 0.5 * (H_d2 @ Yu_U + Yu_U @ H_d2)
        Td_0 = Td_0 + nonuniv_herm_deltaA * m0 * 0.5 * (H_u2 @ Yd_U + Yd_U @ H_u2)

    I3 = np.eye(3, dtype=complex)
    mQ2_0 = (m0 ** 2) * I3.copy()
    muR2_0 = (m0 ** 2) * I3.copy()
    mdR2_0 = (m0 ** 2) * I3.copy()
    mL2_0 = (m0 ** 2) * I3.copy()
    meR2_0 = (m0 ** 2) * I3.copy()
    mHu2_0 = m0 ** 2
    mHd2_0 = m0 ** 2

    def pack(g, M, Yu, Yd, Ye, Tu, Td, Te, mQ2, muR2, mdR2, mL2, meR2, mHu2, mHd2):
        arrs = [
            np.array(g, dtype=float),
            np.array([M[0].real, M[0].imag, M[1].real, M[1].imag, M[2].real, M[2].imag], dtype=float),
        ]
        for mat in (Yu, Yd, Ye, Tu, Td, Te, mQ2, muR2, mdR2, mL2, meR2):
            arrs.append(mat.real.ravel())
            arrs.append(mat.imag.ravel())
        arrs.append(np.array([mHu2, mHd2], dtype=float))
        return np.concatenate(arrs)

    def unpack(y):
        g = y[0:3]
        M = [y[3] + 1j * y[4], y[5] + 1j * y[6], y[7] + 1j * y[8]]
        idx = 9
        mats = []
        for _ in range(11):
            re = y[idx:idx + 9].reshape(3, 3)
            im = y[idx + 9:idx + 18].reshape(3, 3)
            mats.append(re + 1j * im)
            idx += 18
        mHu2, mHd2 = y[idx], y[idx + 1]
        return g, M, mats[0], mats[1], mats[2], mats[3], mats[4], mats[5], mats[6], mats[7], mats[8], mats[9], mats[10], mHu2, mHd2

    k16 = 1.0 / (16.0 * math.pi * math.pi)
    b = (33.0 / 5.0, 1.0, -3.0)

    def rhs(_t, y):
        g, M, Yu, Yd, Ye, Tu, Td, Te, mQ2, muR2, mdR2, mL2, meR2, mHu2, mHd2 = unpack(y)
        g1, g2, g3 = g
        M1, M2, M3 = M
        dg = [k16 * b[i] * (g[i] ** 3) for i in range(3)]
        dM = [2.0 * k16 * b[i] * (g[i] ** 2) * M[i] for i in range(3)]

        Hu, Hd, He = Yu @ Yu.conj().T, Yd @ Yd.conj().T, Ye @ Ye.conj().T
        HuR, HdR, HeR = Yu.conj().T @ Yu, Yd.conj().T @ Yd, Ye.conj().T @ Ye
        tr_u = (3.0 * np.trace(Hu)).real
        tr_d = (3.0 * np.trace(Hd) + np.trace(He)).real

        dYu = k16 * (3.0 * Hu + Hd + (tr_u - 13.0 / 15.0 * g1**2 - 3.0 * g2**2 - 16.0 / 3.0 * g3**2) * I3) @ Yu
        dYd = k16 * (3.0 * Hd + Hu + (tr_d - 7.0 / 15.0 * g1**2 - 3.0 * g2**2 - 16.0 / 3.0 * g3**2) * I3) @ Yd
        dYe = k16 * (3.0 * He + (tr_d - 9.0 / 5.0 * g1**2 - 3.0 * g2**2) * I3) @ Ye

        tr_Tu = 3.0 * np.trace(Tu @ Yu.conj().T)
        tr_Td = 3.0 * np.trace(Td @ Yd.conj().T) + np.trace(Te @ Ye.conj().T)

        dTu = k16 * (
            (tr_u - 13.0 / 15.0 * g1**2 - 3.0 * g2**2 - 16.0 / 3.0 * g3**2) * Tu
            + (2.0 * tr_Tu + 26.0 / 15.0 * g1**2 * M1 + 6.0 * g2**2 * M2 + 32.0 / 3.0 * g3**2 * M3) * Yu
            + 5.0 * Hu @ Tu + 4.0 * Tu @ HuR
            + Hd @ Tu + 2.0 * (Td @ Yd.conj().T) @ Yu
        )
        dTd = k16 * (
            (tr_d - 7.0 / 15.0 * g1**2 - 3.0 * g2**2 - 16.0 / 3.0 * g3**2) * Td
            + (2.0 * tr_Td + 14.0 / 15.0 * g1**2 * M1 + 6.0 * g2**2 * M2 + 32.0 / 3.0 * g3**2 * M3) * Yd
            + 5.0 * Hd @ Td + 4.0 * Td @ HdR
            + Hu @ Td + 2.0 * (Tu @ Yu.conj().T) @ Yd
        )
        dTe = k16 * (
            (tr_d - 9.0 / 5.0 * g1**2 - 3.0 * g2**2) * Te
            + (2.0 * tr_Td + 18.0 / 5.0 * g1**2 * M1 + 6.0 * g2**2 * M2) * Ye
            + 5.0 * He @ Te + 4.0 * Te @ HeR
        )

        # Hypercharge S-term (included explicitly; vanishes at M_U for universal m0^2):
        S_hc = (
            mHu2 - mHd2
            + (np.trace(mQ2 - mL2 - 2.0 * muR2 + mdR2 + meR2)).real
        )
        dmQ2 = k16 * (
            (mQ2 @ Hu + Hu @ mQ2) + 2.0 * Yu @ muR2 @ Yu.conj().T + 2.0 * mHu2 * Hu + 2.0 * Tu @ Tu.conj().T
            + (mQ2 @ Hd + Hd @ mQ2) + 2.0 * Yd @ mdR2 @ Yd.conj().T + 2.0 * mHd2 * Hd + 2.0 * Td @ Td.conj().T
            - (2.0 / 15.0 * g1**2 * abs(M1)**2 + 6.0 * g2**2 * abs(M2)**2 + 32.0 / 3.0 * g3**2 * abs(M3)**2 - 0.2 * g1**2 * S_hc) * I3
        )
        dmuR2 = k16 * (
            2.0 * (muR2 @ HuR + HuR @ muR2) + 4.0 * Yu.conj().T @ mQ2 @ Yu + 4.0 * mHu2 * HuR + 4.0 * Tu.conj().T @ Tu
            - (32.0 / 15.0 * g1**2 * abs(M1)**2 + 32.0 / 3.0 * g3**2 * abs(M3)**2 + 0.8 * g1**2 * S_hc) * I3
        )
        dmdR2 = k16 * (
            2.0 * (mdR2 @ HdR + HdR @ mdR2) + 4.0 * Yd.conj().T @ mQ2 @ Yd + 4.0 * mHd2 * HdR + 4.0 * Td.conj().T @ Td
            - (8.0 / 15.0 * g1**2 * abs(M1)**2 + 32.0 / 3.0 * g3**2 * abs(M3)**2 - 0.4 * g1**2 * S_hc) * I3
        )
        dmL2 = k16 * (
            (mL2 @ He + He @ mL2) + 2.0 * Ye @ meR2 @ Ye.conj().T + 2.0 * mHd2 * He + 2.0 * Te @ Te.conj().T
            - (6.0 / 5.0 * g1**2 * abs(M1)**2 + 6.0 * g2**2 * abs(M2)**2 + 0.6 * g1**2 * S_hc) * I3
        )
        dmeR2 = k16 * (
            2.0 * (meR2 @ HeR + HeR @ meR2) + 4.0 * Ye.conj().T @ mL2 @ Ye + 4.0 * mHd2 * HeR + 4.0 * Te.conj().T @ Te
            - (24.0 / 5.0 * g1**2 * abs(M1)**2 - 1.2 * g1**2 * S_hc) * I3
        )
        dmHu2 = k16 * (
            6.0 * (np.trace(mQ2 @ Hu + muR2 @ HuR + mHu2 * Hu + Tu @ Tu.conj().T)).real
            - 6.0 / 5.0 * g1**2 * abs(M1)**2 - 6.0 * g2**2 * abs(M2)**2 + 0.6 * g1**2 * S_hc
        )
        # Complete dmHd2 INCLUDING the tau-slepton Yukawa trace!
        dmHd2 = k16 * (
            6.0 * (np.trace(mQ2 @ Hd + mdR2 @ HdR + mHd2 * Hd + Td @ Td.conj().T)).real
            + 2.0 * (np.trace(mL2 @ He + meR2 @ HeR + mHd2 * He + Te @ Te.conj().T)).real
            - 6.0 / 5.0 * g1**2 * abs(M1)**2 - 6.0 * g2**2 * abs(M2)**2 - 0.6 * g1**2 * S_hc
        )
        return pack(dg, dM, dYu, dYd, dYe, dTu, dTd, dTe, dmQ2, dmuR2, dmdR2, dmL2, dmeR2, dmHu2, dmHd2)

    y0 = pack([gU, gU, gU], [M1_0, M2_0, M3_0], Yu_U, Yd_U, Ye_U, Tu_0, Td_0, Te_0, mQ2_0, muR2_0, mdR2_0, mL2_0, meR2_0, mHu2_0, mHd2_0)
    sol = solve_ivp(rhs, (0.0, math.log(M_sq_stop / M_U)), y0, rtol=1e-10, atol=1e-13)
    g, M, Yu, Yd, Ye, Tu, Td, Te, mQ2, muR2, mdR2, mL2, meR2, mHu2, mHd2 = unpack(sol.y[:, -1])

    det_rg = np.linalg.det(Yu) * np.linalg.det(Yd)
    theta_rg = float(math.atan2(det_rg.imag, det_rg.real))
    tu_nonherm = float(np.linalg.norm(Tu - Tu.conj().T) / max(1e-30, np.linalg.norm(Tu)))
    td_nonherm = float(np.linalg.norm(Td - Td.conj().T) / max(1e-30, np.linalg.norm(Td)))

    v = 174.104
    sb = tanb / math.sqrt(1.0 + tanb**2)
    cb = 1.0 / math.sqrt(1.0 + tanb**2)
    vu, vd = v * sb, v * cb
    Mu_low, Md_low = Yu * vu, Yd * vd
    alpha3_low = (g[2] ** 2) / (4.0 * math.pi)

    u1r_err = verify_U1R_invariance(Md_low, Td, mQ2, mdR2, vd, vu, mu_val, M[2], alpha3_low, M_sq_stop)
    dMu, m_su_min, m_su_max = compute_exact_6x6_gluino_threshold(Mu_low, Tu, mQ2, muR2, vu, vd, mu_val, M[2], alpha3_low, M_sq_stop)
    dMd, m_sd_min, m_sd_max = compute_exact_6x6_gluino_threshold(Md_low, Td, mQ2, mdR2, vd, vu, mu_val, M[2], alpha3_low, M_sq_stop)

    det_tot = np.linalg.det(Mu_low + dMu) * np.linalg.det(Md_low + dMd)
    theta_tot = float(math.atan2(det_tot.imag, det_tot.real))
    theta_phys = float(theta_tot + 3.0 * math.atan2(M[2].imag, M[2].real))

    # Also report induced finite gluino-sbottom Delta_b at M_sq_stop:
    su_b = float(np.sort(np.linalg.svd(Md_low, compute_uv=False))[-1])
    su_b_shift = float(np.sort(np.linalg.svd(Md_low + dMd, compute_uv=False))[-1])
    delta_b_induced = (su_b_shift - su_b) / su_b

    return dict(
        alpha3_at_Msq=round(alpha3_low, 5),
        M3_at_Msq_GeV=round(abs(M[2]), 1),
        squark_up_range_GeV=[round(m_su_min, 1), round(m_su_max, 1)],
        squark_down_range_GeV=[round(m_sd_min, 1), round(m_sd_max, 1)],
        U1R_invariance_max_err=float(f"{u1r_err:.2e}"),
        induced_gluino_Delta_b=round(delta_b_induced, 5),
        Tu_RG_nonHermiticity=float(f"{tu_nonherm:.3e}"),
        Td_RG_nonHermiticity=float(f"{td_nonherm:.3e}"),
        theta_QCD_1loop_Yukawa_RG_rad=float(f"{abs(theta_rg):.3e}"),
        theta_QCD_full_6x6_gluino_loop_rad=float(f"{abs(theta_phys):.3e}"),
        neutron_EDM_bound_rad=1.3e-10,
        passes_nEDM=bool(abs(theta_phys) < 1.3e-10),
    )


def profile_scan_delta_cp_smooth(targets, tanb, seed_params, zero_mask=("a13", "h23")):
    """Smooth unrounded continuation profile scan of delta_CP^lepton across the FULL circle
    [0, 360) deg using unrounded `delta_CP_lepton_rad` (fixing the v1.1.0 round(..., 1) bug!).
    Walks smoothly in BOTH directions from 0 deg (counterclockwise 0 -> 180 and clockwise 360 -> 180)
    taking the global minimum at every checkpoint!
    """
    obs_keys = list(targets.keys())
    mt_t = targets["m_t"][0]
    all_names = ["f1", "f2", "f3", "h11", "h22", "h33", "h12", "h13", "h23", "a12", "a13", "a23", "s", "c_e", "c_u", "c_nu"]
    active = [n for n in all_names if n not in zero_mask]
    idx_map = {n: i for i, n in enumerate(active)}

    F0, H0, A0 = seed_params["F_diag"], seed_params["H"], seed_params["A_12_13_23"]
    d0 = dict(
        f1=F0[0], f2=F0[1], f3=F0[2], h11=H0[0][0], h22=H0[1][1], h33=H0[2][2],
        h12=H0[0][1], h13=H0[0][2], h23=H0[1][2], a12=A0[0], a13=A0[1], a23=A0[2],
        s=seed_params["s"], c_e=seed_params["c_e"], c_u=seed_params["c_u"], c_nu=seed_params["c_nu"],
    )
    x_seed = np.array([d0[n] for n in active], dtype=float)

    def unpack_eval(x):
        get = lambda n: float(x[idx_map[n]]) if n in idx_map else 0.0
        F = np.diag([get("f1"), get("f2"), get("f3")])
        H = np.array([[get("h11"), get("h12"), get("h13")], [get("h12"), get("h22"), get("h23")], [get("h13"), get("h23"), get("h33")]], dtype=float)
        A = np.array([[0.0, get("a12"), get("a13")], [-get("a12"), 0.0, get("a23")], [-get("a13"), -get("a23"), 0.0]], dtype=float)
        s, c_e, c_u, c_nu = get("s"), get("c_e"), get("c_u"), get("c_nu")
        su_u, _, _ = diag_hermitian(H + s * F + 1j * c_u * A)
        if su_u[2] < 1e-12:
            return None
        return eval_parity_hermitian(H, F, A, mt_t / su_u[2], s, c_e, c_u, c_nu, targets, tanb, beta_II=0.0)

    def step_to_angle(x_init, target_deg, sigma_deg=0.5):
        target_rad = math.radians(target_deg)
        def resid(x):
            ev = unpack_eval(x)
            if ev is None:
                return np.full(20, 1e3)
            pvec = [ev["pulls"][k] for k in obs_keys]
            ymax = ev["yukawa_perturbativity"]["max_Y_all"]
            pvec.append(0.0 if ymax <= 2.0 else 10.0 * (ymax - 2.0))
            # CRITICAL FIX: unrounded exact radian `delta_CP_lepton_rad`!
            d_rad = ev["preds"]["delta_CP_lepton_rad"]
            ang_diff = math.atan2(math.sin(d_rad - target_rad), math.cos(d_rad - target_rad))
            pvec.append(math.degrees(ang_diff) / sigma_deg)
            return np.array(pvec, dtype=float)
        res = least_squares(resid, x_init, method="trf", x_scale="jac", ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=2500)
        return res.x, unpack_eval(res.x)

    # Smooth continuation in 15-deg steps in BOTH directions from 0/360 deg:
    best_by_deg = {}
    checkpoints = {0, 45, 90, 135, 180, 225, 270, 300, 315}

    for direction, angle_seq in [
        ("cw", list(range(360, 165, -15))),
        ("ccw", list(range(0, 195, 15))),
    ]:
        x_cur = x_seed.copy()
        for deg in angle_seq:
            deg_mod = deg % 360
            x_cur, ev = step_to_angle(x_cur, deg_mod, sigma_deg=0.5)
            if deg_mod in checkpoints and ev is not None and ev["yukawa_perturbativity"]["max_Y_all"] <= 2.05:
                d_deg = math.degrees(ev["preds"]["delta_CP_lepton_rad"]) % 360.0
                ang_err = abs(((d_deg - deg_mod + 180.0) % 360.0) - 180.0)
                if ang_err < 3.0:
                    if deg_mod not in best_by_deg or ev["chi2"] < best_by_deg[deg_mod]["chi2_18obs"]:
                        best_by_deg[deg_mod] = dict(
                            target_deg=deg_mod,
                            achieved_deg=round(d_deg, 2),
                            chi2_18obs=round(ev["chi2"], 4),
                            max_abs_pull=round(ev["max_abs_pull"], 3),
                            sin2_th23=ev["preds"]["sin2_th23_lepton"],
                            sum_m_nu_eV=ev["preds"]["sum_m_nu_eV"],
                            m_bb_meV=ev["preds"]["m_beta_beta_meV"],
                            max_Y=float(ev["yukawa_perturbativity"]["max_Y_all"]),
                            branch=direction,
                        )

    return [best_by_deg.get(d, dict(target_deg=d, status="no_solution_found")) for d in sorted(checkpoints)]


def main():
    t0 = time.time()
    with open(P30_DATA / "paper30_master_verification_results.json") as f:
        p30 = json.load(f)

    ps = p30["flavor"]["SUSY_PH16_a13_h23"]["params"]
    v = 174.104
    tanb = 50.0
    sb, cb = tanb / math.sqrt(1.0 + tanb**2), 1.0 / math.sqrt(1.0 + tanb**2)
    vu, vd = v * sb, v * cb

    F = np.diag(ps["F_diag"]).astype(complex)
    H = np.array(ps["H"], dtype=complex)
    A = np.array([
        [0.0, ps["A_12_13_23"][0], ps["A_12_13_23"][1]],
        [-ps["A_12_13_23"][0], 0.0, ps["A_12_13_23"][2]],
        [-ps["A_12_13_23"][1], -ps["A_12_13_23"][2], 0.0],
    ], dtype=complex)
    r, s, c_e, c_u = ps["r"], ps["s"], ps["c_e"], ps["c_u"]
    Yu_U = r * (H + s * F + 1j * c_u * A) / vu
    Yd_U = (H + F + 1j * A) / vd
    Ye_U = (H - 3.0 * F + 1j * c_e * A) / vd

    benchmarks_cp = {
        "B1_Parity_Universal_A0_0": run_exact_mssm_soft_rge_strong_cp(Yu_U, Yd_U, Ye_U, m0=12000.0, M12=1200.0, A0=0.0),
        "B2_Parity_Universal_A0_12TeV": run_exact_mssm_soft_rge_strong_cp(Yu_U, Yd_U, Ye_U, m0=12000.0, M12=1200.0, A0=12000.0),
        "B3_Parity_NonUniversal_Hermitian_dA0p5": run_exact_mssm_soft_rge_strong_cp(Yu_U, Yd_U, Ye_U, m0=12000.0, M12=1200.0, A0=12000.0, nonuniv_herm_deltaA=0.5),
        "B4_SoftCP_Violation_phiA_1e-6": run_exact_mssm_soft_rge_strong_cp(Yu_U, Yd_U, Ye_U, m0=12000.0, M12=1200.0, A0=12000.0, phi_A0=1e-6),
        "B5_SoftCP_Violation_phiM3_1e-10": run_exact_mssm_soft_rge_strong_cp(Yu_U, Yd_U, Ye_U, m0=12000.0, M12=1200.0, A0=0.0, phi_M3=1e-10),
    }
    for k, vres in benchmarks_cp.items():
        sys.stderr.write(
            f"[Bridge 1A {k}] theta_RG={vres['theta_QCD_1loop_Yukawa_RG_rad']:.2e}, "
            f"theta_6x6={vres['theta_QCD_full_6x6_gluino_loop_rad']:.2e}, "
            f"U1R_err={vres['U1R_invariance_max_err']:.1e}, Db={vres['induced_gluino_Delta_b']}, pass={vres['passes_nEDM']}\n"
        )

    targs_S1, _ = run_mssm_targets(delta_b=0.0, tanb=50.0)
    prof_susy = profile_scan_delta_cp_smooth(targs_S1, 50.0, ps, zero_mask=("a13", "h23"))
    for row in prof_susy:
        sys.stderr.write(f"[Bridge 1B Smooth dCP={row['target_deg']:3d} deg] {row}\n")

    out = dict(
        bridge1A_exact_1loop_SUSY_strong_CP=benchmarks_cp,
        bridge1B_SUSY_PH16_smooth_delta_CP_profile=prof_susy,
        elapsed_s=round(time.time() - t0, 2),
    )
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
