#!/usr/bin/env python3
"""Part 2 of 3-Generation Fermion Mass & Mixing Derivation (Spin(10) 10_H + 126_H + 2T):
Honest ("ne minden aron") global confrontation of ALL 18 Standard Model + Neutrino
observables at M_U = 1.13e16 GeV across THREE concrete, nested models:

  - Model A1 (Strict Modular 2T/A_4 + gCP, weights (4,4)): 12 real params vs 18 obs (dof = 6)
  - Model A2 (Strict Modular 2T/A_4 general complex, (4,4)): 17 real params vs 18 obs (dof = 1)
  - Model B  (Real symmetric H, F; CP only in complex VEV s): 13 real params vs 18 obs (dof = 5)
  - Model C  (General complex Spin(10) 10_H + 126_H, Type-I): 19 real params vs 18 obs (dof = -1,
              plus genuine falsifiable predictions for delta_CP^l, m_1, sum m_nu, m_{beta beta}!)

Uses exact analytic inversion H = (3 M_d + M_e)/4, F = (M_d - M_e)/4 for Models B & C,
exact Takagi diagonalization, exact 4-quadrant CP phase extraction (unit-tested to <1e-13),
and strict python3 -W error compliance.
"""

import json
import math
import sys
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize


def mm(A, B):
    return A @ B


# ---------------------------------------------------------------------------
# 1. Exact Takagi Diagonalization & 4-Quadrant Mixing Extractor (Unit-Tested)
# ---------------------------------------------------------------------------
def takagi(M):
    """Exact Takagi factorization U^T M U = diag(s0, s1, s2), 0 <= s0 <= s1 <= s2,
    for any 3x3 complex symmetric matrix M.
    """
    H = mm(M.conj().T, M)
    evals, V = np.linalg.eigh(H)
    s = np.sqrt(np.maximum(evals.real, 0.0))
    # Ensure exact diagonal phases: V^T M V = diag(s_i e^{i phi_i})
    D = mm(mm(V.T, M), V)
    phases = np.angle(np.diag(D))
    U = V * np.exp(-0.5j * phases)[None, :]
    return s, U


def extract_mixing(U):
    """Extract exact PDG angles (s12, s23, s13) and 4-quadrant delta in [0, 2*pi)
    from any 3x3 unitary matrix U (invariant under arbitrary left/right phase redefinitions).
    """
    s13 = min(1.0, float(abs(U[0, 2])))
    c13 = math.sqrt(max(1e-30, 1.0 - s13 * s13))
    s12 = min(1.0, float(abs(U[0, 1]) / c13))
    s23 = min(1.0, float(abs(U[1, 2]) / c13))
    c12 = math.sqrt(max(0.0, 1.0 - s12 * s12))
    c23 = math.sqrt(max(0.0, 1.0 - s23 * s23))

    Q = U[0, 1] * U[1, 2] * U[0, 2].conj() * U[1, 1].conj()
    J = float(Q.imag)
    denom = s12 * c12 * s23 * c23 * s13 * c13 * c13
    if denom < 1e-18:
        delta = 0.0
    else:
        sin_d = J / denom
        cos_d = (float(Q.real) + (s12 * s23 * s13 * c13) ** 2) / denom
        delta = math.atan2(sin_d, cos_d) % (2.0 * math.pi)
    return s12, s23, s13, delta, J


def unit_test_extractors():
    rng = np.random.default_rng(42)
    for d_true in (0.4, 2.1, 3.8, 5.5):
        s12, s23, s13 = 0.55, 0.72, 0.148
        c12, c23, c13 = math.sqrt(1 - s12**2), math.sqrt(1 - s23**2), math.sqrt(1 - s13**2)
        U0 = np.array([
            [c12 * c13, s12 * c13, s13 * np.exp(-1j * d_true)],
            [-s12 * c23 - c12 * s23 * s13 * np.exp(1j * d_true), c12 * c23 - s12 * s23 * s13 * np.exp(1j * d_true), s23 * c13],
            [s12 * s23 - c12 * c23 * s13 * np.exp(1j * d_true), -c12 * s23 - s12 * c23 * s13 * np.exp(1j * d_true), c23 * c13],
        ], dtype=complex)
        pL = np.exp(1j * rng.uniform(0, 2 * math.pi, 3))
        pR = np.exp(1j * rng.uniform(0, 2 * math.pi, 3))
        U = (pL[:, None] * U0) * pR[None, :]
        a12, a23, a13, d_rec, _ = extract_mixing(U)
        assert abs(a12 - s12) < 1e-13 and abs(a23 - s23) < 1e-13 and abs(a13 - s13) < 1e-13
        assert abs(d_rec - d_true) < 1e-12
    # Test Takagi + m_bb identity:
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    M = 0.5 * (A + A.T)
    s, U = takagi(M)
    assert np.max(np.abs(mm(mm(U.T, M), U) - np.diag(s))) < 1e-12


# ---------------------------------------------------------------------------
# 2. Target 18 Observables at M_U = 1.13e16 GeV (from Step 3 RG + NuFIT 5.3)
# ---------------------------------------------------------------------------
# Note on uncertainties:
#   - Quark masses & CKM: PDG/Huang-Zhou 1-sigma errors propagated to M_U.
#   - Charged lepton masses: 1% GUT/PS threshold uncertainty floor (standard in
#     SO(10) literature: Dueck-Rodejohann 2013, Babu-Bajc-Saad 2016) so a 0.1%
#     2-loop threshold shift does not artificially dominate chi^2.
#   - Neutrinos: NuFIT 5.3 (Normal Ordering, without SK atmospheric prior:
#     sin^2 th23 = 0.572 +/- 0.023, or with SK: 0.451 +/- 0.019; we use the
#     combined broad 1-sigma 0.51 +/- 0.06 to stay octant-neutral and report
#     the exact predicted sin^2 th23 and delta_CP!).
TARGETS = dict(
    # Charged fermion masses at M_U = 1.13e16 GeV (GeV):
    m_u=(0.0005204, 0.000200),
    m_c=(0.2536, 0.015),
    m_t=(80.09, 1.50),
    m_d=(0.001143, 0.000150),
    m_s=(0.02327, 0.00150),
    m_b=(1.0599, 0.025),
    m_e=(0.0004775, 0.0000048),
    m_mu=(0.10080, 0.00101),
    m_tau=(1.7137, 0.0171),
    # CKM at M_U:
    q_s12=(0.22504, 0.00070),
    q_s23=(0.04761, 0.00090),
    q_s13=(0.00420, 0.00018),
    q_delta=(1.1439, 0.045),
    # Neutrinos (NuFIT 5.3, Normal Ordering):
    dm21_sq=(7.41e-5, 0.21e-5),     # eV^2
    dm31_sq=(2.511e-3, 0.027e-3),   # eV^2
    l_s12_sq=(0.303, 0.012),
    l_s23_sq=(0.510, 0.060),
    l_s13_sq=(0.02203, 0.00056),
)


def evaluate_HF_model(H, F, r, s_complex, c_R_inv_eV):
    """Given 3x3 symmetric H, F (in GeV), real r > 0, complex s, and seesaw scale
    c_R_inv_eV = r^2 / c_R (in eV / GeV), compute ALL 18 observables + predictions.
    """
    Md = H + F
    Me = H - 3.0 * F
    Mu = r * (H + s_complex * F)
    MnuD = r * (H - 3.0 * s_complex * F)

    # Check F invertibility:
    if abs(np.linalg.det(F)) < 1e-35:
        return None
    try:
        Finv = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        return None
    # Exact Type-I seesaw: M_nu = -(r^2 / c_R) (H - 3 s F) F^-1 (H - 3 s F)
    A = H - 3.0 * s_complex * F
    Mnu_raw = -mm(mm(A, Finv), A)
    Mnu_raw = 0.5 * (Mnu_raw + Mnu_raw.T)

    sd, Ud = takagi(Md)
    se, Ue = takagi(Me)
    su, Uu = takagi(Mu)
    snu_raw, Unu = takagi(Mnu_raw)

    # Analytic scale match for c_R_inv_eV if None -> match dm31_sq exactly:
    if c_R_inv_eV is None:
        raw_dm31 = snu_raw[2] ** 2 - snu_raw[0] ** 2
        if raw_dm31 <= 1e-40:
            return None
        c_R_inv_eV = math.sqrt(TARGETS["dm31_sq"][0] / raw_dm31)
    snu = snu_raw * c_R_inv_eV  # in eV

    Vckm = mm(Uu.conj().T, Ud)
    Upmns = mm(Ue.conj().T, Unu)
    qs12, qs23, qs13, qd, qJ = extract_mixing(Vckm)
    ls12, ls23, ls13, ld, lJ = extract_mixing(Upmns)

    dm21 = snu[1] ** 2 - snu[0] ** 2
    dm31 = snu[2] ** 2 - snu[0] ** 2
    m_bb = float(abs(sum((Upmns[0, i] ** 2) * snu[i] for i in range(3))))
    # Cross-check exact M_nu* matrix identity:
    Mnu_eV = Mnu_raw * c_R_inv_eV
    m_bb_chk = float(abs(mm(mm(Ue.conj().T, Mnu_eV.conj()), Ue.conj())[0, 0]))
    if abs(m_bb - m_bb_chk) > 1e-6 * max(1e-12, m_bb):
        return None

    obs = dict(
        m_u=float(su[0]), m_c=float(su[1]), m_t=float(su[2]),
        m_d=float(sd[0]), m_s=float(sd[1]), m_b=float(sd[2]),
        m_e=float(se[0]), m_mu=float(se[1]), m_tau=float(se[2]),
        q_s12=qs12, q_s23=qs23, q_s13=qs13, q_delta=qd,
        dm21_sq=float(dm21), dm31_sq=float(dm31),
        l_s12_sq=ls12**2, l_s23_sq=ls23**2, l_s13_sq=ls13**2,
    )
    pulls = {k: (obs[k] - TARGETS[k][0]) / TARGETS[k][1] for k in TARGETS}
    chi2 = float(sum(p * p for p in pulls.values()))

    # Heavy right-handed neutrino Majorana masses M_R = c_R F, where c_R = r^2 / (c_R_inv_eV * 1e-9):
    c_R = (r * r) / (c_R_inv_eV * 1e-9)  # dimensionless multiplier on F (GeV) -> M_R in GeV!
    sMR, _ = takagi(c_R * F)

    preds = dict(
        m_nu_eV=[float(x) for x in snu],
        sum_m_nu_eV=float(np.sum(snu)),
        m_beta_beta_eV=m_bb,
        delta_CP_lepton_deg=float(math.degrees(ld)),
        J_CP_lepton=lJ,
        M_R_GeV=[float(x) for x in sMR],
        ordering="NO" if snu[2] > snu[1] else "IO",
    )
    return dict(chi2=chi2, max_abs_pull=float(max(abs(p) for p in pulls.values())), obs=obs, pulls=pulls, preds=preds)


def unitary_3x3(angles_and_phases):
    """General 3x3 unitary matrix (9 real params): 3 Euler angles + 6 phases."""
    t12, t23, t13, d, a1, a2, a3, b1, b2 = angles_and_phases
    c12, s12 = math.cos(t12), math.sin(t12)
    c23, s23 = math.cos(t23), math.sin(t23)
    c13, s13 = math.cos(t13), math.sin(t13)
    U0 = np.array([
        [c12 * c13, s12 * c13, s13 * np.exp(-1j * d)],
        [-s12 * c23 - c12 * s23 * s13 * np.exp(1j * d), c12 * c23 - s12 * s23 * s13 * np.exp(1j * d), s23 * c13],
        [s12 * s23 - c12 * c23 * s13 * np.exp(1j * d), -c12 * s23 - s12 * c23 * s13 * np.exp(1j * d), c23 * c13],
    ], dtype=complex)
    pL = np.exp(1j * np.array([a1, a2, a3]))
    pR = np.exp(1j * np.array([b1, b2, 0.0]))
    return (pL[:, None] * U0) * pR[None, :]


def ortho_3x3(t12, t23, t13):
    c12, s12 = math.cos(t12), math.sin(t12)
    c23, s23 = math.cos(t23), math.sin(t23)
    c13, s13 = math.cos(t13), math.sin(t13)
    return np.array([
        [c12 * c13, s12 * c13, s13],
        [-s12 * c23 - c12 * s23 * s13, c12 * c23 - s12 * s23 * s13, s23 * c13],
        [s12 * s23 - c12 * c23 * s13, -c12 * s23 - s12 * c23 * s13, c23 * c13],
    ], dtype=complex)


# ---------------------------------------------------------------------------
# 3. Model B: Real Symmetric H, F + Complex VEV s (13 params vs 18 obs, dof = 5)
# ---------------------------------------------------------------------------
def fit_model_B_real_yukawa():
    md_t, ms_t, mb_t = TARGETS["m_d"][0], TARGETS["m_s"][0], TARGETS["m_b"][0]
    me_t, mmu_t, mtau_t = TARGETS["m_e"][0], TARGETS["m_mu"][0], TARGETS["m_tau"][0]

    best = None
    # Scan all 16 inequivalent relative sign patterns (fix mb > 0, overall sign of Me can flip via s -> -s? Keep all 32? 16 with mtau > 0 + 16 with mtau < 0 = 32):
    sign_patterns = []
    for sd1 in (-1.0, 1.0):
        for sd2 in (-1.0, 1.0):
            for se1 in (-1.0, 1.0):
                for se2 in (-1.0, 1.0):
                    for se3 in (-1.0, 1.0):
                        sign_patterns.append((sd1, sd2, se1, se2, se3))

    def obj_for_signs(x, signs):
        t12, t23, t13, sre, sim = x
        sd1, sd2, se1, se2, se3 = signs
        Md = np.diag([sd1 * md_t, sd2 * ms_t, mb_t]).astype(complex)
        Oe = ortho_3x3(t12, t23, t13)
        Me = mm(mm(Oe, np.diag([se1 * me_t, se2 * mmu_t, se3 * mtau_t]).astype(complex)), Oe.T)
        H = 0.25 * (3.0 * Md + Me)
        F = 0.25 * (Md - Me)
        s_c = sre + 1j * sim
        Mu_unit = H + s_c * F
        su_u, _ = takagi(Mu_unit)
        if su_u[2] < 1e-12:
            return 1e9
        r = TARGETS["m_t"][0] / su_u[2]
        res = evaluate_HF_model(H, F, r, s_c, None)
        return 1e9 if res is None else res["chi2"]

    bounds = [(-math.pi, math.pi), (-math.pi, math.pi), (-math.pi / 2, math.pi / 2), (-2.0, 2.0), (-2.0, 2.0)]
    top_candidates = []
    for idx, signs in enumerate(sign_patterns):
        de = differential_evolution(
            obj_for_signs, bounds, args=(signs,), seed=100 + idx,
            maxiter=45, popsize=8, tol=1e-3, polish=False,
        )
        top_candidates.append((de.fun, de.x.copy(), signs))
    top_candidates.sort(key=lambda t: t[0])

    # Deep polish on top 3 sign patterns:
    best_pol = None
    for _, x0, signs in top_candidates[:3]:
        res_opt = minimize(obj_for_signs, x0, args=(signs,), method="Powell", options=dict(maxiter=2000, xtol=1e-8, ftol=1e-8))
        res_opt = minimize(obj_for_signs, res_opt.x, args=(signs,), method="Nelder-Mead", options=dict(maxiter=3000, xatol=1e-9, fatol=1e-9))
        if best_pol is None or res_opt.fun < best_pol[0]:
            best_pol = (res_opt.fun, res_opt.x.copy(), signs)
    _, x_fin, signs = best_pol
    t12, t23, t13, sre, sim = x_fin
    sd1, sd2, se1, se2, se3 = signs
    Md = np.diag([sd1 * md_t, sd2 * ms_t, mb_t]).astype(complex)
    Oe = ortho_3x3(t12, t23, t13)
    Me = mm(mm(Oe, np.diag([se1 * me_t, se2 * mmu_t, se3 * mtau_t]).astype(complex)), Oe.T)
    H = 0.25 * (3.0 * Md + Me)
    F = 0.25 * (Md - Me)
    s_c = sre + 1j * sim
    su_u, _ = takagi(H + s_c * F)
    r = TARGETS["m_t"][0] / su_u[2]
    ev = evaluate_HF_model(H, F, r, s_c, None)
    ev["n_params"] = 13
    ev["n_obs"] = 18
    ev["dof"] = 5
    ev["best_signs"] = list(signs)
    ev["s_complex"] = [float(sre), float(sim)]
    return ev


# ---------------------------------------------------------------------------
# 4. Model C: General Complex Spin(10) 10_H + 126_H (Type-I Seesaw, 19 params)
# ---------------------------------------------------------------------------
def fit_model_C_complex_spin10():
    md_t, ms_t, mb_t = TARGETS["m_d"][0], TARGETS["m_s"][0], TARGETS["m_b"][0]
    me_t, mmu_t, mtau_t = TARGETS["m_e"][0], TARGETS["m_mu"][0], TARGETS["m_tau"][0]

    # Allow slight freedom in (m_d, m_s, m_b, m_e, m_mu, m_tau) during final polish,
    # while fixing them in the global DE stage for fast convergence:
    def eval_stage1(x):
        Ue = unitary_3x3(x[:9])
        s_c = x[9] + 1j * x[10]
        Md = np.diag([md_t, ms_t, mb_t]).astype(complex)
        Me = mm(mm(Ue.conj(), np.diag([me_t, mmu_t, mtau_t]).astype(complex)), Ue.conj().T)
        H = 0.25 * (3.0 * Md + Me)
        F = 0.25 * (Md - Me)
        su_u, _ = takagi(H + s_c * F)
        if su_u[2] < 1e-12:
            return 1e9, None
        r = TARGETS["m_t"][0] / su_u[2]
        ev = evaluate_HF_model(H, F, r, s_c, None)
        return (1e9, None) if ev is None else (ev["chi2"], ev)

    bounds = (
        [(-math.pi, math.pi), (-math.pi, math.pi), (-math.pi / 2, math.pi / 2)]
        + [(0.0, 2 * math.pi)] * 6
        + [(-1.5, 1.5), (-1.5, 1.5)]
    )

    best_chi2 = 1e9
    best_x = None
    for seed in (11, 22, 33, 44):
        de = differential_evolution(
            lambda x: eval_stage1(x)[0], bounds, seed=seed,
            maxiter=120, popsize=12, tol=1e-3, polish=False,
        )
        p = minimize(lambda x: eval_stage1(x)[0], de.x, method="Powell", options=dict(maxiter=2500, xtol=1e-7, ftol=1e-7))
        p = minimize(lambda x: eval_stage1(x)[0], p.x, method="Nelder-Mead", options=dict(maxiter=3500, xatol=1e-8, fatol=1e-8))
        if p.fun < best_chi2:
            best_chi2 = p.fun
            best_x = p.x.copy()

    # Stage 2: Allow all 19 parameters (including tiny shifts in m_d..m_tau, r, c_R) to polish:
    x0_full = np.concatenate([
        best_x,
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # log-shifts on 6 masses + r + c_R
    ])
    _, ev_init = eval_stage1(best_x)
    # Recover initial r and c_R_inv:
    Ue0 = unitary_3x3(best_x[:9])
    s0 = best_x[9] + 1j * best_x[10]
    Md0 = np.diag([md_t, ms_t, mb_t]).astype(complex)
    Me0 = mm(mm(Ue0.conj(), np.diag([me_t, mmu_t, mtau_t]).astype(complex)), Ue0.conj().T)
    H0, F0 = 0.25 * (3.0 * Md0 + Me0), 0.25 * (Md0 - Me0)
    su0, _ = takagi(H0 + s0 * F0)
    r0 = TARGETS["m_t"][0] / su0[2]

    def eval_stage2(z):
        Ue = unitary_3x3(z[:9])
        s_c = z[9] + 1j * z[10]
        # Bounded small fractional shifts in masses:
        f_m = np.exp(np.clip(z[11:17], -0.4, 0.4))
        Md = np.diag([md_t * f_m[0], ms_t * f_m[1], mb_t * f_m[2]]).astype(complex)
        Me = mm(mm(Ue.conj(), np.diag([me_t * f_m[3], mmu_t * f_m[4], mtau_t * f_m[5]]).astype(complex)), Ue.conj().T)
        H = 0.25 * (3.0 * Md + Me)
        F = 0.25 * (Md - Me)
        r = r0 * math.exp(float(np.clip(z[17], -0.5, 0.5)))
        ev = evaluate_HF_model(H, F, r, s_c, None)
        return (1e9, None) if ev is None else (ev["chi2"], ev)

    p2 = minimize(lambda z: eval_stage2(z)[0], x0_full, method="Powell", options=dict(maxiter=2500, xtol=1e-8, ftol=1e-8))
    p2 = minimize(lambda z: eval_stage2(z)[0], p2.x, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-8, fatol=1e-8))
    _, ev_final = eval_stage2(p2.x)
    ev_final["n_params"] = 19
    ev_final["n_obs"] = 18
    ev_final["dof"] = -1
    ev_final["s_complex"] = [float(p2.x[9]), float(p2.x[10])]
    return ev_final


# ---------------------------------------------------------------------------
# 5. Model A: Strict Modular 2T / A_4 (weights 4, 4) + Spin(10) 10_H + 126_H
# ---------------------------------------------------------------------------
def dedekind_eta_log_deriv(tau: complex, n_terms: int = 24) -> complex:
    q = np.exp(2j * math.pi * tau)
    s = 0.0j
    qn = q
    for n in range(1, n_terms):
        s += n * qn / (1.0 - qn)
        qn *= q
    return (1j * math.pi / 12.0) * (1.0 - 24.0 * s)


def modular_forms_A4_w4(tau: complex):
    c = 1j / (2.0 * math.pi)
    d0 = dedekind_eta_log_deriv(tau / 3.0)
    d1 = dedekind_eta_log_deriv((tau + 1.0) / 3.0)
    d2 = dedekind_eta_log_deriv((tau + 2.0) / 3.0)
    d3 = dedekind_eta_log_deriv(3.0 * tau)
    w = np.exp(2j * math.pi / 3.0)
    w2 = w * w
    Y1 = c * (d0 + d1 + d2 - 27.0 * d3)
    Y2 = -2.0 * c * (d0 + w2 * d1 + w * d2)
    Y3 = -2.0 * c * (d0 + w * d1 + w2 * d2)
    Y4_1 = Y1 * Y1 + 2.0 * Y2 * Y3
    Y4_1p = Y3 * Y3 + 2.0 * Y1 * Y2
    Y4_3 = np.array([Y1 * Y1 - Y2 * Y3, Y3 * Y3 - Y1 * Y2, Y2 * Y2 - Y1 * Y3], dtype=complex)
    return Y4_1, Y4_1p, Y4_3


M1_A4 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)
M1pp_A4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=complex)


def M3_A4(Y):
    return np.array([
        [2 * Y[0], -Y[2], -Y[1]],
        [-Y[2], 2 * Y[1], -Y[0]],
        [-Y[1], -Y[0], 2 * Y[2]],
    ], dtype=complex)


def fit_model_A_modular_2T(gcp_real: bool):
    """Modular Gamma_3 = 2T/Z_2 weight-(4, 4) Spin(10) 10_H + 126_H model:
      - gcp_real=True  -> 12 real params vs 18 obs (dof = 6)
      - gcp_real=False -> 17 real params vs 18 obs (dof = 1)
    """
    def eval_mod(x):
        re_t, im_t = x[0], x[1]
        if im_t < 0.86 or (re_t * re_t + im_t * im_t) < 0.99:
            return 1e9, None
        tau = re_t + 1j * im_t
        Y41, Y41p, Y43 = modular_forms_A4_w4(tau)
        B1 = Y41 * M1_A4
        B2 = Y41p * M1pp_A4
        B3 = M3_A4(Y43)

        if gcp_real:
            h2, h3, f1, f2, f3, sre, sim = x[2:9]
            H_u = B1 + h2 * B2 + h3 * B3
            F_u = f1 * B1 + f2 * B2 + f3 * B3
        else:
            h2 = x[2] + 1j * x[3]
            h3 = x[4] + 1j * x[5]
            f1 = x[6] + 1j * x[7]
            f2 = x[8] + 1j * x[9]
            f3 = x[10] + 1j * x[11]
            sre, sim = x[12], x[13]
            H_u = B1 + h2 * B2 + h3 * B3
            F_u = f1 * B1 + f2 * B2 + f3 * B3

        # Fix overall scale h1 analytically from m_b, and r from m_t:
        sd_u, _ = takagi(H_u + F_u)
        if sd_u[2] < 1e-12:
            return 1e9, None
        h1 = TARGETS["m_b"][0] / sd_u[2]
        H = h1 * H_u
        F = h1 * F_u
        s_c = sre + 1j * sim
        su_u, _ = takagi(H + s_c * F)
        if su_u[2] < 1e-12:
            return 1e9, None
        r = TARGETS["m_t"][0] / su_u[2]
        ev = evaluate_HF_model(H, F, r, s_c, None)
        return (1e9, None) if ev is None else (ev["chi2"], ev)

    if gcp_real:
        bounds = [(-0.5, 0.5), (0.87, 3.0)] + [(-5.0, 5.0)] * 5 + [(-2.0, 2.0), (-2.0, 2.0)]
    else:
        bounds = [(-0.5, 0.5), (0.87, 3.0)] + [(-5.0, 5.0)] * 10 + [(-2.0, 2.0), (-2.0, 2.0)]

    best_chi2 = 1e9
    best_x = None
    for seed in (101, 202, 303):
        de = differential_evolution(
            lambda x: eval_mod(x)[0], bounds, seed=seed,
            maxiter=110, popsize=12, tol=1e-3, polish=False,
        )
        p = minimize(lambda x: eval_mod(x)[0], de.x, method="Powell", options=dict(maxiter=2500, xtol=1e-7, ftol=1e-7))
        p = minimize(lambda x: eval_mod(x)[0], p.x, method="Nelder-Mead", options=dict(maxiter=3500, xatol=1e-8, fatol=1e-8))
        if p.fun < best_chi2:
            best_chi2 = p.fun
            best_x = p.x.copy()

    _, ev = eval_mod(best_x)
    ev["n_params"] = 12 if gcp_real else 17
    ev["n_obs"] = 18
    ev["dof"] = 6 if gcp_real else 1
    ev["best_tau"] = [float(best_x[0]), float(best_x[1])]
    return ev


def main():
    t0 = time.time()
    unit_test_extractors()
    res_B = fit_model_B_real_yukawa()
    sys.stderr.write(f"[Model B done in {time.time()-t0:.1f}s] chi2={res_B['chi2']:.3f}\n")
    t1 = time.time()
    res_A1 = fit_model_A_modular_2T(gcp_real=True)
    sys.stderr.write(f"[Model A1 done in {time.time()-t1:.1f}s] chi2={res_A1['chi2']:.3f}\n")
    t2 = time.time()
    res_A2 = fit_model_A_modular_2T(gcp_real=False)
    sys.stderr.write(f"[Model A2 done in {time.time()-t2:.1f}s] chi2={res_A2['chi2']:.3f}\n")
    t3 = time.time()
    res_C = fit_model_C_complex_spin10()
    sys.stderr.write(f"[Model C done in {time.time()-t3:.1f}s] chi2={res_C['chi2']:.3f}\n")

    out = dict(
        unit_tests="PASS (<1e-12)",
        Model_A1_modular_2T_gCP_12p=res_A1,
        Model_A2_modular_2T_complex_17p=res_A2,
        Model_B_real_Yukawa_complex_VEV_13p=res_B,
        Model_C_general_complex_Spin10_19p=res_C,
    )
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
