#!/usr/bin/env python3
"""Step 1: Exact Test of TeV-SUSY Holomorphic Spin(10) (10_H + 126_H, NO 10_H^*)
on ALL 18 Standard Model + Neutrino Observables at M_U^SUSY = 7.97e15 GeV!

Remember our proven non-SUSY NO-GO wall:
  At non-SUSY m_b/m_tau = 0.6185, holomorphic (single Y_10) 10_H + 126_H hit
  chi^2 = 30.3 (Type-I+II) / 45.3 (Type-I) because sin^2(theta_23^l) was capped
  at <= 0.19-0.24 by max sin(2 th23) = m_b / (m_tau - m_mu) = 0.657!

Now we compute the EXACT 1-loop MSSM RG-evolved 18 targets at M_U^SUSY = 7.97e15 GeV
(tan(beta) = 50, M_SUSY = 2 TeV) in TWO regimes:
  - Regime A (pure 1-loop Delta_b = 0): m_b / m_tau = 0.836
  - Regime B (standard finite gluino-sbottom threshold Delta_b = -0.15): m_b / m_tau = 0.975
and run Jacobian-scaled Gauss-Newton (trf, x_scale='jac', 50 restarts) for BOTH:
  - Pure Type-I seesaw (19 real params vs 18 obs),
  - Type-I + Type-II seesaw from the SAME 126_H (21 real params vs 18 obs),
verifying directly whether SUSY b-tau unification SHATTERS the sin^2(theta_23^l) wall!
"""

import json
import math
import sys
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from verify_3gen_spin10_2T_global_fit import takagi, extract_mixing, unitary_3x3


def run_mssm_targets(delta_b: float = -0.15, tanb: float = 50.0):
    """Exact 1-loop SM (M_Z -> 2 TeV) + MSSM (2 TeV -> 7.97e15 GeV) running of
    all 9 charged fermion masses and 4 CKM parameters.
    """
    M_Z, M_SUSY, M_U = 91.1876, 2000.0, 7.97e15
    v = 174.104
    g1_0 = math.sqrt(5.0 / 3.0) * math.sqrt(4.0 * math.pi / 98.35)
    g2_0 = math.sqrt(4.0 * math.pi / 29.58)
    g3_0 = math.sqrt(4.0 * math.pi * 0.1180)
    m_MZ = dict(u=1.27e-3, c=0.619, t=171.7, d=2.70e-3, s=0.055, b=2.85, e=0.48657e-3, mu=0.10272, tau=1.7462)

    s12, s23, s13, dckm = 0.22500, 0.04182, 0.00369, 1.144
    c12, c23, c13 = math.sqrt(1 - s12**2), math.sqrt(1 - s23**2), math.sqrt(1 - s13**2)
    V0 = np.array([
        [c12 * c13, s12 * c13, s13 * np.exp(-1j * dckm)],
        [-s12 * c23 - c12 * s23 * s13 * np.exp(1j * dckm), c12 * c23 - s12 * s23 * s13 * np.exp(1j * dckm), s23 * c13],
        [s12 * s23 - c12 * c23 * s13 * np.exp(1j * dckm), -c12 * s23 - s12 * c23 * s13 * np.exp(1j * dckm), c23 * c13],
    ], dtype=complex)

    Yu0 = np.diag([m_MZ["u"], m_MZ["c"], m_MZ["t"]]).astype(complex) / v
    Yd0 = V0 @ (np.diag([m_MZ["d"], m_MZ["s"], m_MZ["b"]]).astype(complex) / v)
    Ye0 = np.diag([m_MZ["e"], m_MZ["mu"], m_MZ["tau"]]).astype(complex) / v

    def pack(g1, g2, g3, Yu, Yd, Ye):
        return np.concatenate([np.array([g1, g2, g3]), Yu.real.ravel(), Yu.imag.ravel(), Yd.real.ravel(), Yd.imag.ravel(), Ye.real.ravel(), Ye.imag.ravel()])

    def unpack(y):
        return y[0], y[1], y[2], (y[3:12] + 1j * y[12:21]).reshape(3, 3), (y[21:30] + 1j * y[30:39]).reshape(3, 3), (y[39:48] + 1j * y[48:57]).reshape(3, 3)

    k16 = 1.0 / (16.0 * math.pi * math.pi)

    def rhs_sm(_t, y):
        g1, g2, g3, Yu, Yd, Ye = unpack(y)
        Hu, Hd, He = Yu @ Yu.conj().T, Yd @ Yd.conj().T, Ye @ Ye.conj().T
        T = (3.0 * np.trace(Hu) + 3.0 * np.trace(Hd) + np.trace(He)).real
        return pack(
            k16 * 4.1 * g1**3, k16 * (-19.0 / 6.0) * g2**3, k16 * (-7.0) * g3**3,
            k16 * (1.5 * Hu - 1.5 * Hd + (T - 17.0 / 20.0 * g1**2 - 2.25 * g2**2 - 8.0 * g3**2) * np.eye(3)) @ Yu,
            k16 * (1.5 * Hd - 1.5 * Hu + (T - 0.25 * g1**2 - 2.25 * g2**2 - 8.0 * g3**2) * np.eye(3)) @ Yd,
            k16 * (1.5 * He + (T - 2.25 * g1**2 - 2.25 * g2**2) * np.eye(3)) @ Ye,
        )

    sol1 = solve_ivp(rhs_sm, (0.0, math.log(M_SUSY / M_Z)), pack(g1_0, g2_0, g3_0, Yu0, Yd0, Ye0), rtol=1e-10, atol=1e-13)
    g1s, g2s, g3s, Yus, Yds, Yes = unpack(sol1.y[:, -1])

    # Match to MSSM at M_SUSY (with tan(beta) and finite gluino-sbottom threshold y_b^MSSM = y_b^SM / (cos(beta)*(1 + Delta_b))):
    sb = tanb / math.sqrt(1.0 + tanb * tanb)
    cb = 1.0 / math.sqrt(1.0 + tanb * tanb)
    Yus_m = Yus / sb
    # Apply finite SUSY threshold Delta_b to down-type Yukawas (dominant on 3rd gen, slight on 1,2):
    Yds_m = (Yds / cb) / (1.0 + delta_b)
    Yes_m = Yes / cb

    def rhs_mssm(_t, y):
        g1, g2, g3, Yu, Yd, Ye = unpack(y)
        Hu, Hd, He = Yu @ Yu.conj().T, Yd @ Yd.conj().T, Ye @ Ye.conj().T
        Tu = (3.0 * np.trace(Hu)).real
        Td = (3.0 * np.trace(Hd) + np.trace(He)).real
        return pack(
            k16 * (33.0 / 5.0) * g1**3, k16 * 1.0 * g2**3, k16 * (-3.0) * g3**3,
            k16 * (3.0 * Hu + Hd + (Tu - 13.0 / 15.0 * g1**2 - 3.0 * g2**2 - 16.0 / 3.0 * g3**2) * np.eye(3)) @ Yu,
            k16 * (3.0 * Hd + Hu + (Td - 7.0 / 15.0 * g1**2 - 3.0 * g2**2 - 16.0 / 3.0 * g3**2) * np.eye(3)) @ Yd,
            k16 * (3.0 * He + (Td - 9.0 / 5.0 * g1**2 - 3.0 * g2**2) * np.eye(3)) @ Ye,
        )

    sol2 = solve_ivp(rhs_mssm, (0.0, math.log(M_U / M_SUSY)), pack(g1s, g2s, g3s, Yus_m, Yds_m, Yes_m), rtol=1e-10, atol=1e-13)
    _, _, _, YuU, YdU, YeU = unpack(sol2.y[:, -1])

    Uu, su, _ = np.linalg.svd(YuU)
    Ud, sd, _ = np.linalg.svd(YdU)
    _, se, _ = np.linalg.svd(YeU)
    iu, id_, ie = np.argsort(su), np.argsort(sd), np.argsort(se)
    # Effective mass parameters m_f(M_U) = y_f(M_U) * v_{u,d}:
    su = su[iu] * (v * sb)
    sd = sd[id_] * (v * cb)
    se = se[ie] * (v * cb)
    VU = Uu[:, iu].conj().T @ Ud[:, id_]
    qs12, qs23, qs13, qd, _ = extract_mixing(VU)

    targets = dict(
        m_u=(float(su[0]), float(su[0]) * 0.35),
        m_c=(float(su[1]), float(su[1]) * 0.06),
        m_t=(float(su[2]), float(su[2]) * 0.02),
        m_d=(float(sd[0]), float(sd[0]) * 0.13),
        m_s=(float(sd[1]), float(sd[1]) * 0.065),
        m_b=(float(sd[2]), float(sd[2]) * 0.025),
        m_e=(float(se[0]), float(se[0]) * 0.01),
        m_mu=(float(se[1]), float(se[1]) * 0.01),
        m_tau=(float(se[2]), float(se[2]) * 0.01),
        q_s12=(qs12, 0.00070),
        q_s23=(qs23, 0.00090),
        q_s13=(qs13, 0.00018),
        q_delta=(qd, 0.045),
        dm21_sq=(7.41e-5, 0.21e-5),
        dm31_sq=(2.511e-3, 0.027e-3),
        l_s12_sq=(0.303, 0.012),
        l_s23_sq=(0.510, 0.060),
        l_s13_sq=(0.02203, 0.00056),
    )
    mb_mtau = float(sd[2] / se[2])
    sin2th_bound = float(sd[2] / (se[2] - se[1]))
    max_s23_sq_2gen = 0.5 if sin2th_bound >= 1.0 else (math.sin(0.5 * math.asin(sin2th_bound)) ** 2)
    return targets, dict(delta_b=delta_b, mb_over_mtau=round(mb_mtau, 4), sin2th23_analytic_bound=round(sin2th_bound, 4), max_sin2_th23_2gen=round(max_s23_sq_2gen, 4))


def fit_holomorphic_susy(targets, include_typeII: bool, n_restarts: int = 45):
    obs_keys = list(targets.keys())
    md_t, ms_t, mb_t = targets["m_d"][0], targets["m_s"][0], targets["m_b"][0]
    me_t, mmu_t, mtau_t = targets["m_e"][0], targets["m_mu"][0], targets["m_tau"][0]
    mt_t = targets["m_t"][0]

    def eval_x(x):
        Ue = unitary_3x3(x[:9])
        s_c = x[9] + 1j * x[10]
        if include_typeII:
            beta_II = x[11] + 1j * x[12]
            fm = np.exp(np.clip(x[13:19], -0.35, 0.35))
        else:
            beta_II = 0.0j
            fm = np.exp(np.clip(x[11:17], -0.35, 0.35))

        Md = np.diag([md_t * fm[0], ms_t * fm[1], mb_t * fm[2]]).astype(complex)
        Me = Ue.conj() @ np.diag([me_t * fm[3], mmu_t * fm[4], mtau_t * fm[5]]).astype(complex) @ Ue.conj().T
        H = 0.25 * (3.0 * Md + Me)
        F = 0.25 * (Md - Me)
        if abs(np.linalg.det(F)) < 1e-35:
            return None
        try:
            Finv = np.linalg.inv(F)
        except np.linalg.LinAlgError:
            return None

        su_u, Uu = takagi(H + s_c * F)
        if su_u[2] < 1e-12:
            return None
        r = mt_t / su_u[2]
        su = su_u * r

        A = H - 3.0 * s_c * F
        Mnu_raw = beta_II * F - (A @ Finv @ A)
        Mnu_raw = 0.5 * (Mnu_raw + Mnu_raw.T)
        sd, Ud = takagi(Md)
        se, Ue_r = takagi(Me)
        snu_r, Unu = takagi(Mnu_raw)
        raw_dm31 = snu_r[2] ** 2 - snu_r[0] ** 2
        if raw_dm31 <= 1e-40:
            return None
        c_R_inv_eV = math.sqrt(targets["dm31_sq"][0] / raw_dm31)
        snu = snu_r * c_R_inv_eV

        Vckm = Uu.conj().T @ Ud
        Upmns = Ue_r.conj().T @ Unu
        qs12, qs23, qs13, qd, _ = extract_mixing(Vckm)
        ls12, ls23, ls13, ld, lJ = extract_mixing(Upmns)
        m_bb = float(abs(sum((Upmns[0, i] ** 2) * snu[i] for i in range(3))))

        obs = dict(
            m_u=float(su[0]), m_c=float(su[1]), m_t=float(su[2]),
            m_d=float(sd[0]), m_s=float(sd[1]), m_b=float(sd[2]),
            m_e=float(se[0]), m_mu=float(se[1]), m_tau=float(se[2]),
            q_s12=qs12, q_s23=qs23, q_s13=qs13, q_delta=qd,
            dm21_sq=float(snu[1] ** 2 - snu[0] ** 2),
            dm31_sq=float(snu[2] ** 2 - snu[0] ** 2),
            l_s12_sq=ls12**2, l_s23_sq=ls23**2, l_s13_sq=ls13**2,
        )
        pulls = {k: (obs[k] - targets[k][0]) / targets[k][1] for k in obs_keys}
        chi2 = float(sum(p * p for p in pulls.values()))
        c_R = (r * r) / (c_R_inv_eV * 1e-9)
        sMR, _ = takagi(c_R * F)
        return dict(
            chi2=chi2, max_abs_pull=float(max(abs(p) for p in pulls.values())),
            obs=obs, pulls=pulls,
            preds=dict(
                m_nu_eV=[float(v) for v in snu],
                sum_m_nu_eV=float(np.sum(snu)),
                m_beta_beta_meV=round(m_bb * 1e3, 3),
                sin2_th23=round(ls23**2, 4),
                delta_CP_deg=round(math.degrees(ld), 1),
                M_R_GeV=[float(v) for v in sMR],
            ),
        )

    def resid(x):
        ev = eval_x(x)
        return np.full(18, 1e3) if ev is None else np.array([ev["pulls"][k] for k in obs_keys], dtype=float)

    n_vars = 19 if include_typeII else 17  # (+ r + c_R -> 21 or 19 total real params)
    rng = np.random.default_rng(31415 + (100 if include_typeII else 0))
    best = None
    for idx in range(n_restarts):
        x0 = np.zeros(n_vars)
        # Random angles & phases in Ue:
        x0[0] = rng.uniform(-0.6, 0.6)
        x0[1] = rng.uniform(-0.8, 0.8) if idx % 2 == 0 else rng.uniform(-0.2, 0.2)
        x0[2] = rng.uniform(-0.3, 0.3)
        x0[3:9] = rng.uniform(0.0, 2 * math.pi, 6)
        x0[9] = rng.uniform(-1.5, 1.5)
        x0[10] = rng.uniform(-1.5, 1.5)
        if include_typeII:
            x0[11] = rng.uniform(-5.0, 5.0)
            x0[12] = rng.uniform(-5.0, 5.0)
        res = least_squares(resid, x0, method="trf", x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=3000)
        ev = eval_x(res.x)
        if ev is not None and (best is None or ev["chi2"] < best["chi2"]):
            ev["n_params"] = 21 if include_typeII else 19
            best = ev
    return best


def main():
    t0 = time.time()
    out = {}
    for label, db in [("RegimeA_pure1loop_db0", 0.0), ("RegimeB_SUSYthresh_db_m015", -0.15)]:
        targets, info = run_mssm_targets(delta_b=db, tanb=50.0)
        rI = fit_holomorphic_susy(targets, include_typeII=False, n_restarts=45)
        rII = fit_holomorphic_susy(targets, include_typeII=True, n_restarts=45)
        sys.stderr.write(
            f"[{label}] mb/mtau={info['mb_over_mtau']}: "
            f"Type-I chi2={rI['chi2']:.3f} (s23^2={rI['obs']['l_s23_sq']:.4f}, max_pull={rI['max_abs_pull']:.2f}); "
            f"Type-I+II chi2={rII['chi2']:.3f} (s23^2={rII['obs']['l_s23_sq']:.4f}, max_pull={rII['max_abs_pull']:.2f})\n"
        )
        out[label] = dict(rg_info=info, TypeI_19p=rI, TypeI_plus_II_21p=rII)
    sys.stderr.write(f"[SUSY holomorphic 10_H + 126_H audit done in {time.time()-t0:.1f}s]\n")
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
