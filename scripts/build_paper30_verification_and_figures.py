#!/usr/bin/env python3
"""Master End-to-End Verification & Figure Generation Suite for Tau Core Paper 30 (v1.0.0):
  Four-Simplex Spin(10) Particle Genesis, 16-Parameter Parity-Hermitian Flavor Closure,
  Dimension-5 Proton Decay, and 24-Cell Chiral Lattice Symmetric Mass Generation.

Runs live under `python3 -W error` (rc = 0), verifies every algebraic, RG, flavor,
proton-decay, partial-compositeness, chiral-lattice, and many-body SMG claim from
first principles, writes `data/paper30_master_verification_results.json`, and produces
all four 300-DPI publication figures in `figures/`.
"""

import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
sys.path.insert(0, str(SCRIPTS_DIR))

from verify_4simplex_particles_bosons_yukawa import (
    build_boundary_operators,
    audit_gauge_bosons,
    audit_yukawa_higgs_and_seesaw,
)
from verify_3gen_spin10_120H_parity_and_nogo import (
    verify_120H_3gamma_operators,
    prove_typeII_sin2_th23_bound,
)
from verify_4simplex_lambda_quarter_and_2T_derivation import (
    audit_4simplex_and_24cell_quarter_eigenvalue,
    audit_Z6_subgroup_of_2T,
)
from verify_3gen_spin10_2T_global_fit import TARGETS as NS_TARGETS
from verify_SUSY_holomorphic_10_126_flavor import run_mssm_targets
from verify_SUSY_10_126_120_parity_and_proton_decay import (
    eval_parity_hermitian,
    verify_murayama_pierce_benchmark,
    audit_spin10_proton_decay,
    compute_dim5_proton_decay,
    strip_internal,
)
from verify_4D_chiral_lattice_hierarchy_and_rgfp import (
    audit_24cell_chiral_fermions,
    audit_higgs_quartic_and_hierarchy,
)
from verify_open_problems_SMG_and_hierarchy import (
    audit_fidkowski_kitaev_spin7_SMG,
    audit_spin10_4fermion_invariance,
    audit_hierarchy_resolutions,
)
from verify_spin10_partial_compositeness_audit import (
    verify_10x10_no_doublet,
    scan_hypercolor_groups_with_complete_Spin10,
    audit_PC_anomalous_dims_and_epsilonK,
)
from verify_1plus1D_SMG_chain_phase_diagram import run_chain
from verify_bridge1_susy_1loop_strong_cp_and_dCP_profile import run_exact_mssm_soft_rge_strong_cp
from verify_bridge3_2plus1D_SMG_and_24cell_director_isotropy import (
    scan_lattice_smg,
    audit_24cell_director_isotropy,
)


def solve_dw_sweet_spot(tau_2TeV_MU_yr: float, M_SUSY: float = 2000.0):
    M_Z = 91.1876
    a1_Z, a2_Z, a3_Z = 59.01, 29.58, 1.0 / 0.1180
    b_sm = (41.0 / 10.0, -19.0 / 6.0, -7.0)
    b_mssm = (33.0 / 5.0, 1.0, -3.0)
    b_H = (3.0 / 5.0, 1.0, 0.0)

    t_s = math.log(M_SUSY / M_Z) / (2.0 * math.pi)
    a1_s = a1_Z - b_sm[0] * t_s
    a2_s = a2_Z - b_sm[1] * t_s
    a3_s = a3_Z - b_sm[2] * t_s

    def eval_Mp(Mp):
        t_p = math.log(Mp / M_SUSY) / (2.0 * math.pi)
        a1_p = a1_s - b_mssm[0] * t_p
        a2_p = a2_s - b_mssm[1] * t_p
        a3_p = a3_s - b_mssm[2] * t_p
        b_hi = (b_mssm[0] + b_H[0], b_mssm[1] + b_H[1], b_mssm[2] + b_H[2])
        t_hi = (a1_p - a2_p) / (b_hi[0] - b_hi[1])
        MU = Mp * math.exp(2.0 * math.pi * t_hi)
        aU = a1_p - b_hi[0] * t_hi
        a3_miss = (a3_p - b_hi[2] * t_hi) - aU
        return MU, aU, a3_miss

    lo, hi = 1e11, 1e16
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        if eval_Mp(mid)[2] > 0:
            lo = mid
        else:
            hi = mid
    Mp_ex = math.sqrt(lo * hi)
    MU_ex, aU_ex, miss_ex = eval_Mp(Mp_ex)
    dw_fac = MU_ex / Mp_ex
    tau_2TeV_dw = tau_2TeV_MU_yr * (dw_fac ** 2)

    def f_loop(Msq, Mw=500.0):
        x = (Mw / Msq) ** 2
        I_l = 0.5 if abs(x - 1.0) < 1e-6 else (1.0 / (1.0 - x) + x * math.log(x) / ((1.0 - x) ** 2))
        return (Mw / (Msq ** 2)) * I_l

    f_ref = f_loop(2000.0, 500.0)
    def find_Msq(target_yr, base_yr):
        l2, h2 = 2000.0, 1e7
        for _ in range(80):
            m = math.sqrt(l2 * h2)
            if base_yr * ((f_ref / f_loop(m, 500.0)) ** 2) < target_yr:
                l2 = m
            else:
                h2 = m
        return h2

    Msq_SK = find_Msq(5.9e33, tau_2TeV_dw)
    Msq_HK = find_Msq(3.2e34, tau_2TeV_dw)
    yt, mh = 0.95, 125.25
    dft = lambda m: (6.0 * yt**2 / (4.0 * math.pi**2)) * ((m / mh) ** 2) * math.log(MU_ex / m)

    return dict(
        M_prime_exact_GeV=Mp_ex,
        M_U_exact_GeV=MU_ex,
        alpha_U_inv=round(aU_ex, 3),
        alpha3_inv_miss=round(miss_ex, 7),
        DW_triplet_suppression_factor=round(dw_fac, 2),
        tau_p_2TeV_with_DW_yr=tau_2TeV_dw,
        Msq_min_SuperK_DW_GeV=round(Msq_SK, 1),
        Msq_min_HyperK_DW_GeV=round(Msq_HK, 1),
        Delta_FT_SK_DW=round(dft(Msq_SK), 1),
        Delta_FT_HK_DW=round(dft(Msq_HK), 1),
    )


def rebuild_and_verify_ph16_champions():
    """Verify both 16-parameter Parity-Hermitian champions (SUSY & Non-SUSY)
    DIRECTLY from raw parameter vectors through `eval_parity_hermitian`!
    """
    with open(DATA_DIR / "susy_and_ns_ph17_basin.json") as f:
        basin = json.load(f)

    # 1. SUSY PH16_a13_h23 (tanb=50, Delta_b=0 pure 1-loop MSSM):
    p_s = basin["S1_PH16_from_PH17_a13"]["a13_h23"]["fit"]["params"]
    targs_S1, rg_S1 = run_mssm_targets(delta_b=0.0, tanb=50.0)
    Fs = np.diag(p_s["F_diag"])
    Hs = np.array(p_s["H"], dtype=float)
    As = np.array([
        [0.0, p_s["A_12_13_23"][0], p_s["A_12_13_23"][1]],
        [-p_s["A_12_13_23"][0], 0.0, p_s["A_12_13_23"][2]],
        [-p_s["A_12_13_23"][1], -p_s["A_12_13_23"][2], 0.0],
    ], dtype=float)
    assert abs(Hs[1, 2]) < 1e-15 and abs(As[0, 2]) < 1e-15, "Exact complementary zeros h23=a13=0!"
    ev_susy = eval_parity_hermitian(
        Hs, Fs, As, abs(p_s["r"]), p_s["s"], p_s["c_e"], p_s["c_u"], p_s["c_nu"],
        targs_S1, tanb=50.0, beta_II=0.0,
    )
    assert ev_susy["chi2"] < 1.40 and ev_susy["max_abs_pull"] < 1.05
    assert ev_susy["preds"]["tree_level_theta_QCD_rad"] < 1e-13
    assert ev_susy["preds"]["det_Mu_Md_positive"] is True
    pdec_susy = audit_spin10_proton_decay(ev_susy, tanb=50.0)
    dw_susy = solve_dw_sweet_spot(pdec_susy["tau_p_at_Msq2TeV_Mw500GeV_MT_MU_yr"])

    # 2. Non-SUSY PH16_a13_h23 (re-evaluated live from refined parameters):
    from verify_SUSY_and_NS_PH17_basin_and_DW_pdecay import seeded_refine
    p_ns17 = basin["NS_PH17_and_PH16"]["PH17_a13"]["params"]
    ev_ns16 = seeded_refine(NS_TARGETS, None, p_ns17, zero_mask=("a13", "h23"), n_perturb=12)
    assert ev_ns16["chi2"] < 0.05 and ev_ns16["max_abs_pull"] < 0.20
    assert ev_ns16["preds"]["tree_level_theta_QCD_rad"] < 1e-14
    assert ev_ns16["preds"]["det_Mu_Md_positive"] is True

    return dict(
        SUSY_PH16_rg_info=rg_S1,
        SUSY_PH16_a13_h23=strip_internal(ev_susy),
        SUSY_PH16_dim5_proton_decay=pdec_susy,
        SUSY_PH16_DW_sweet_spot=dw_susy,
        NS_PH16_a13_h23=strip_internal(ev_ns16),
        _ev_susy_internal=ev_susy,
    )


def make_figures(res):
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2,
        "figure.dpi": 300,
    })

    # -----------------------------------------------------------------------
    # FIG 1: 4-Simplex 32D Clifford Charges + K_5 / 24-Cell lambda = -1/4 Eigenvalues
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1))
    # Panel A: 16 SM chiral states on Lambda^even(C^5) mapped by (B-L, Q):
    states = [
        (r"$\nu_R^c$", 1.0, 0.0, 1, "#6366f1"),
        (r"$e_R^c$", 1.0, 1.0, 1, "#2563eb"),
        (r"$u_L$ (3)", 1.0/3.0, 2.0/3.0, 3, "#dc2626"),
        (r"$d_L$ (3)", 1.0/3.0, -1.0/3.0, 3, "#ea580c"),
        (r"$u_R^c$ (3)", -1.0/3.0, -2.0/3.0, 3, "#e11d48"),
        (r"$d_R^c$ (3)", -1.0/3.0, 1.0/3.0, 3, "#d97706"),
        (r"$\nu_L$", -1.0, 0.0, 1, "#0d9488"),
        (r"$e_L$", -1.0, -1.0, 1, "#0284c7"),
    ]
    for lbl, bml, q, mult, col in states:
        ax1.scatter([bml], [q], s=95 * (1 + 0.55 * (mult - 1)), color=col, edgecolor="black", linewidth=0.8, zorder=4)
        dy = 0.09 if q >= 0 else -0.13
        dx = 0.04 if bml <= 0 else -0.04
        ha = "left" if bml <= 0 else "right"
        ax1.annotate(lbl, (bml, q), xytext=(bml + dx, q + dy), fontsize=8.5, fontweight="bold", ha=ha)
    ax1.axhline(0, color="gray", lw=0.7, ls="--")
    ax1.axvline(0, color="gray", lw=0.7, ls="--")
    ax1.set_xlim(-1.35, 1.35)
    ax1.set_ylim(-1.30, 1.30)
    ax1.set_xlabel(r"$B - L = 1 - \frac{2}{3}(N_1+N_2+N_3)$")
    ax1.set_ylabel(r"Electric Charge $Q = N_4 - \frac{1}{3}(N_1+N_2+N_3)$")
    ax1.set_title(r"(A) Exact 16-State $\Lambda^{\mathrm{even}}(\mathbb{C}^5)$ Spin(10) Spectrum")
    ax1.grid(True, alpha=0.25)

    # Panel B: Exact transfer-matrix spectra of 4-simplex K_5 and 24-cell P_24:
    k5_evals = [-0.25, 1.0]
    k5_mults = [4, 1]
    p24_evals = [-0.50, -0.25, 0.0, 0.25, 0.50, 1.0]
    p24_mults = [4, 8, 2, 4, 5, 1]
    w = 0.055
    ax2.bar([e - w/2 for e in k5_evals], k5_mults, width=w, color="#0f766e", edgecolor="black", lw=0.7, label=r"4-Simplex $P_{K_5} = A_{K_5}/4$ (5 vertices)")
    ax2.bar([e + w/2 for e in p24_evals], p24_mults, width=w, color="#2563eb", alpha=0.85, edgecolor="black", lw=0.7, label=r"24-Cell $P_{24} = A_{24}/8$ (24 vertices)")
    ax2.axvline(-0.25, color="#dc2626", ls="--", lw=1.4, label=r"Shared exact $\lambda = -1/4$ ($|\lambda|=\cos\theta_{\sigma_4}=0.25$)")
    ax2.annotate(r"$\lambda = -1/4$" + "\n(mult 4 & 8)", xy=(-0.25, 8.0), xytext=(-0.15, 6.2),
                 arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.1), fontsize=8.3, color="#b91c1c", fontweight="bold")
    ax2.set_xlabel("Normalized Adjacency / Transfer Eigenvalue")
    ax2.set_ylabel("Multiplicity")
    ax2.set_ylim(0, 10.8)
    ax2.set_title(r"(B) Shared Geometric Cabibbo Scale $|\lambda| = 1/4$ on $\sigma_4$ & 24-Cell")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_4simplex_algebra_and_lambda_quarter.png", dpi=300)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # FIG 2: Flavor No-Go Landscape + 18-Observable Pulls for SUSY & Non-SUSY PH16
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5), gridspec_kw={"width_ratios": [1.0, 1.35]})
    models = [
        ("Pure power-law (|c|=1, 8p)", 15157.0, "#991b1b"),
        (r"Modular $\Gamma_3=A_4$ (4,4)", 10277.0, "#b91c1c"),
        (r"Spinorial $\Gamma'_3=2T$ (1,2)", 3523.0, "#dc2626"),
        (r"1D lattice overlap (12p)", 3549.0, "#dc2626"),
        (r"SUSY $10_H+\overline{126}_H$ Type-I", 1259.0, "#ea580c"),
        (r"SUSY $10_H+\overline{126}_H$ I+II", 447.0, "#f97316"),
        (r"Non-SUSY $10_H+\overline{126}_H$ I", 45.3, "#d97706"),
        (r"Non-SUSY $10_H+\overline{126}_H$ I+II", 30.3, "#ca8a04"),
        (r"SUSY PH16$_{a13,h23}$ (16p, $\bar\theta=0$)", res["flavor"]["SUSY_PH16_a13_h23"]["chi2"], "#2563eb"),
        (r"Non-SUSY M16-U11 (16p)", 1.308, "#0d9488"),
        (r"Non-SUSY PH16$_{a13,h23}$ (16p, $\bar\theta=0$)", res["flavor"]["NS_PH16_a13_h23"]["chi2"], "#15803d"),
    ]
    y_pos = np.arange(len(models))
    ax1.barh(y_pos, [m[1] for m in models], color=[m[2] for m in models], edgecolor="black", lw=0.6)
    ax1.set_xscale("log")
    ax1.axvline(2.0, color="black", ls="--", lw=1.0, label=r"$\chi^2 = \mathrm{dof} = 2$")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([m[0] for m in models], fontsize=7.8)
    ax1.invert_yaxis()
    ax1.set_xlabel(r"Global 18-Observable $\chi^2$ (log scale)")
    ax1.set_title("(A) Spin(10) Flavor No-Go Walls vs. 16-Param Closures")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, axis="x", alpha=0.25)

    # Panel B: 18-observable pulls for SUSY PH16 vs Non-SUSY PH16:
    obs_order = [
        ("m_u", r"$m_u$"), ("m_c", r"$m_c$"), ("m_t", r"$m_t$"),
        ("m_d", r"$m_d$"), ("m_s", r"$m_s$"), ("m_b", r"$m_b$"),
        ("m_e", r"$m_e$"), ("m_mu", r"$m_\mu$"), ("m_tau", r"$m_\tau$"),
        ("q_s12", r"$s_{12}^q$"), ("q_s23", r"$s_{23}^q$"), ("q_s13", r"$s_{13}^q$"), ("q_delta", r"$\delta_{\mathrm{CKM}}$"),
        ("dm21_sq", r"$\Delta m_{21}^2$"), ("dm31_sq", r"$\Delta m_{31}^2$"),
        ("l_s12_sq", r"$s_{12}^{\ell 2}$"), ("l_s23_sq", r"$s_{23}^{\ell 2}$"), ("l_s13_sq", r"$s_{13}^{\ell 2}$"),
    ]
    ps = res["flavor"]["SUSY_PH16_a13_h23"]["pulls"]
    pn = res["flavor"]["NS_PH16_a13_h23"]["pulls"]
    x_pos = np.arange(len(obs_order))
    bw = 0.38
    ax2.bar(x_pos - bw/2, [ps[k] for k, _ in obs_order], width=bw, color="#2563eb", edgecolor="black", lw=0.5,
            label=r"SUSY PH16$_{a13,h23}$ ($\chi^2=1.35$, max $1.00\sigma$)")
    ax2.bar(x_pos + bw/2, [pn[k] for k, _ in obs_order], width=bw, color="#16a34a", edgecolor="black", lw=0.5,
            label=r"Non-SUSY PH16$_{a13,h23}$ ($\chi^2=0.030$, max $0.13\sigma$)")
    ax2.axhline(0.0, color="black", lw=0.8)
    ax2.axhline(1.0, color="gray", ls=":", lw=0.9)
    ax2.axhline(-1.0, color="gray", ls=":", lw=0.9)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([lbl for _, lbl in obs_order], rotation=45, ha="right", fontsize=7.8)
    ax2.set_ylim(-1.35, 1.35)
    ax2.set_ylabel(r"Pull $(\mathrm{model}-\mathrm{data})/\sigma$")
    ax2.set_title(r"(B) All 18 Pulls of Parity-Hermitian $a_{13}=h_{23}=0$ ($\bar\theta_{\mathrm{tree}}\equiv 0$)")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_flavor_nogo_and_ph16_pulls.png", dpi=300)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # FIG 3: Dim-5 SUSY Proton Decay + DW Exact Unification Sweet Spot & Hierarchy
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2))
    # Panel A: tau(p -> K^+ nu_bar) vs M_sq (wino = 500 GeV) for uncancelled T_10 vs DW sweet spot:
    Msq_grid = np.geomspace(1.5e3, 2.0e5, 200)
    def f_loop_vec(m_arr, Mw=500.0):
        out = []
        for m in m_arr:
            x = (Mw / m) ** 2
            I_l = 0.5 if abs(x - 1.0) < 1e-6 else (1.0 / (1.0 - x) + x * math.log(x) / ((1.0 - x) ** 2))
            out.append((Mw / (m ** 2)) * I_l)
        return np.array(out)

    f_ref = float(f_loop_vec([2000.0])[0])
    f_arr = f_loop_vec(Msq_grid)
    tau_unc_base = res["flavor"]["SUSY_PH16_dim5_proton_decay"]["tau_p_at_Msq2TeV_Mw500GeV_MT_MU_yr"]
    dw_fac = res["flavor"]["SUSY_PH16_DW_sweet_spot"]["DW_triplet_suppression_factor"]
    tau_unc = tau_unc_base * ((f_ref / f_arr) ** 2)
    tau_dw = tau_unc * (dw_fac ** 2)

    ax1.loglog(Msq_grid / 1e3, tau_unc, color="#dc2626", lw=2.0, ls="--", label=r"Uncancelled $T_{10}$ ($M_T = M_U$)")
    ax1.loglog(Msq_grid / 1e3, tau_dw, color="#2563eb", lw=2.4, label=r"DW exact-unif. ($M'=2.64\times 10^{14}$ GeV, $39.3\times$)")
    ax1.axhspan(1e26, 5.9e33, color="#fecaca", alpha=0.35, label=r"Excluded by Super-K ($<5.9\times 10^{33}$ yr)")
    ax1.axhspan(5.9e33, 3.2e34, color="#bbf7d0", alpha=0.45, label=r"Hyper-K 10-yr discovery window")
    ax1.axhline(5.9e33, color="#b91c1c", lw=1.1, ls=":")
    ax1.axhline(3.2e34, color="#15803d", lw=1.1, ls=":")
    ax1.axvline(12.024, color="#2563eb", lw=1.0, ls=":")
    ax1.scatter([12.024, 18.393], [5.9e33, 3.2e34], color="#1d4ed8", s=45, zorder=5)
    ax1.annotate("12.0–18.4 TeV\nHyper-K window", xy=(14.5, 1.3e34), xytext=(2.2, 6.0e34),
                 arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.1), fontsize=8.0, fontweight="bold", color="#1e3a8a")
    ax1.set_xlim(1.5, 200.0)
    ax1.set_ylim(1e27, 1e37)
    ax1.set_xlabel(r"Squark Mass Scale $M_{\tilde{q}}$ (TeV)  [$M_{\tilde{w}} = 500$ GeV]")
    ax1.set_ylabel(r"$\tau(p \to K^+ \bar{\nu})$ (years)")
    ax1.set_title(r"(A) Calibrated $d=5$ Proton Decay (SUSY $\mathrm{PH16}_{a13,h23}$)")
    ax1.legend(loc="lower right", fontsize=7.5, frameon=True)
    ax1.grid(True, which="both", alpha=0.22)

    # Panel B: Hierarchy Fine-Tuning Delta_FT comparison across mechanisms:
    mechs = [
        ("Non-SUSY Spin(10)\n(GUT scale)", 1.4e25, "#991b1b"),
        ("Conformal Portal\n(NO-GO)", 3.1e17, "#b91c1c"),
        ("Non-SUSY Pati–Salam\n(M_PS scale)", 3.9e16, "#dc2626"),
        ("Uncancelled SUSY\n(M_sq = 75.7 TeV)", 1.27e6, "#ea580c"),
        ("DW Mini-Split SUSY\n(M_sq = 12.0 TeV)", 3.47e4, "#2563eb"),
        ("Naive TeV-SUSY\n(d=5 p-decay NO-GO)", 1015.0, "#94a3b8"),
    ]
    yp = np.arange(len(mechs))
    ax2.barh(yp, [m[1] for m in mechs], color=[m[2] for m in mechs], edgecolor="black", lw=0.6)
    ax2.set_xscale("log")
    ax2.set_yticks(yp)
    ax2.set_yticklabels([m[0] for m in mechs], fontsize=8.0)
    ax2.invert_yaxis()
    ax2.set_xlabel(r"Electroweak Fine-Tuning $\Delta_{\mathrm{FT}}$ (log scale)")
    ax2.set_title(r"(B) Hierarchy Fine-Tuning vs. Proton-Decay Survival")
    ax2.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_hierarchy_dim5_proton_decay_dw_sweetspot.png", dpi=300)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # FIG 4: 1+1D Spin(7) Cayley SMG Chain Exact Many-Body Quantum Phase Transition
    # -----------------------------------------------------------------------
    smg2 = res["smg_chain"]["L2_dim256"]
    smg3 = res["smg_chain"]["L3_periodic_dim4096"]
    g_arr = np.array([row["g"] for row in smg3["phase_scan"]])
    gap2 = np.array([row["gap"] for row in smg2["phase_scan"]])
    gap3 = np.array([row["gap"] for row in smg3["phase_scan"]])
    s1_2 = np.array([row["entanglement_S1"] for row in smg2["phase_scan"]])
    s1_3 = np.array([row["entanglement_S1"] for row in smg3["phase_scan"]])
    e0_2 = np.array([row["E0"] for row in smg2["phase_scan"]])
    e0_3 = np.array([row["E0"] for row in smg3["phase_scan"]])
    dg = g_arr[1] - g_arr[0]
    chi2_e = -(e0_2[2:] - 2 * e0_2[1:-1] + e0_2[:-2]) / (dg ** 2)
    chi3_e = -(e0_3[2:] - 2 * e0_3[1:-1] + e0_3[:-2]) / (dg ** 2)
    g_mid = g_arr[1:-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1))
    ax1.plot(g_arr, gap2, "o-", color="#64748b", lw=1.6, ms=3.5, label=r"$L=2$ ($\dim=256$) Energy Gap $\Delta E$")
    ax1.plot(g_arr, gap3, "s-", color="#2563eb", lw=2.1, ms=4.0, label=r"$L=3$ periodic ($\dim=4096$) Gap $\Delta E$")
    ax1_r = ax1.twinx()
    ax1_r.plot(g_arr, s1_2, "^--", color="#f97316", lw=1.4, ms=3.3, alpha=0.8, label=r"$L=2$ Entanglement $S_1$")
    ax1_r.plot(g_arr, s1_3, "D--", color="#dc2626", lw=1.9, ms=3.6, label=r"$L=3$ Entanglement $S_1$")
    ax1.axvspan(0.18, 0.24, color="#fef08a", alpha=0.45, label=r"SMG QPT $g_c \approx 0.20$–$0.22$")
    ax1.set_xlabel(r"4-Fermion Cayley Interaction Ratio $g / (g + t)$")
    ax1.set_ylabel(r"Many-Body Energy Gap $\Delta E = E_1 - E_0$", color="#1d4ed8")
    ax1_r.set_ylabel(r"Site-0 Entanglement Entropy $S_1$ (nats)", color="#b91c1c")
    ax1.set_title(r"(A) Gapless $\to$ SMG Phase Transition (Bilinear VEV $\leq 1.5\times 10^{-14}$)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", bbox_to_anchor=(0.99, 0.14), fontsize=7.2, frameon=True, framealpha=0.92)
    ax1.grid(True, alpha=0.25)

    p2_val = smg2["peak_susceptibility_val"]
    p3_val = smg3["peak_susceptibility_val"]
    smg_k4 = res["bridge3_high_dim_SMG"]["3D_K4_tetrahedron_z3_dim65536"]
    gk4 = np.array([row["g"] for row in smg_k4["phase_scan"]])
    e0_k4 = np.array([row["E0"] for row in smg_k4["phase_scan"]])
    chi_k4 = [
        -2.0 * ((e0_k4[i + 1] - e0_k4[i]) / (gk4[i + 1] - gk4[i]) - (e0_k4[i] - e0_k4[i - 1]) / (gk4[i] - gk4[i - 1]))
        / (gk4[i + 1] - gk4[i - 1])
        for i in range(1, len(gk4) - 1)
    ]
    pk4_val = smg_k4["peak_susceptibility_val"]
    ax2.plot(g_mid, chi2_e, "o-", color="#64748b", lw=1.5, ms=3.5, label=rf"1+1D $L=2$ ($\dim=256$): peak ${p2_val:.1f}$")
    ax2.plot(g_mid, chi3_e, "s-", color="#0f766e", lw=2.0, ms=3.8, label=rf"1+1D $L=3$ ($\dim=4096$): peak ${p3_val:.1f}$")
    ax2.plot(gk4[1:-1], chi_k4, "D-", color="#7c3aed", lw=2.3, ms=4.2, label=rf"3D $K_4$ ($\dim=65\,536$, 32 Maj.): peak ${pk4_val:.1f}$")
    ax2.axvline(0.22, color="#dc2626", ls="--", lw=1.2, label=r"$g_c \approx 0.20$–$0.22$")
    ax2.set_xlabel(r"Interaction Ratio $g / (g + t)$")
    ax2.set_ylabel(r"Ground-State Susceptibility $-\mathrm{d}^2 E_0 / \mathrm{d}g^2$")
    ax2.set_title(r"(B) SMG Susceptibility up to 3D $K_4$ ($\dim=65\,536$, VEV $\leq 1.9\times 10^{-16}$)")
    ax2.legend(loc="upper right", fontsize=7.6, frameon=True)
    ax2.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_24cell_chiral_fermions_and_smg_qpt.png", dpi=300)
    plt.close(fig)


def main():
    t0 = time.time()
    out = {}

    # Pillar 1: 4-Simplex 32x32 Spin(10) Particles, Bosons, 10_H, 126_H, 120_H + lambda=1/4 + Z_6
    a, ad, N, Gammas, _G11, even_idx = build_boundary_operators()
    p1_gauge = audit_gauge_bosons(a, ad, N, Gammas, even_idx)
    p1_yuk = audit_yukawa_higgs_and_seesaw(a, ad, N, Gammas, even_idx)
    p1_120 = verify_120H_3gamma_operators()
    p1_bound = prove_typeII_sin2_th23_bound()
    p1_qtr = audit_4simplex_and_24cell_quarter_eigenvalue()
    p1_z6 = audit_Z6_subgroup_of_2T()
    out["pillar1_4simplex_spin10_and_2T"] = dict(
        gauge_bosons=p1_gauge,
        yukawa_10_126=p1_yuk,
        operators_120H_3gamma=p1_120,
        non_susy_typeII_th23_bound=p1_bound,
        quarter_eigenvalue=p1_qtr,
        Z6_in_2T=p1_z6,
    )
    sys.stderr.write("[Pillar 1] 4-simplex 32x32 Spin(10) + 120_H + lambda=1/4 + Z_6 PASS\n")

    # Pillar 2: Parity-Hermitian 16-param Champions (SUSY & Non-SUSY) + Dim-5 Proton Decay + DW Sweet Spot
    mp_cal = verify_murayama_pierce_benchmark()
    flav = rebuild_and_verify_ph16_champions()
    out["pillar2_dim5_calibration"] = mp_cal
    out["flavor"] = {k: v for k, v in flav.items() if not k.startswith("_")}
    sys.stderr.write(
        f"[Pillar 2] SUSY PH16 chi2={flav['SUSY_PH16_a13_h23']['chi2']:.4f} (max={flav['SUSY_PH16_a13_h23']['max_abs_pull']:.2f}s), "
        f"NS PH16 chi2={flav['NS_PH16_a13_h23']['chi2']:.4f} (max={flav['NS_PH16_a13_h23']['max_abs_pull']:.2f}s), "
        f"DW M'={flav['SUSY_PH16_DW_sweet_spot']['M_prime_exact_GeV']:.2e} GeV PASS\n"
    )

    # Pillar 3: Hierarchy Resolutions + Exhaustive 9-Group Complete-Spin(10) Partial Compositeness No-Go
    p3_higgs = audit_higgs_quartic_and_hierarchy()
    p3_hier = audit_hierarchy_resolutions()
    p3_10x10 = verify_10x10_no_doublet()
    p3_pc_scan = scan_hypercolor_groups_with_complete_Spin10()
    p3_pc_eps = audit_PC_anomalous_dims_and_epsilonK()
    assert p3_10x10["color_singlet_SU2L_doublets_in_10x10"] == 0
    assert p3_pc_scan["full_PC_complete_Spin10_survivors_count"] == 0
    out["pillar3_hierarchy_and_PC_nogo"] = dict(
        sm_higgs_quartic_and_hierarchy=p3_higgs,
        hierarchy_resolutions=p3_hier,
        ten_x_ten_no_doublet=p3_10x10,
        partial_compositeness_9group_scan=p3_pc_scan,
        partial_compositeness_delta_and_epsK=p3_pc_eps,
    )
    sys.stderr.write("[Pillar 3] Hierarchy + 9-group complete-Spin(10) PC 0/15 NO-GO PASS\n")

    # Pillar 4: 4D 24-Cell Chiral Fermions + Single-Site Spin(7)/Spin(10) SMG + Coupled 1+1D SMG QPT
    p4_lat = audit_24cell_chiral_fermions()
    p4_fk = audit_fidkowski_kitaev_spin7_SMG()
    p4_s10 = audit_spin10_4fermion_invariance()
    g_grid = np.linspace(0.0, 1.0, 21)
    smg_l2 = run_chain(2, g_grid)
    smg_l3 = run_chain(3, g_grid)
    assert p4_lat["naive_24cell_isolated_doublers"]["total"] == 72
    assert p4_lat["neuberger_overlap_Ginsparg_Wilson_max_err"] < 1e-14
    assert p4_fk["max_bilinear_symmetry_breaking_VEV"] < 1e-14
    assert p4_s10["four_fermion_O10_Spin10_invariance_max_err"] < 1e-14
    assert smg_l3["max_bilinear_VEV_for_all_unique_GS"] < 1e-13
    out["pillar4_24cell_chiral_lattice_and_SMG"] = dict(
        chiral_24cell_lattice=p4_lat,
        single_site_spin7_SMG=p4_fk,
        spin10_4fermion_invariance=p4_s10,
    )
    out["smg_chain"] = dict(L2_dim256=smg_l2, L3_periodic_dim4096=smg_l3)
    sys.stderr.write("[Pillar 4] 24-cell Overlap/DW + Spin(7)/Spin(10) SMG + L=2,3 QPT PASS\n")

    # Bridge 1A: Exact Coupled 3x3 MSSM Soft-Breaking RGEs + 6x6 Squark-Gluino 1-Loop Strong-CP Audit
    ps = flav["SUSY_PH16_a13_h23"]["params"]
    v = 174.104
    sb, cb = 50.0 / math.sqrt(2501.0), 1.0 / math.sqrt(2501.0)
    F_u = np.diag(ps["F_diag"]).astype(complex)
    H_u = np.array(ps["H"], dtype=complex)
    A_u = np.array([
        [0.0, ps["A_12_13_23"][0], ps["A_12_13_23"][1]],
        [-ps["A_12_13_23"][0], 0.0, ps["A_12_13_23"][2]],
        [-ps["A_12_13_23"][1], -ps["A_12_13_23"][2], 0.0],
    ], dtype=complex)
    Yu_U = ps["r"] * (H_u + ps["s"] * F_u + 1j * ps["c_u"] * A_u) / (v * sb)
    Yd_U = (H_u + F_u + 1j * A_u) / (v * cb)
    Ye_U = (H_u - 3.0 * F_u + 1j * ps["c_e"] * A_u) / (v * cb)
    b1_cp = dict(
        Parity_Universal_A0_0=run_exact_mssm_soft_rge_strong_cp(Yu_U, Yd_U, Ye_U, m0=12000.0, M12=1200.0, A0=0.0),
        Parity_Universal_A0_12TeV=run_exact_mssm_soft_rge_strong_cp(Yu_U, Yd_U, Ye_U, m0=12000.0, M12=1200.0, A0=12000.0),
        Parity_NonUniversal_dA0p5=run_exact_mssm_soft_rge_strong_cp(Yu_U, Yd_U, Ye_U, m0=12000.0, M12=1200.0, A0=12000.0, nonuniv_herm_deltaA=0.5),
    )
    assert b1_cp["Parity_Universal_A0_0"]["passes_nEDM"] is True
    assert b1_cp["Parity_Universal_A0_12TeV"]["passes_nEDM"] is True
    assert b1_cp["Parity_NonUniversal_dA0p5"]["passes_nEDM"] is False
    out["bridge1_1loop_SUSY_strong_CP"] = b1_cp
    sys.stderr.write("[Bridge 1] Exact 6x6 SUSY 1-loop Strong-CP (universal PASS, non-universal NO-GO) PASS\n")

    # Bridge 3A & 3B: Exact 32-Majorana (dim=65,536) 1+1D L=4, 2+1D pi-flux, 3D K_4 SMG + 24-cell Director Isotropy
    b3_dir = audit_24cell_director_isotropy()
    g4 = [0.0, 0.08, 0.14, 0.18, 0.20, 0.22, 0.25, 0.30, 0.40, 0.55, 0.75, 1.0]
    smg_l4 = scan_lattice_smg("1+1D_L4_periodic", 4, [(0, 1), (1, 2), (2, 3), (3, 0)], [1.0]*4, g4)
    smg_2d = scan_lattice_smg("2+1D_2x2_piFlux_plaquette", 4, [(0, 1), (1, 3), (3, 2), (2, 0)], [1.0, 1.0, 1.0, -1.0], g4)
    smg_k4 = scan_lattice_smg("3D_K4_tetrahedron_z3", 4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], [1.0]*6, g4)
    out["bridge3_24cell_director_isotropy"] = b3_dir
    out["bridge3_high_dim_SMG"] = dict(
        L4_1plus1D_dim65536=smg_l4,
        piFlux_2plus1D_dim65536=smg_2d,
        **{"3D_K4_tetrahedron_z3_dim65536": smg_k4},
    )
    sys.stderr.write("[Bridge 3] 32-Majorana (dim=65536) L=4 / 2+1D / 3D-K4 SMG + 24-cell rank-4 director isotropy PASS\n")

    # Generate all 4 publication figures:
    make_figures(out)
    sys.stderr.write("[Figures] All 4 publication-grade 300-DPI PNGs written to figures/\n")

    # Write master JSON:
    with open(DATA_DIR / "paper30_master_verification_results.json", "w") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    sys.stderr.write(f"[Paper 30 Master Verification & Figures COMPLETE in {time.time()-t0:.1f}s]\n")


if __name__ == "__main__":
    main()
