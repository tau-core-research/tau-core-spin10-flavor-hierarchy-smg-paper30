#!/usr/bin/env python3
"""Strict Non-SUSY Hierarchy Audit on the 4-Simplex Spin(10) -> Pati-Salam -> SM Framework:

Tests THREE concrete, strictly non-SUSY hierarchy mechanisms quantitatively (python3 -W error):
  1. Candidate 3A — Minimal Composite pNGB Higgs (Sp(4) = Spin(5) hypercolor,
     SU(4)/Sp(4) -> SO(5)/SO(4) coset, custodial SO(4) exact!):
       - Solves the EXACT 2-step Spin(10) -> Pati-Salam -> SM gauge unification
         INCLUDING the 4 Weyl Sp(4) hyperfermions in (1,2,1) + (1,1,2) of Pati-Salam
         from M_HC ~ 2 TeV to M_U (checking M_PS, M_U, alpha_U^-1, and Super-K tau_p!).
       - Computes the EXACT self-consistent Sp(4) dimensional-transmutation confinement
         scale Lambda_HC = M_U * exp(-2 pi / (b_HC * alpha_U)) using ZERO new gauge coupling!
       - Computes exact pNGB fine-tuning Delta_FT = 2/xi (at LHC-safe f = 1.0 TeV, xi = 0.0606)
         AND explicitly proves the naive-ETC top-mass NO-GO (showing partial compositeness /
         walking CFT is strictly required between TeV and M_PS!).
  2. Candidate 3B — Cosmological Relaxion (Graham-Kaplan-Rajendran):
       - Proves the exact model-independent cutoff ceiling
             M_max = (sqrt(3) M_Pl v_EW)^{1/2} = 3.22e10 GeV < M_PS = 5.63e11 GeV
         (and M_max^QCD ~ 1e5 GeV), plus Starobinsky H_inf = 1.3e13 GeV >> 246 GeV conflict
         -> PROVEN NO-GO for Relaxion at M_PS and M_U!
  3. Candidate 3C — Z_2 Twin Higgs (Mirror-Wall Copy):
       - Shows exact little-hierarchy protection up to Lambda_UV ~ 4 pi f_twin ~ 15 TeV,
         proving it STILL requires a Composite or SUSY UV completion above 15 TeV to reach M_PS!
"""

import json
import math
import numpy as np


# ---------------------------------------------------------------------------
# Candidate 3A: Sp(4) Composite pNGB Higgs + Self-Consistent 2-Step Unification
# ---------------------------------------------------------------------------
def audit_composite_Sp4_higgs():
    M_Z = 91.1876
    M_HC = 2000.0  # 2 TeV hypercolor resonance scale (LHC-safe!)
    v = 246.22
    f = 1000.0     # 1 TeV decay constant -> xi = v^2/f^2 = 0.0606, kappa_V = sqrt(1-xi) = 0.969 (LHC-safe!)
    xi = (v / f) ** 2
    kappa_V = math.sqrt(1.0 - xi)
    ft_mchm = 2.0 / xi  # standard Barbieri-Giudice measure for pNGB misalignment

    # PDG 2024 alpha_i^-1(M_Z):
    a1_Z, a2_Z, a3_Z = 59.01, 29.58, 1.0 / 0.1180
    k = 1.0 / (2.0 * math.pi)

    # Pure SM beta coeffs (M_Z -> M_HC):
    b_SM = (41.0 / 10.0, -19.0 / 6.0, -7.0)
    t_hc = math.log(M_HC / M_Z)
    a1_hc = a1_Z - k * b_SM[0] * t_hc
    a2_hc = a2_Z - k * b_SM[1] * t_hc
    a3_hc = a3_Z - k * b_SM[2] * t_hc

    # Above M_HC = 2 TeV, add Sp(4) hyperfermions (dim = 4 of Sp(4), pseudoreal!):
    #   Under SM SU(3)_C x SU(2)_L x U(1)_Y:
    #     1 EW doublet (1, 2, 0) + 2 EW singlets (1, 1, +1/2) + (1, 1, -1/2)  (4 Weyl flavors -> SU(4)/Sp(4)!)
    #   Wait: check exact Pati-Salam embedding!
    #     Under SU(4)_C x SU(2)_L x SU(2)_R:
    #       psi_L ~ (1, 2, 1)  and  psi_R ~ (1, 1, 2)  (each in fundamental 4 of Sp(4)!)
    #     Check 1-loop beta shifts of 4 Weyl fundamentals of Sp(4):
    #       - For SU(2)_L: 4 Weyl doublets (T = 1/2) -> db_2L = (2/3) * 4 * (1/2) = +4/3!
    #       - For U(1)_Y (GUT norm 3/5): two singlets with Y = +-1/2, mult 4 each ->
    #         db_1 = (3/5) * (2/3) * 4 * ( (1/2)^2 + (-1/2)^2 ) = (3/5) * (4/3) = +4/5!
    #       - For SU(3)_C: color singlets -> db_3 = 0!
    #     Check Pati-Salam (above M_PS):
    #       - db_4C = 0 (color singlets)
    #       - db_2L = +4/3
    #       - db_2R = +4/3  (EXACT Left-Right symmetric!!)
    db_SM_HC = (4.0 / 5.0, 4.0 / 3.0, 0.0)
    # Note: below M_PS, the SM Higgs IS composite (not an elementary scalar above M_HC!),
    # so above M_HC we ALSO subtract the elementary SM Higgs doublet contribution (1/10, 1/6, 0)!
    # Let's test BOTH (with composite Higgs replacing elementary H, AND with extra scalar)!
    b_low_comp = (
        b_SM[0] - 1.0 / 10.0 + db_SM_HC[0],  # 4.1 - 0.1 + 0.8 = 4.8
        b_SM[1] - 1.0 / 6.0 + db_SM_HC[1],   # -19/6 - 1/6 + 8/6 = -2.0
        b_SM[2] + db_SM_HC[2],               # -7.0
    )

    # Above M_PS, Pati-Salam beta functions (our verified minimal PS content from earlier:
    #   3 gen fermions + Delta_R (1,1,3,10_bar) + Phi(1,2,2) composite or elementary):
    #   Baseline PS from verify_taucore_4simplex_spin10_bridge.py had:
    #     b_4C = -11*4/3 + 4 + 2 = -26/3? Let's recall exact minimal PS b_i:
    #     3 gen (4,2,1)+(4_bar,1,2): +4 to all (4C, 2L, 2R).
    #     1 complex bidoublet (1,2,2) [if composite, 0; if elementary, (0, 2/3, 2/3)]
    #     1 complex Delta_R (10_bar, 1, 3): T_4(10)=3, dim_2R=3 -> (1/3)*3*3 = +3 to b_4C;
    #                                       T_2R(3)=2, dim_4=10 -> (1/3)*2*10 = +20/3 to b_2R!
    #     Gauge: (-44/3, -22/3, -22/3).
    #     Total baseline (elementary Phi): b_4C = -23/3, b_2L = -8/3, b_2R = +10/3? Wait:
    #     Let's check: -44/3 + 12/3 + 9/3 = -23/3; -22/3 + 12/3 + 2/3 = -8/3; -22/3 + 12/3 + 2/3 + 20/3 = +12/3 = +4!
    #     Adding Sp(4) hyperfermions (0, +4/3, +4/3) and removing elementary Phi (0, -2/3, -2/3):
    #     b_PS_comp = (-23/3, -2, +14/3)!
    # Wait: let's solve the exact 2x2 linear system for (t_PS, t_U) where:
    #   Matching at M_PS: a_4 = a_3, a_2L = a_2, a_2R = (5/3) a_1 - (2/3) a_3
    #   Unification at M_U: a_4(M_U) == a_2L(M_U) == a_2R(M_U) == a_U^-1!
    def solve_2step(b_low, b_ps):
        b1, b2, b3 = b_low
        b4, b2L, b2R = b_ps
        # Let x = ln(M_PS / M_HC), y = ln(M_U / M_PS):
        # At M_PS:
        #   a4_p  = a3_hc - k b3 x
        #   a2L_p = a2_hc - k b2 x
        #   a2R_p = (5/3)(a1_hc - k b1 x) - (2/3)(a3_hc - k b3 x)
        # At M_U:
        #   a4_p - k b4 y == a2L_p - k b2L y  =>  k (b3 - b2) x + k (b4 - b2L) y = a3_hc - a2_hc
        #   a4_p - k b4 y == a2R_p - k b2R y  =>  k ( (5/3)b3 - (5/3)b1 ) x + k (b4 - b2R) y = (5/3)(a3_hc - a1_hc)
        A = np.array([
            [k * (b3 - b2), k * (b4 - b2L)],
            [k * (5.0 / 3.0) * (b3 - b1), k * (b4 - b2R)],
        ])
        rhs = np.array([a3_hc - a2_hc, (5.0 / 3.0) * (a3_hc - a1_hc)])
        x, y = np.linalg.solve(A, rhs)
        M_PS = M_HC * math.exp(x)
        M_U = M_PS * math.exp(y)
        aU_inv = a3_hc - k * b3 * x - k * b4 * y
        # Dim-6 proton lifetime scaling vs our benchmark (M_U=1.13e16, aU^-1=45.68 -> 1.43e36 yr):
        tau_p = 1.43e36 * ((M_U / 1.13e16) ** 4) * ((aU_inv / 45.68) ** 2)
        # Self-consistent Sp(4) confinement scale starting from EXACT alpha_HC(M_U) = alpha_U = 1 / aU_inv!
        # For Sp(4) (N=2) with 4 Weyl fundamentals: b_Sp4 = (11/3)*(2+1) - (2/3)*4*(1/2) = 11 - 4/3 = 29/3 = 9.6667!
        b_Sp4 = 29.0 / 3.0
        Lambda_Sp4 = M_U * math.exp(-2.0 * math.pi * aU_inv / b_Sp4)
        # Also check Sp(4) with 4 Weyl fundamentals + 1 antisymmetric 5 (for top partial compositeness, T(5)=1):
        #   Wait: 1 Weyl antisymmetric 5 shifts b_Sp4 by -(2/3)*1 = -2/3 -> b = 27/3 = 9.0 (or 2 Weyl -> 25/3 = 8.33).
        return dict(
            M_PS_GeV=f"{M_PS:.2e}",
            M_U_GeV=f"{M_U:.2e}",
            alpha_U_inv=round(float(aU_inv), 2),
            tau_p_yr=f"{tau_p:.2e}",
            passes_SuperK_2_4e34=bool(tau_p > 2.4e34),
            Sp4_b0=round(b_Sp4, 4),
            Sp4_self_consistent_Lambda_HC_GeV=f"{Lambda_Sp4:.2e}",
            Sp4_ratio_Lambda_over_1TeV=round(float(Lambda_Sp4 / 1000.0), 2),
        )

    # Case 1: Pure composite Higgs (replaces elementary H) + Sp(4) hyperfermions (1,2,1)+(1,1,2):
    b_PS_comp = (-23.0 / 3.0, -2.0, 14.0 / 3.0)
    res_comp = solve_2step(b_low_comp, b_PS_comp)

    # Case 2: Also check our baseline non-composite PS (M_PS=5.63e11, M_U=1.13e16, aU^-1=45.68) with an
    #         exact dark/spectator Sp(4) sector (showing what Lambda_Sp4 is at aU^-1 = 45.68!):
    Lambda_Sp4_at_baseline = 1.13e16 * math.exp(-2.0 * math.pi * 45.68 / (29.0 / 3.0))

    # Naive ETC top-mass NO-GO check:
    #   If top mass comes from dim-6 operator (1/M_PS^2) (Q_L t_R)(psi_HC psi_HC),
    #   m_t ~ (Lambda_HC^3 / M_PS^2) ~ (1e4)^3 / (5.63e11)^2 GeV:
    mt_naive_ETC_GeV = ((4.0 * math.pi * f) ** 3) / ((5.63e11) ** 2)

    return dict(
        LHC_safe_f_GeV=f,
        misalignment_xi=round(xi, 4),
        Higgs_coupling_kappa_V=round(kappa_V, 4),
        fine_tuning_Delta_MCHM=round(ft_mchm, 1),
        fine_tuning_improvement_orders=round(math.log10(1.42e25 / ft_mchm), 1),
        self_consistent_2step_unification_with_Sp4=res_comp,
        Sp4_Lambda_HC_at_baseline_aU45_68_GeV=round(Lambda_Sp4_at_baseline, 1),
        naive_ETC_top_mass_from_MPS_GeV=f"{mt_naive_ETC_GeV:.2e}",
        flavor_NOGO_caveat="PROVEN NO-GO for naive dim-6 ETC from M_PS (gives m_t ~ 6e-12 GeV!): generating m_t = 173 GeV requires Partial Compositeness (walking CFT anomalous dimension gamma ~ 1 between TeV and M_PS)!",
    )


# ---------------------------------------------------------------------------
# Candidate 3B: Cosmological Relaxion (GKR) Exact Cutoff Ceiling NO-GO
# ---------------------------------------------------------------------------
def audit_cosmological_relaxion():
    M_Pl = 2.435e18   # reduced Planck mass (GeV)
    v_EW = 246.22     # GeV
    Lambda_QCD = 0.1  # GeV
    M_PS = 5.63e11    # GeV
    M_U = 1.13e16     # GeV
    H_Starobinsky = 1.3e13  # GeV (Paper 24!)

    # Exact model-independent GKR relaxion bounds:
    #   (1) Subdominant energy density: H > M^2 / (sqrt(3) M_Pl)
    #   (2) Barrier forms during inflation (not washed out by Gibbons-Hawking T = H/(2 pi)): H < Lambda_br
    #   Combining (1) and (2):
    #       M < (sqrt(3) * M_Pl * Lambda_br)^{1/2}
    #   AND for QCD barrier with f_a >= M and delta_theta < 1e-10 (or even ignoring theta_QCD):
    #       classical beats quantum: H < (g M^3)^{1/3}, g M f_a ~ Lambda_QCD^4 -> M^5 < 3 sqrt(3) Lambda_QCD^4 M_Pl^3
    M_max_QCD = (3.0 * math.sqrt(3.0) * (Lambda_QCD ** 4) * (M_Pl ** 3)) ** (1.0 / 5.0)
    M_max_nonQCD_H_bound = math.sqrt(math.sqrt(3.0) * M_Pl * v_EW)
    H_min_at_MPS = (M_PS ** 2) / (math.sqrt(3.0) * M_Pl)

    assert M_max_nonQCD_H_bound < M_PS
    return dict(
        M_max_QCD_relaxion_GeV=f"{M_max_QCD:.2e}",
        M_max_nonQCD_relaxion_GeV=f"{M_max_nonQCD_H_bound:.2e}",
        TauCore_M_PS_GeV=f"{M_PS:.2e}",
        ratio_MPS_over_Mmax_nonQCD=round(M_PS / M_max_nonQCD_H_bound, 2),
        H_min_required_at_MPS_GeV=f"{H_min_at_MPS:.2e}",
        Starobinsky_H_inf_GeV=f"{H_Starobinsky:.2e}",
        verdict="PROVEN NO-GO: Even non-QCD Relaxion hits an absolute model-independent ceiling M_max = 3.22e10 GeV (17.5x BELOW M_PS = 5.63e11 GeV, and 3.5e5x below M_U!), AND H < 246 GeV conflicts with Starobinsky H_inf ~ 1.3e13 GeV by 11 orders of magnitude!",
    )


# ---------------------------------------------------------------------------
# Candidate 3C: Z_2 Twin Higgs (Mirror-Wall Copy) Quantitative Scope
# ---------------------------------------------------------------------------
def audit_twin_higgs():
    v = 246.22
    f_twin = 1000.0  # 1 TeV mirror-sector VEV (LHC-safe invisible Higgs BR < 11% -> v^2/f^2 ~ 0.06!)
    xi = (v / f_twin) ** 2
    Lambda_UV_twin = 4.0 * math.pi * f_twin  # ~ 12.6 TeV cutoff where Twin Higgs itself needs UV completion!
    return dict(
        f_twin_GeV=f_twin,
        fine_tuning_Delta_twin=round(2.0 / xi, 1),
        UV_cutoff_Lambda_twin_GeV=round(Lambda_UV_twin, 0),
        verdict="Partial solution ONLY: Z_2 mirror-wall Twin Higgs solves the 'little hierarchy' up to ~12.6 TeV without LHC colored top partners, but STILL requires Composite Higgs (3A) or SUSY above 13 TeV to reach M_PS and M_U!",
    )


def main():
    out = dict(
        Candidate_3A_Composite_Sp4_pNGB_Higgs=audit_composite_Sp4_higgs(),
        Candidate_3B_Cosmological_Relaxion_NOGO=audit_cosmological_relaxion(),
        Candidate_3C_Z2_Twin_Higgs=audit_twin_higgs(),
    )
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
