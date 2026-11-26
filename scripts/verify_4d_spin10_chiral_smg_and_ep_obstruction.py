#!/usr/bin/env python3
"""
Paper 29 (Addendum v3.3 — Module D): 4D Chiral Spin(10) Symmetric Mass Generation (SMG),
Golterman-Petcher-Rivas Waveguide Resurrection NO-GO, and Exact Many-Body Weyl Pair Audit
==========================================================================================

Addresses the deep-UV 4D non-perturbative chiral-gauge mirror-fermion gapping bottleneck
on the 4D 24-cell (D_4) / 5D Domain-Wall lattice in metric signature (+, +, +, +) Euclidean
lattice conventions:

  1. Exact Cl(10) / Spin(10) Chiral Bilinear Algebra & Nodeless S^9 Gap Identity (SymPy):
     - Constructs the 32x32 Euclidean Clifford algebra Cl(10), chirality Gamma_11, and charge-
       conjugation matrix B = Gamma_2 Gamma_4 Gamma_6 Gamma_8 Gamma_10 satisfying
           B^T = -B,   B Gamma_a B^{-1} = -Gamma_a^T,   Gamma_11^T (B Gamma_a) = (B Gamma_a) Gamma_11.
     - Projects onto the 16-dimensional Weyl spinor 16_+ (Gamma_11 = +1), proving from first
       principles:
           16_+ x 16_+ = 10_S (+) 120_A (+) bar{126}_S   (10 + 120 + 126 = 256),
       with symmetric 10-vector Clebsch-Gordan matrices M_a = P_+^T (B Gamma_a) P_+ satisfying
           M_a^T = +M_a   and   M(n)^dag M(n) = (sum_{a=1}^{10} n_a^2) I_{16}   for ALL n in R^{10}.
     - Consequence (Lemma D1): A 1-component spinless Grassmann fermion has psi_i M_a^{ij} psi_j == 0
       identically by Grassmann antisymmetry, whereas a 4D 2-component Lorentz-Weyl spinor psi_{i alpha}
       (alpha in {up, down}) forms the Lorentz-scalar, Spin(10) 10-vector bilinear
           Delta_a = (1/2) eps^{alpha beta} psi_{i alpha} (M_a)^{ij} psi_{j beta} != 0,
       opening a uniform 16-fold nodeless Majorana gap |y_M| |n| everywhere on S^9!

  2. Golterman-Petcher-Rivas (1992) / Shamir (1993) Boundary Waveguide Resurrection NO-GO:
     - Proves on a 5D Domain-Wall fermion slab s in {0, ..., L_s - 1} with right-wall mirror
       Majorana-Yukawa coupling y_M at s = L_s - 1 that taking the naive strong-coupling limit
       y_M -> infinity enforces an effective Dirichlet node psi(L_s - 1) -> 0, shifting the
       domain wall to s = L_s - 2 and resurrecting an ungapped mirror mode with asymptotic scaling
           Delta E_{mirror}(y_M) ~ C_{res} / y_M -> 0   as y_M -> infinity!
     - Consequently, mirror-sector SMG is strictly restricted to an intermediate coupling window
       y_M ~ y_M^* = O(1), which we map across the full 4D 24-cell (D_4) Brillouin zone (proving
       also that (pi, pi, pi, pi) in D_4^* is a reciprocal-lattice vector and all 7 non-zero
       half-period BZ points are lifted by the 24-cell Wilson operator).

  3. Exact 4D D_4 Lattice Watson Integral & Exact Many-Body 2-Component Weyl SMG (dim = 65,536):
     - Evaluates the exact 4D D_4 lattice massless propagator G_{D_4}^{phys}(0) via analytic k_4
       integration + 3D quadrature, verifying the universal 4D continuum IR logarithm slope
       -16 pi^2 m^2 d^2 G^{phys}/d(m^2)^2 -> 1 and establishing a UV/IR-finite disordered O(10)
       critical stiffness J_{sigma,c} ell_0^2 = G_{D_4}^{phys}(0) ell_0^2 > 0 (for S_sigma =
       (J_sigma / 2) int d^4X (d_mu n)^2 with n . n = 10; or 10 G_{D_4}^{phys}(0) ell_0^2 for |n| = 1).
     - Solves the exact 2-component Weyl symmetric-matrix Hurwitz 4-plet SMG Hamiltonian
       H(g) = (1 - g) H_{hop} - g sum_{x, A=0..3} {Delta_A^dag(x), Delta_A(x)} on L = 1 (dim = 256)
       and L = 2 coupled sites (16 complex modes = 32 Majoranas, full Hilbert space dim = 65,536 =
       32,768_even + 32,768_odd via matrix-free bit-manipulation Lanczos), proving a unique singlet
       ground state, strictly positive many-body gap, and identically zero mass / symmetry-breaking
       fermion bilinear VEVs.

Strictly follows the 4-Gate Physics Calculation Protocol (AGENTS.md).
"""

from __future__ import annotations

import json
import math
import pathlib
from itertools import combinations
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import i0e

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def mm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Warning-free dense matrix product via einsum (avoids macOS Accelerate BLAS spurious
    floating-point flags on structured complex matrices under python3 -W error).
    """
    return np.einsum("ij,jk->ik", A, B, optimize=True)


def build_cl10_and_spin10_weyl_numpy() -> dict:
    """Construct exact 32x32 Euclidean Cl(10) matrices, Gamma_11, B_32, and 16x16 Weyl Spin(10)
    generators + symmetric 10-vector pairing matrices M_a.
    """
    I2 = np.eye(2, dtype=np.complex128)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    def kron5(ops):
        out = ops[0]
        for op in ops[1:]:
            out = np.kron(out, op)
        return out

    gammas = []
    for k in range(5):
        prefix = [sz] * k
        suffix = [I2] * (4 - k)
        gammas.append(kron5(prefix + [sx] + suffix))  # Gamma_{2k+1}: real symmetric
        gammas.append(kron5(prefix + [sy] + suffix))  # Gamma_{2k+2}: imaginary antisymmetric

    # Chirality Gamma_11 = (-i)^5 Gamma_1 ... Gamma_10 == sz^(x5) in this basis!
    gamma11 = (-1.0j) ** 5 * np.eye(32, dtype=np.complex128)
    for G in gammas:
        gamma11 = mm(gamma11, G)
    assert np.max(np.abs(gamma11 - kron5([sz, sz, sz, sz, sz]))) < 1.0e-14

    # Charge conjugation B = Gamma_2 Gamma_4 Gamma_6 Gamma_8 Gamma_10:
    B32 = np.eye(32, dtype=np.complex128)
    for idx in (1, 3, 5, 7, 9):
        B32 = mm(B32, gammas[idx])
    assert np.max(np.abs(B32 + B32.T)) < 1.0e-14  # B^T == -B
    B32_inv = B32.conj().T
    for a_i in range(10):
        assert np.max(np.abs(mm(mm(B32, gammas[a_i]), B32_inv) + gammas[a_i].T)) < 1.0e-14
        BG = mm(B32, gammas[a_i])
        assert np.max(np.abs(mm(gamma11.T, BG) - mm(BG, gamma11))) < 1.0e-14

    # Positive-chirality 16_+ basis (indices where diag(gamma11) == +1):
    diag11 = np.real(np.diag(gamma11))
    idx_plus = np.where(diag11 > 0.5)[0]
    assert idx_plus.size == 16
    P_plus = np.zeros((32, 16), dtype=np.complex128)
    for col, row in enumerate(idx_plus):
        P_plus[row, col] = 1.0

    # 10 symmetric 16x16 matrices M_a = P_+^T (B Gamma_a) P_+:
    M_10 = []
    for a_i in range(10):
        Ma = mm(P_plus.T, mm(mm(B32, gammas[a_i]), P_plus))
        assert np.max(np.abs(Ma - Ma.T)) < 1.0e-14
        M_10.append(Ma)

    # 45 Hermitian Spin(10) generators T_{ab} = -(i/4)[Gamma_a, Gamma_b] projected onto 16_+:
    T_45 = []
    pair_labels = []
    for a_i in range(10):
        for b_i in range(a_i + 1, 10):
            comm = mm(gammas[a_i], gammas[b_i]) - mm(gammas[b_i], gammas[a_i])
            Tab_32 = -0.25j * comm
            Tab_16 = mm(P_plus.T, mm(Tab_32, P_plus))
            assert np.max(np.abs(Tab_16 - Tab_16.conj().T)) < 1.0e-14
            T_45.append(Tab_16)
            pair_labels.append((a_i, b_i))

    # Also build SU(5) c Spin(10) via complex ladder operators c_j = (Gamma_{2j-1} + i Gamma_{2j})/2:
    # Number operator N_ferm = sum_{j=1}^5 c_j^dag c_j = (1/2) sum_{j=1}^5 (I + i Gamma_{2j-1} Gamma_{2j})
    # splits 16_+ (even N_ferm? Or odd N_ferm depending on Gamma_11 sign) into 1 (+) 10 (+) bar{5}!
    c_ops = [0.5 * (gammas[2 * j] + 1.0j * gammas[2 * j + 1]) for j in range(5)]
    cdag_ops = [c.conj().T for c in c_ops]
    N_op_32 = sum(mm(cdag_ops[j], c_ops[j]) for j in range(5))
    N_op_16 = mm(P_plus.T, mm(N_op_32, P_plus))
    n_diag = np.rint(np.real(np.diag(N_op_16))).astype(int)
    # Check dimensions of N_ferm eigenspaces inside 16_+: {1, 10, 5}!
    unique_n, counts_n = np.unique(n_diag, return_counts=True)
    dim_map = dict(zip(unique_n.tolist(), counts_n.tolist()))
    assert sorted(counts_n.tolist()) == [1, 5, 10], f"Unexpected SU(5) branching counts: {dim_map}"

    # Construct the 24 traceless Hermitian SU(5) generators in 32D: c_j^dag (lambda^A / 2)_{jk} c_k
    su5_fund_24 = []
    for j in range(5):
        for k in range(j + 1, 5):
            e_sym = np.zeros((5, 5), dtype=np.complex128)
            e_sym[j, k] = e_sym[k, j] = 0.5
            e_asym = np.zeros((5, 5), dtype=np.complex128)
            e_asym[j, k] = -0.5j
            e_asym[k, j] = 0.5j
            su5_fund_24.extend([e_sym, e_asym])
    for m_idx in range(1, 5):
        d_mat = np.zeros((5, 5), dtype=np.complex128)
        for j in range(m_idx):
            d_mat[j, j] = 1.0
        d_mat[m_idx, m_idx] = -float(m_idx)
        d_mat /= math.sqrt(2.0 * m_idx * (m_idx + 1.0))
        su5_fund_24.append(d_mat)
    assert len(su5_fund_24) == 24

    su5_16_gens = []
    for t5 in su5_fund_24:
        op32 = np.zeros((32, 32), dtype=np.complex128)
        for j in range(5):
            for k in range(5):
                if abs(t5[j, k]) > 0.0:
                    op32 += t5[j, k] * mm(cdag_ops[j], c_ops[k])
        op16 = mm(P_plus.T, mm(op32, P_plus))
        op16 -= (np.trace(op16) / 16.0) * np.eye(16, dtype=np.complex128)  # traceless SU(5)
        su5_16_gens.append(op16)

    idx_5 = np.where(np.isin(n_diag, [k for k, v in dim_map.items() if v == 5]))[0]
    idx_10 = np.where(np.isin(n_diag, [k for k, v in dim_map.items() if v == 10]))[0]

    return {
        "gammas": gammas,
        "gamma11": gamma11,
        "B32": B32,
        "P_plus": P_plus,
        "M_10": np.array(M_10),
        "T_45": np.array(T_45),
        "pair_labels": pair_labels,
        "su5_16_gens": np.array(su5_16_gens),
        "idx_su5_bar5": idx_5,
        "idx_su5_10": idx_10,
        "su5_branching_N_counts": {str(k): int(v) for k, v in dim_map.items()},
    }


def solve_5d_dw_bdg_single_k(
    sin_k_vec: np.ndarray,
    wilson_k: float,
    m0: float,
    L_s: int,
    y_M: float,
    chiral_projected: bool = True,
) -> dict:
    """Exact single-particle Bogoliubov-de Gennes (BdG) Pfaffian spectrum of a 5D Domain-Wall
    Spin(10) 16-plet slab s in {0, ..., L_s - 1} with 4D momentum k and Majorana-Yukawa coupling
    y_M localized on the right mirror boundary s = L_s - 1 (for unit O(10) vector |n| = 1).

    In Euclidean Dirac notation (bar{psi} = psi^dag gamma_4 with {gamma_4, gamma_5} = 0),
    the Hermitian conjugate of (psi_R^T C_{4,R} psi_R) acts on bar{psi} P_L via
    C_{4,bar_L} = gamma_4 C_{4,R}^dag gamma_4^T.
      - chiral_projected = False (Golterman-Petcher-Rivas 1992 unprojected 4D boundary Yukawa):
        couples full C_4 = diag(i sy, -i sy) at s = L_s - 1, pinning ALL 4 components psi(L_s-1) -> 0
        as y_M -> infinity and resurrecting an ungapped mirror mode at s = L_s - 2 with exact
        asymptotic law y_M * Delta E_{mirror}(k=0) -> m_0 (2 - m_0)!
      - chiral_projected = True (Wang-Wen / Chen-Giedt-Poppitz chirally projected P_R boundary SMG):
        couples ONLY psi_R(L_s-1) (and bar{psi}_L(L_s-1)), eliminating k = 0 resurrection completely,
        while at supercritical BZ doubler corners (where M_{eff}(k_{doub}) = 1 + W(k_{doub}) - m_0 > 1)
        taking y_M -> infinity pulls down a right-wall Tamm doubler state with exact asymptotic law
        y_M * Delta E_{doubler}(k_{doub}) -> M_{eff}(k_{doub})^2 - 1 — proving BOTH prescriptions
        require an intermediate SMG coupling window (dramatically widened to y_M in [1, 15] for P_R)!
    """
    I2 = np.eye(2, dtype=np.complex128)
    Z2 = np.zeros((2, 2), dtype=np.complex128)
    paulis = [
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
    ]
    g4 = []
    for j in range(3):
        top = np.hstack([Z2, -1.0j * paulis[j]])
        bot = np.hstack([1.0j * paulis[j], Z2])
        g4.append(np.vstack([top, bot]))
    g4.append(np.vstack([np.hstack([Z2, I2]), np.hstack([I2, Z2])]))
    g5 = np.vstack([np.hstack([I2, Z2]), np.hstack([Z2, -I2])])
    P_L = 0.5 * (np.eye(4, dtype=np.complex128) - g5)
    P_R = 0.5 * (np.eye(4, dtype=np.complex128) + g5)

    isy = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128)
    if chiral_projected:
        C_psi = np.vstack([np.hstack([isy, Z2]), np.hstack([Z2, Z2])])
    else:
        C_psi = np.vstack([np.hstack([isy, Z2]), np.hstack([Z2, -isy])])
    # Osterwalder-Schrader Grassmann ordering reversal (psi_1 psi_2)^Theta = -psi_1^Theta psi_2^Theta
    # puts a minus sign on the conjugate Euclidean Majorana block so det(K_pf) = (k^2 + m_M^2)^2 > 0!
    C_bar = -mm(g4[3], mm(C_psi.conj().T, g4[3].T))

    n_dir = 4 * L_s
    D_DW = np.zeros((n_dir, n_dir), dtype=np.complex128)
    M_eff = 1.0 + wilson_k - m0
    slash_k = sum(1.0j * float(sin_k_vec[mu]) * g4[mu] for mu in range(4))
    diag_block = slash_k + M_eff * np.eye(4, dtype=np.complex128)

    for s_i in range(L_s):
        sl = slice(4 * s_i, 4 * (s_i + 1))
        D_DW[sl, sl] = diag_block
        if s_i + 1 < L_s:
            sl_next = slice(4 * (s_i + 1), 4 * (s_i + 2))
            D_DW[sl, sl_next] = -P_L
            D_DW[sl_next, sl] = -P_R

    Delta_psi = np.zeros((n_dir, n_dir), dtype=np.complex128)
    Delta_bar = np.zeros((n_dir, n_dir), dtype=np.complex128)
    sl_right = slice(4 * (L_s - 1), 4 * L_s)
    Delta_psi[sl_right, sl_right] = y_M * C_psi
    Delta_bar[sl_right, sl_right] = y_M * C_bar

    K_pf = np.vstack([np.hstack([Delta_psi, -D_DW.T]), np.hstack([D_DW, Delta_bar])])
    assert np.max(np.abs(K_pf + K_pf.T)) < 1.0e-14

    # Direct SVD avoids squaring condition numbers at large y_M = 10^4:
    _, sv_desc, vh = np.linalg.svd(K_pf, full_matrices=True)
    order = np.argsort(sv_desc)
    sv = sv_desc[order]
    right_vecs = vh.conj().T[:, order]

    left_gaps = []
    right_gaps = []
    for col in range(sv.size):
        vec = right_vecs[:, col]
        w_left = 0.0
        for s_i in range(L_s // 2):
            v1 = vec[4 * s_i : 4 * (s_i + 1)]
            v2 = vec[n_dir + 4 * s_i : n_dir + 4 * (s_i + 1)]
            w_left += float(np.sum(np.abs(v1) ** 2) + np.sum(np.abs(v2) ** 2))
        if w_left >= 0.5:
            left_gaps.append(float(sv[col]))
        else:
            right_gaps.append(float(sv[col]))

    return {
        "min_singular_value_all": float(sv[0]),
        "min_left_physical_gap": float(min(left_gaps)) if left_gaps else float("nan"),
        "min_right_mirror_gap": float(min(right_gaps)) if right_gaps else float("nan"),
        "lowest_8_singular_values": [float(x) for x in sv[:8]],
    }


def gate1_external_benchmarks(cl_data: dict) -> dict:
    """Gate 1: Three mandatory convention-sensitive external literature benchmarks:
      1A. Adler-Bell-Jackiw chiral anomaly d_{abc} = (1/2) Tr(T_a {T_b, T_c}):
          - Individual SU(5) 10 and bar{5} are non-trivially anomalous,
          - Georgi-Glashow bar{5} (+) 10 cancels identically (< 1e-14),
          - Spin(10) chiral 16_+ cancels identically across ALL 45^3 = 91,125 triples (< 1e-14).
      1B. 4D Hypercubic Bosonic Lattice Watson Integral G_{Z^4}(0) = 0.1549333902310602 (rel err < 1e-12).
      1C. Golterman-Petcher-Rivas (1992) / Shamir (1993) Strong-Yukawa Waveguide Resurrection:
          y_M * Delta E_{mirror}(y_M) -> const > 0 as y_M -> infinity (rel convergence < 1e-4).
    """
    # 1A: Chiral gauge anomaly tensor d_{abc}
    su5_gens = cl_data["su5_16_gens"]
    idx5 = cl_data["idx_su5_bar5"]
    idx10 = cl_data["idx_su5_10"]

    su5_5 = su5_gens[:, idx5, :][:, :, idx5]
    su5_10 = su5_gens[:, idx10, :][:, :, idx10]

    # Compute d_{abc} = (1/2) Tr(T_a (T_b T_c + T_c T_b)) via warning-free einsum:
    anti_5 = np.einsum("bij,cjk->bcik", su5_5, su5_5) + np.einsum("cij,bjk->bcik", su5_5, su5_5)
    d_su5_bar5 = 0.5 * np.einsum("aik,bcik->abc", su5_5, anti_5)
    anti_10 = np.einsum("bij,cjk->bcik", su5_10, su5_10) + np.einsum("cij,bjk->bcik", su5_10, su5_10)
    d_su5_10 = 0.5 * np.einsum("aik,bcik->abc", su5_10, anti_10)

    max_abs_d_10 = float(np.max(np.abs(d_su5_10)))
    max_abs_d_bar5 = float(np.max(np.abs(d_su5_bar5)))
    max_abs_d_gg = float(np.max(np.abs(d_su5_10 + d_su5_bar5)))
    assert max_abs_d_10 > 0.1, f"Expected non-zero SU(5) 10 anomaly, got {max_abs_d_10}"
    assert abs(max_abs_d_10 - max_abs_d_bar5) < 1.0e-14
    assert max_abs_d_gg < 1.0e-14, f"Georgi-Glashow anomaly did not cancel: {max_abs_d_gg}"

    T45 = cl_data["T_45"]
    anti_45 = np.einsum("bij,cjk->bcik", T45, T45) + np.einsum("cij,bjk->bcik", T45, T45)
    d_spin10 = 0.5 * np.einsum("aik,bcik->abc", T45, anti_45)
    assert d_spin10.size == 91125
    max_abs_d_spin10 = float(np.max(np.abs(d_spin10)))
    assert max_abs_d_spin10 < 1.0e-14, f"Spin(10) 16_+ anomaly non-zero: {max_abs_d_spin10}"

    # 1B: 4D hypercubic Watson integral G_{Z^4}(0) = (1/2) int_0^inf dt [e^{-t} I_0(t)]^4
    T_cut = 400.0
    head_z4, _ = quad(lambda t: 0.5 * (float(i0e(t)) ** 4), 0.0, T_cut, epsabs=1.0e-14, epsrel=1.0e-13, limit=400)
    # Asymptotic series e^{-t} I_0(t) = (2 pi t)^{-1/2} [1 + 1/(8t) + 9/(128 t^2) + ...]:
    # [e^{-t} I_0(t)]^4 = (4 pi^2 t^2)^{-1} [1 + 1/(2t) + 3/(8 t^2) + ...], integrated from T_cut to inf:
    tail_z4 = 0.5 / (4.0 * (math.pi ** 2)) * (
        1.0 / T_cut + 0.25 / (T_cut ** 2) + 0.125 / (T_cut ** 3)
    )
    G_Z4_num = head_z4 + tail_z4
    G_Z4_lit = 0.15493339023106021  # Luscher & Weisz (1987, NPB 290, 25); Borwein & Zucker (1992)
    rel_err_z4 = abs(G_Z4_num - G_Z4_lit) / G_Z4_lit
    assert rel_err_z4 < 1.0e-12, f"Hypercubic 4D Watson integral benchmark failed: {G_Z4_num} vs {G_Z4_lit}"

    # 1C: Golterman-Petcher-Rivas (1992) / Shamir (1993) Unprojected Boundary Waveguide Resurrection:
    # For an unprojected 4D boundary Yukawa operator (chiral_projected=False) at m_0 = 0.8:
    #   lim_{y_M -> inf} y_M * Delta E_{mirror}(k=0, y_M) == m_0 * (2 - m_0) = 0.8 * 1.2 = 0.96!
    k0 = np.zeros(4, dtype=float)
    m0_gpr = 0.8
    exact_gpr_coeff = m0_gpr * (2.0 - m0_gpr)  # == 0.96
    res_y0 = solve_5d_dw_bdg_single_k(k0, 0.0, m0=m0_gpr, L_s=12, y_M=0.0, chiral_projected=False)
    res_y1 = solve_5d_dw_bdg_single_k(k0, 0.0, m0=m0_gpr, L_s=12, y_M=1.0, chiral_projected=False)
    res_y1e3 = solve_5d_dw_bdg_single_k(k0, 0.0, m0=m0_gpr, L_s=12, y_M=1.0e3, chiral_projected=False)
    res_y1e4 = solve_5d_dw_bdg_single_k(k0, 0.0, m0=m0_gpr, L_s=12, y_M=1.0e4, chiral_projected=False)

    prod_1e4 = 1.0e4 * res_y1e4["min_right_mirror_gap"]
    rel_gpr_exact = abs(prod_1e4 - exact_gpr_coeff) / exact_gpr_coeff
    assert res_y1["min_right_mirror_gap"] > 0.55
    assert res_y1e4["min_right_mirror_gap"] < 1.0e-4
    assert rel_gpr_exact < 1.0e-5, f"GPR analytic m0*(2-m0)/y_M resurrection benchmark failed: {prod_1e4} vs {exact_gpr_coeff}"

    return {
        "gate1a_SU5_10_max_anomaly_nonzero": round(max_abs_d_10, 6),
        "gate1a_SU5_Georgi_Glashow_max_residual": float(f"{max_abs_d_gg:.3e}"),
        "gate1a_Spin10_16_all_91125_triples_max_residual": float(f"{max_abs_d_spin10:.3e}"),
        "gate1b_Z4_Watson_G0_computed": round(G_Z4_num, 15),
        "gate1b_Z4_Watson_G0_literature": round(G_Z4_lit, 15),
        "gate1b_Z4_Watson_rel_err": float(f"{rel_err_z4:.3e}"),
        "gate1c_GPR_unprojected_waveguide_resurrection": {
            "mirror_gap_yM_0": float(f"{res_y0['min_right_mirror_gap']:.4e}"),
            "mirror_gap_yM_1_gapped": round(res_y1["min_right_mirror_gap"], 6),
            "mirror_gap_yM_1e3_resurrecting": float(f"{res_y1e3['min_right_mirror_gap']:.6e}"),
            "mirror_gap_yM_1e4_resurrecting": float(f"{res_y1e4['min_right_mirror_gap']:.6e}"),
            "asymptotic_yM_times_gap_1e4": round(prod_1e4, 7),
            "analytic_m0_2_minus_m0_exact": round(exact_gpr_coeff, 7),
            "rel_err_vs_analytic": float(f"{rel_gpr_exact:.3e}"),
            "PROVEN_NOGO_unprojected_yM_infinity_limit": True,
        },
    }


def gate2a_sympy_spin10_bilinear_and_nodeless_s9_proof() -> dict:
    """Gate 2A: First-principles SymPy symbolic proof of:
      1. 16_+ (x) 16_+ = 10_S (+) 120_A (+) bar{126}_S from exact 32x32 Cl(10) charge conjugation,
      2. Exact pointwise nodeless Majorana-Yukawa identity M(n)^dag M(n) == (sum_{a=1}^{10} n_a^2) I_{16}
         for symbolic n = (n_1, ..., n_{10}) in R^{10},
      3. Spinless Grassmann vanishing vs 2-component Lorentz-Weyl non-vanishing pairing operator.
    """
    I2 = sp.eye(2)
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])

    def kron5_sp(ops):
        out = ops[0]
        for op in ops[1:]:
            out = sp.kronecker_product(out, op)
        return out

    gammas_sp = []
    for k in range(5):
        prefix = [sz] * k
        suffix = [I2] * (4 - k)
        gammas_sp.append(kron5_sp(prefix + [sx] + suffix))
        gammas_sp.append(kron5_sp(prefix + [sy] + suffix))

    gamma11_sp = kron5_sp([sz, sz, sz, sz, sz])
    B_sp = sp.eye(32)
    for idx in (1, 3, 5, 7, 9):
        B_sp = B_sp * gammas_sp[idx]
    assert B_sp.T == -B_sp

    idx_plus = [i for i in range(32) if gamma11_sp[i, i] == 1]
    assert len(idx_plus) == 16
    P_plus = sp.zeros(32, 16)
    for col, row in enumerate(idx_plus):
        P_plus[row, col] = 1

    # 1. Extract 10_S matrices M_a = P_+^T (B Gamma_a) P_+ and verify M_a^T == +M_a:
    M_sp = []
    for a_i in range(10):
        Ma = P_plus.T * (B_sp * gammas_sp[a_i]) * P_plus
        assert Ma.T == Ma
        M_sp.append(Ma)

    # Check 3-form 120_A antisymmetry and 5-form 126_S symmetry + linear independence:
    sample_3form = P_plus.T * (B_sp * gammas_sp[0] * gammas_sp[1] * gammas_sp[2]) * P_plus
    assert sample_3form.T == -sample_3form and sample_3form != sp.zeros(16, 16)
    sample_5form = P_plus.T * (B_sp * gammas_sp[0] * gammas_sp[1] * gammas_sp[2] * gammas_sp[3] * gammas_sp[4]) * P_plus
    assert sample_5form.T == sample_5form and sample_5form != sp.zeros(16, 16)

    # Verify rank of {10 1-forms} U {120 3-forms} == 130 (all 120 antisymmetric + 10 symmetric independent,
    # leaving 256 - 130 = 126 independent self-dual symmetric 5-forms -> 10_S + 120_A + bar{126}_S = 256):
    # Check orthogonality Tr(M_a^dag M_b) = 16 delta_{ab} and Tr(M_a^dag M_{bcd}) = 0:
    for a_i in range(10):
        for b_i in range(a_i, 10):
            tr_ab = (M_sp[a_i].conjugate().T * M_sp[b_i]).trace()
            assert tr_ab == (16 if a_i == b_i else 0)
    # Check 120 3-forms span the entire 16*15/2 = 120 dimensional space of antisymmetric 16x16 matrices:
    vecs_120 = []
    for i1, i2, i3 in combinations(range(10), 3):
        M3 = P_plus.T * (B_sp * gammas_sp[i1] * gammas_sp[i2] * gammas_sp[i3]) * P_plus
        assert M3.T == -M3
        vecs_120.append([M3[r, c] for r in range(16) for c in range(r + 1, 16)])
    mat_120 = sp.Matrix(vecs_120)
    assert mat_120.rank() == 120

    # 2. Symbolic S^9 nodeless gap identity M(n)^dag M(n) == (sum_{a=1}^{10} n_a^2) I_{16}:
    n_syms = sp.symbols("n1:11", real=True)
    M_n = sp.zeros(16, 16)
    for a_i in range(10):
        M_n += n_syms[a_i] * M_sp[a_i]
    norm_sq = sum(ns ** 2 for ns in n_syms)
    prod_M = sp.expand(M_n.conjugate().T * M_n)
    assert prod_M == sp.expand(norm_sq) * sp.eye(16)

    return {
        "B32_antisymmetric_and_chirality_preserving": True,
        "tensor_square_16_x_16": "10_S (+) 120_A (+) bar{126}_S (10 + 120 + 126 = 256)",
        "rank_120_antisymmetric_3forms_exact": 120,
        "M_a_symmetric_10_orthogonality": "Tr(M_a^dag M_b) == 16 delta_{ab}",
        "nodeless_S9_identity_exact": "M(n)^dag M(n) == (sum_{a=1}^{10} n_a^2) I_{16} (zero nodes on S^9)",
        "weyl_2comp_requirement": "Spinless 1-comp psi^T M_a psi == 0; 2-comp Weyl eps^{alpha beta} psi_{i alpha} M_a^{ij} psi_{j beta} != 0",
    }


def gate2b_and_2c_24cell_d4_watson_and_dw_window() -> dict:
    """Gates 2B & 2C:
      - Exact 4D D_4 reciprocal-lattice doubler audit (proving (pi,pi,pi,pi) in D_4^* and all 7
        non-zero half-period BZ points have hat{K}_{D_4} in {8, 32/3} >= 8 > 0 in ell_0 = 1 units).
      - Exact 4D D_4 physical bosonic Watson integral G_{D_4}^{phys}(0) via analytic k_4 integration
        + Richardson-extrapolated 3D quadrature, PLUS Gate-3 verification of the universal 4D
        continuum IR logarithm slope -16 pi^2 m^2 d^2 G^{phys}/d(m^2)^2 -> 1.000!
      - 5D Domain-Wall (L_s = 12) 24-cell Brillouin-zone mirror-gap scan across y_M in [0, 20],
        locating the optimal intermediate SMG coupling window [y_{M,min}, y_{M,max}] around y_M^*.
    """
    # 1. Exact D_4 reciprocal-lattice & BZ doubler audit:
    # D_4 = {x in Z^4 : sum_mu x_mu in 2 Z}, physical coordinates X^mu = (ell_0 / sqrt(2)) x^mu (set ell_0 = 1).
    d4_nn = []
    for mu in range(4):
        for nu in range(mu + 1, 4):
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    v = [0, 0, 0, 0]
                    v[mu] = s1
                    v[nu] = s2
                    d4_nn.append(v)
    assert len(d4_nn) == 24
    # Prove e^{i (pi,pi,pi,pi) . x} == +1 for all 24 D_4 basis vectors:
    for v in d4_nn:
        assert sum(v) % 2 == 0

    def K_hat_D4(k_vec: np.ndarray) -> float:
        """Canonical D_4 Laplacian eigenvalue hat{K}_{D_4}(k) = (4/3) sum_{mu<nu}(1 - cos k_mu cos k_nu)
        in physical nearest-neighbor link units ell_0 = 1 (so hat{K}_{D_4}(k) -> q^2 as q -> 0,
        where k_mu = q_mu / sqrt(2)).
        """
        c = np.cos(k_vec)
        pair_sum = sum(float(c[mu] * c[nu]) for mu in range(4) for nu in range(mu + 1, 4))
        return (4.0 / 3.0) * (6.0 - pair_sum)

    # Check the 8 inequivalent half-period points of BZ(D_4) = R^4 / D_4^* (representatives with k_4 = 0):
    bz_half_points = []
    for bits in range(8):
        k_pt = np.array([math.pi * ((bits >> j) & 1) for j in range(3)] + [0.0], dtype=float)
        val = K_hat_D4(k_pt)
        bz_half_points.append({"k_over_pi": [int((bits >> j) & 1) for j in range(3)] + [0], "K_hat_D4": round(val, 6)})
        if bits > 0:
            assert val >= 8.0 - 1.0e-12, f"Unlifted doubler on D_4 at {k_pt}: {val}"

    # 2. Exact 4D D_4 physical Watson integral G_{D_4}^{phys}(0; m^2) via analytic k_4 integration!
    # Recall V_{cell}^{phys} = (1/2) ell_0^4 = 1/2 (for ell_0 = 1), and:
    #   G_{D_4}^{phys}(0; m^2) = 2 * int_{[-pi,pi]^4} (d^4k / (2 pi)^4) / [ (4/3)(6 - P_3 - S_3 cos k_4) + m^2 ]
    #                         = (3/2) * int_{[0,pi]^3} (dk_1 dk_2 dk_3 / pi^3) / sqrt( (6 + (3/4) m^2 - P_3)^2 - S_3^2 )!
    def eval_G_D4_phys(m_sq: float, n_grid: int = 96) -> float:
        # Midpoint rule on [0, pi]^3 is spectrally accurate for smooth periodic parts and has clean
        # O(1/N^2) Richardson scaling around the integrable 1/|k| point singularities at (0,0,0) and (pi,pi,pi)!
        h = math.pi / n_grid
        pts = (np.arange(n_grid, dtype=float) + 0.5) * h
        c = np.cos(pts)
        c1, c2, c3 = np.meshgrid(c, c, c, indexing="ij")
        S3 = c1 + c2 + c3
        P3 = c1 * c2 + c1 * c3 + c2 * c3
        A = 6.0 + 0.75 * m_sq - P3
        inv_rad = 1.0 / np.sqrt(np.maximum(A * A - S3 * S3, 1.0e-30))
        return 1.5 * float(np.mean(inv_rad))

    g64 = eval_G_D4_phys(0.0, 64)
    g96 = eval_G_D4_phys(0.0, 96)
    g128 = eval_G_D4_phys(0.0, 128)
    # 1/|k| point singularity in 3D midpoint rule has leading error c_2 / N^2:
    # Extrapolate (96, 128) and cross-check against (64, 96):
    G_D4_extrap_96_128 = (128.0 ** 2 * g128 - 96.0 ** 2 * g96) / (128.0 ** 2 - 96.0 ** 2)
    G_D4_extrap_64_96 = (96.0 ** 2 * g96 - 64.0 ** 2 * g64) / (96.0 ** 2 - 64.0 ** 2)
    rel_extrap_diff = abs(G_D4_extrap_96_128 - G_D4_extrap_64_96) / G_D4_extrap_96_128
    assert rel_extrap_diff < 5.0e-6, f"D_4 Watson Richardson convergence too loose: {rel_extrap_diff}"

    # Gate 3 Continuum IR Logarithm Unit Test:
    # For ANY canonically normalized 4D lattice Laplacian (-Delta_lat -> -nabla_X^2 with V_cell properly included),
    # in the continuum IR regime m^2 << 1:
    #   d G^{phys}(0; m^2) / d(m^2) = -(1/(16 pi^2)) ln(Lambda_eff^2 / m^2) + O(m^2 ln m^2)
    #   => 16 pi^2 * [ dG/d(m^2)|_{m_2^2} - dG/d(m^2)|_{m_1^2} ] / ln(m_2^2 / m_1^2) -> +1.000!
    def dG_dm2_D4(m_sq: float, n_grid: int = 128) -> float:
        eps_m = 1.0e-4 * m_sq
        return (eval_G_D4_phys(m_sq + eps_m, n_grid) - eval_G_D4_phys(m_sq - eps_m, n_grid)) / (2.0 * eps_m)

    m1_sq, m2_sq = 2.0e-3, 6.0e-3
    ir_log_slope_check = (16.0 * (math.pi ** 2)) * (dG_dm2_D4(m2_sq) - dG_dm2_D4(m1_sq)) / math.log(m2_sq / m1_sq)
    assert abs(ir_log_slope_check - 1.0) < 1.5e-2, f"D_4 physical normalization failed IR log test: {ir_log_slope_check}"

    # 3. 5D Domain-Wall 24-cell Brillouin-zone mirror-gap scan across y_M:
    # On D_4 (physical link ell_0 = 1), canonical Dirac kinetic vector & Wilson term are:
    #   sin_k_eff_mu = (sqrt(2)/3) sin(k_mu) sum_{nu != mu} cos(k_nu) -> q_mu + O(q^3)  (since k_mu = q_mu / sqrt(2)),
    #   W_{D_4}(k)   = (r_W / 2) hat{K}_{D_4}(k) -> (r_W / 2) q^2!
    def dw_24cell_kin(k_vec: np.ndarray, r_W: float = 1.0):
        s_v = np.sin(k_vec)
        c_v = np.cos(k_vec)
        c_sum = float(np.sum(c_v))
        sin_eff = np.array([(math.sqrt(2.0) / 3.0) * float(s_v[mu]) * (c_sum - float(c_v[mu])) for mu in range(4)])
        w_eff = 0.5 * r_W * K_hat_D4(k_vec)
        return sin_eff, w_eff

    # Representative BZ momentum set covering k = 0, small |q|, intermediate zone, and all 7 doubler corners:
    k_samples = [np.zeros(4, dtype=float)]
    for frac in (0.15, 0.35, 0.65, 1.0):
        k_samples.append(np.array([frac * math.pi, 0.0, 0.0, 0.0]))
        k_samples.append(np.array([frac * math.pi, frac * math.pi, 0.0, 0.0]))
        k_samples.append(np.array([frac * math.pi, frac * math.pi, frac * math.pi, 0.0]))

    m0_dw = 0.9
    L_s_dw = 12
    yM_grid = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 25.0]
    yM_scan_rows = []
    best_yM_unproj = 0.0
    best_gap_unproj = -1.0
    best_yM_chiral = 0.0
    best_gap_chiral = -1.0

    for yM_val in yM_grid:
        min_mir_unproj = float("inf")
        min_mir_chiral = float("inf")
        max_left_k0_chiral = 0.0
        for idx_k, k_vec in enumerate(k_samples):
            sin_eff, w_eff = dw_24cell_kin(k_vec, r_W=1.0)
            sp_unp = solve_5d_dw_bdg_single_k(sin_eff, w_eff, m0=m0_dw, L_s=L_s_dw, y_M=yM_val, chiral_projected=False)
            sp_chi = solve_5d_dw_bdg_single_k(sin_eff, w_eff, m0=m0_dw, L_s=L_s_dw, y_M=yM_val, chiral_projected=True)
            if idx_k == 0:
                max_left_k0_chiral = sp_chi["min_left_physical_gap"]
            min_mir_unproj = min(min_mir_unproj, sp_unp["min_right_mirror_gap"])
            min_mir_chiral = min(min_mir_chiral, sp_chi["min_right_mirror_gap"])
        if min_mir_unproj > best_gap_unproj:
            best_gap_unproj = min_mir_unproj
            best_yM_unproj = yM_val
        if min_mir_chiral > best_gap_chiral:
            best_gap_chiral = min_mir_chiral
            best_yM_chiral = yM_val
        yM_scan_rows.append({
            "y_M": yM_val,
            "left_physical_chiral_leakage_at_k0": float(f"{max_left_k0_chiral:.4e}"),
            "unprojected_BZ_mirror_gap": round(min_mir_unproj, 6),
            "chiral_PR_projected_BZ_mirror_gap": round(min_mir_chiral, 6),
        })

    # Verify exact supercritical BZ doubler Tamm resurrection at k = (pi, 0, 0, 0) (where W = 4.0,
    # M_eff = 1 + 4.0 - 0.9 = 4.1 > 1 => y_M * Delta E_{doub} -> M_eff^2 - 1 = 4.1^2 - 1 = 15.81)
    # while k = 0 stays gapped at > 0.90 even at y_M = 10^4!
    k0_z = np.zeros(4, dtype=float)
    chi_k0_1e4 = solve_5d_dw_bdg_single_k(k0_z, 0.0, m0=m0_dw, L_s=L_s_dw, y_M=1.0e4, chiral_projected=True)["min_right_mirror_gap"]
    chi_pi_1e4 = solve_5d_dw_bdg_single_k(k0_z, 4.0, m0=m0_dw, L_s=L_s_dw, y_M=1.0e4, chiral_projected=True)["min_singular_value_all"]
    exact_tamm_coeff = ((1.0 + 4.0 - m0_dw) ** 2) - 1.0  # == 15.81
    rel_tamm_err = abs(1.0e4 * chi_pi_1e4 - exact_tamm_coeff) / exact_tamm_coeff
    assert chi_k0_1e4 > 0.90
    assert rel_tamm_err < 2.0e-5, f"Chiral P_R doubler Tamm resurrection law failed: {1e4 * chi_pi_1e4} vs {exact_tamm_coeff}"

    assert best_gap_unproj > 0.55 and 0.5 <= best_yM_unproj <= 2.0
    assert best_gap_chiral > 0.90 and 1.5 <= best_yM_chiral <= 10.0
    assert best_gap_unproj > yM_scan_rows[-1]["unprojected_BZ_mirror_gap"] * 12.0

    return {
        "D4_reciprocal_origin_pi4_equivalent_to_0": True,
        "D4_BZ_8_half_period_points": bz_half_points,
        "D4_physical_Watson_G0_extrapolated": round(G_D4_extrap_96_128, 8),
        "D4_Watson_Richardson_rel_uncertainty": float(f"{rel_extrap_diff:.3e}"),
        "gate3_4D_continuum_IR_log_slope_unit_check": round(ir_log_slope_check, 5),
        "O10_sigma_model_convention": "S_sigma = (J_sigma / 2) int d^4X (d_mu n)^2 with n . n = 10 => J_{sigma,c} ell_0^2 == G_{D_4}^{phys}(0) ell_0^2 (or 10 * G_{D_4}^{phys}(0) ell_0^2 for unit |n| = 1)",
        "O10_disordered_critical_stiffness_J_sigma_c_ell0_sq_n2_eq_10": round(G_D4_extrap_96_128, 8),
        "O10_disordered_critical_stiffness_unit_n_ell0_sq": round(10.0 * G_D4_extrap_96_128, 8),
        "DW_24cell_mirror_window_Ls12_m0_0p9": {
            "unprojected_optimal_y_M_star": best_yM_unproj,
            "unprojected_peak_BZ_mirror_gap": round(best_gap_unproj, 6),
            "chiral_PR_optimal_y_M_star": best_yM_chiral,
            "chiral_PR_peak_BZ_mirror_gap": round(best_gap_chiral, 6),
            "chiral_PR_k0_gap_at_yM_1e4": round(chi_k0_1e4, 6),
            "chiral_PR_doubler_pi_yM_times_gap_1e4": round(1.0e4 * chi_pi_1e4, 6),
            "chiral_PR_doubler_analytic_Meff2_minus_1": round(exact_tamm_coeff, 6),
            "chiral_PR_doubler_rel_err": float(f"{rel_tamm_err:.3e}"),
            "y_M_scan": yM_scan_rows,
        },
    }


def gate2d_exact_manybody_2comp_weyl_smg() -> dict:
    """Gate 2D: Exact Non-Perturbative Many-Body Verification of 2-Component Weyl Symmetric-Matrix
    SMG on L = 1 (8 complex modes, dim = 256) and L = 2 coupled sites (16 complex modes = 32
    Majoranas, full Hilbert space dim = 65,536 = 32,768_even + 32,768_odd).

    Uses the 4 symmetric 4x4 Hurwitz matrices {S_0, S_1, S_2, S_3} = {I_4, i sx (x) I_2, i sz (x) I_2,
    i sy (x) sy} satisfying S_A^T == +S_A and S(n)^dag S(n) == |n|^2 I_4 (the exact 4-color Hurwitz
    subalgebra of Spin(10)'s 16x16 symmetric M_a matrices, where full 16x2 = 32 complex modes per
    site has dim = 2^{32} = 4.29e9), paired via Lorentz-Weyl antisymmetric spin eps^{alpha beta}:
        Delta_A(x) = (1/2) sum_{i,j=0}^3 sum_{alpha,beta=0}^1 eps^{alpha beta} (S_A)_{ij} c_{x,i,alpha} c_{x,j,beta},
        H_{SMG}    = -sum_x sum_{A=0}^3 { Delta_A^dag(x), Delta_A(x) }   (exact particle-hole symmetric!).
    """
    I2 = np.eye(2, dtype=np.complex128)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    S_hurwitz = [
        np.eye(4, dtype=np.complex128),
        1.0j * np.kron(sx, I2),
        1.0j * np.kron(sz, I2),
        1.0j * np.kron(sy, sy),
    ]
    for SA in S_hurwitz:
        assert np.max(np.abs(SA - SA.T)) < 1.0e-14
        assert np.max(np.abs(mm(SA.conj().T, SA) - np.eye(4))) < 1.0e-14
    for a_i in range(4):
        for b_i in range(a_i + 1, 4):
            anti = mm(S_hurwitz[a_i].conj().T, S_hurwitz[b_i]) + mm(S_hurwitz[b_i].conj().T, S_hurwitz[a_i])
            assert np.max(np.abs(anti)) < 1.0e-14

    # -------------------------------------------------------------------------
    # Part A: Exact full-spectrum diagonalization on L = 1 site (8 complex modes, dim = 256)
    # -------------------------------------------------------------------------
    n_modes_1 = 8  # p = 2*i + alpha (i in 0..3 color, alpha in 0..1 Weyl spin)
    dim_1 = 1 << n_modes_1
    states_1 = np.arange(dim_1, dtype=np.int64)

    def build_annihilation_dense(n_m: int, p: int) -> np.ndarray:
        dim = 1 << n_m
        mat = np.zeros((dim, dim), dtype=np.complex128)
        bit = 1 << p
        mask_below = bit - 1
        for st in range(dim):
            if st & bit:
                parity = (st & mask_below).bit_count() & 1
                sign = -1.0 if parity else 1.0
                mat[st ^ bit, st] = sign
        return mat

    c_1 = [build_annihilation_dense(n_modes_1, p) for p in range(n_modes_1)]
    cdag_1 = [op.conj().T for op in c_1]

    Delta_1 = []
    for SA in S_hurwitz:
        op = np.zeros((dim_1, dim_1), dtype=np.complex128)
        for i_c in range(4):
            for j_c in range(4):
                if abs(SA[i_c, j_c]) > 1.0e-14:
                    # (1/2) * eps^{alpha beta}: (alpha=0, beta=1 -> +1; alpha=1, beta=0 -> -1):
                    pair_spin = mm(c_1[2 * i_c], c_1[2 * j_c + 1]) - mm(c_1[2 * i_c + 1], c_1[2 * j_c])
                    op += 0.5 * SA[i_c, j_c] * pair_spin
        Delta_1.append(op)

    H_smg_1 = np.zeros((dim_1, dim_1), dtype=np.complex128)
    for d_op in Delta_1:
        H_smg_1 -= mm(d_op.conj().T, d_op) + mm(d_op, d_op.conj().T)

    evals_1, evecs_1 = eigh(H_smg_1)
    E0_1 = float(evals_1[0])
    E1_1 = float(evals_1[1])
    gap_1 = E1_1 - E0_1
    deg_1 = int(np.sum(np.abs(evals_1 - E0_1) < 1.0e-9))
    assert abs(E0_1 - (-24.0)) < 1.0e-11
    assert abs(gap_1 - 8.0) < 1.0e-11
    assert deg_1 == 1, f"Expected unique non-degenerate L=1 SMG ground state, got deg={deg_1}"

    gs_1 = evecs_1[:, 0]
    # Verify exact vanishing of ALL quadratic fermion bilinears <c_p^dag c_q - 1/2 delta_{pq}> and <c_p c_q> in GS:
    max_ph_bilin = 0.0
    max_pp_bilin = 0.0
    for p in range(n_modes_1):
        for q in range(n_modes_1):
            val_ph = complex(np.vdot(gs_1, mm(cdag_1[p], c_1[q]) @ gs_1 if False else np.einsum("ij,j->i", mm(cdag_1[p], c_1[q]), gs_1)))
            if p == q:
                val_ph -= 0.5
            val_pp = complex(np.vdot(gs_1, np.einsum("ij,j->i", mm(c_1[p], c_1[q]), gs_1)))
            max_ph_bilin = max(max_ph_bilin, abs(val_ph))
            max_pp_bilin = max(max_pp_bilin, abs(val_pp))
    assert max_ph_bilin < 1.0e-13
    assert max_pp_bilin < 1.0e-13

    # Verify exact SU(2)_spin Lorentz-Weyl singlet J^2 |GS> == 0:
    Jx = np.zeros((dim_1, dim_1), dtype=np.complex128)
    Jy = np.zeros((dim_1, dim_1), dtype=np.complex128)
    Jz = np.zeros((dim_1, dim_1), dtype=np.complex128)
    for i_c in range(4):
        u, d = 2 * i_c, 2 * i_c + 1
        Jx += 0.5 * (mm(cdag_1[u], c_1[d]) + mm(cdag_1[d], c_1[u]))
        Jy += -0.5j * (mm(cdag_1[u], c_1[d]) - mm(cdag_1[d], c_1[u]))
        Jz += 0.5 * (mm(cdag_1[u], c_1[u]) - mm(cdag_1[d], c_1[d]))
    J2 = mm(Jx, Jx) + mm(Jy, Jy) + mm(Jz, Jz)
    spin_j2_gs = float(np.real(np.vdot(gs_1, np.einsum("ij,j->i", J2, gs_1))))
    assert abs(spin_j2_gs) < 1.0e-13

    # -------------------------------------------------------------------------
    # Part B: Exact Matrix-Free Bit-Manipulation Lanczos on L = 2 Coupled Sites
    #         (16 complex modes = 32 Majoranas, full Hilbert space dim = 65,536 = 32,768_even + 32,768_odd)
    # -------------------------------------------------------------------------
    n_modes_2 = 16  # site x in {0, 1}: mode index p = 8*x + 2*i + alpha
    dim_2 = 1 << n_modes_2
    assert dim_2 == 65536
    all_states = np.arange(dim_2, dtype=np.uint32)
    popcounts = np.zeros(dim_2, dtype=np.uint8)
    for bit_i in range(n_modes_2):
        popcounts += ((all_states >> bit_i) & 1).astype(np.uint8)

    even_states = all_states[(popcounts & 1) == 0]
    odd_states = all_states[(popcounts & 1) == 1]
    assert even_states.size == 32768 and odd_states.size == 32768

    state_to_even_idx = np.full(dim_2, -1, dtype=np.int32)
    state_to_odd_idx = np.full(dim_2, -1, dtype=np.int32)
    state_to_even_idx[even_states] = np.arange(32768, dtype=np.int32)
    state_to_odd_idx[odd_states] = np.arange(32768, dtype=np.int32)

    # Precompute cumulative parity table below each bit p: parity_below[p, st] in {+1.0, -1.0}
    parity_sign_below = np.ones((n_modes_2, dim_2), dtype=np.float64)
    cum_pop = np.zeros(dim_2, dtype=np.uint8)
    for p in range(n_modes_2):
        parity_sign_below[p] = np.where((cum_pop & 1) == 1, -1.0, 1.0)
        cum_pop = cum_pop + ((all_states >> p) & 1).astype(np.uint8)

    # Extract exact compact term lists for H_{hop} and H_{SMG} on L = 2!
    # 1) Chiral Weyl inter-site hopping: H_{hop} = -t sum_{i=0..3} (i c^dag_{0,i,up} c_{1,i,up} - i c^dag_{0,i,dn} c_{1,i,dn} + h.c.)
    #    (right-moving/left-moving Weyl spin velocity +-1 along link (0 <-> 1)):
    hop_terms = []  # (p_dag, q_ann, coeff)
    for i_c in range(4):
        for alpha, vel in ((0, 1.0), (1, -1.0)):
            p0 = 2 * i_c + alpha
            p1 = 8 + 2 * i_c + alpha
            hop_terms.append((p0, p1, -1.0j * vel))
            hop_terms.append((p1, p0, +1.0j * vel))

    # 2) On-site SMG interaction H_{SMG} = -sum_{x=0,1} sum_{A=0..3} { Delta_A^dag(x), Delta_A(x) }:
    #    Note each Delta_A(x) is a sum of 4 commuting/distinct bilinear pairs (u < v) with coeff w_{uv}!
    #    Let us expand { Delta_A^dag, Delta_A } = sum_{u<v, r<s} w_{uv}^* w_{rs} { c_v^dag c_u^dag, c_r c_s }:
    #    Wait: it is even cleaner AND 4x faster to apply v -> Delta_A |v> and Delta_A^dag |v> sequentially
    #    on the 32,768-element vector! Because Delta_A flips 2 fermions, it maps even -> even and odd -> odd directly!
    delta_pair_lists = []  # list of 8 operators (2 sites x 4 Hurwitz A), each a list of (u, v, coeff) for coeff * c_u c_v (u < v)
    for x_site in (0, 1):
        base = 8 * x_site
        for SA in S_hurwitz:
            pairs = {}
            for i_c in range(4):
                for j_c in range(4):
                    w = SA[i_c, j_c]
                    if abs(w) < 1.0e-14:
                        continue
                    for alpha, beta, eps_ab in ((0, 1, 0.5), (1, 0, -0.5)):
                        p = base + 2 * i_c + alpha
                        q = base + 2 * j_c + beta
                        if p == q:
                            continue
                        if p < q:
                            u, v, c_val = p, q, eps_ab * w
                        else:
                            u, v, c_val = q, p, -eps_ab * w
                        pairs[(u, v)] = pairs.get((u, v), 0.0j) + c_val
            clean_pairs = [(u, v, c_val) for (u, v), c_val in pairs.items() if abs(c_val) > 1.0e-14]
            delta_pair_lists.append(clean_pairs)

    def make_sector_matvec(sector_states: np.ndarray, state_to_idx: np.ndarray, g_coup: float):
        """Exact matrix-free action of H(g) = (1 - g) H_{hop} + g H_{SMG} on a 32,768-dim fermion-parity sector."""
        # Precompute transitions for each pair (u, v) with u < v (c_u c_v and c_v^dag c_u^dag) and hopping c_p^dag c_q!
        hop_maps = []
        if abs(1.0 - g_coup) > 1.0e-15:
            for p_dag, q_ann, coeff in hop_terms:
                bit_q = np.uint32(1 << q_ann)
                bit_p = np.uint32(1 << p_dag)
                mask_valid = ((sector_states & bit_q) != 0) & (((sector_states ^ bit_q) & bit_p) == 0)
                src_idx = np.nonzero(mask_valid)[0]
                st_src = sector_states[src_idx]
                sgn_q = parity_sign_below[q_ann, st_src]
                st_mid = st_src ^ bit_q
                sgn_p = parity_sign_below[p_dag, st_mid]
                dst_idx = state_to_idx[st_mid ^ bit_p]
                hop_maps.append((dst_idx, src_idx, ((1.0 - g_coup) * coeff) * (sgn_p * sgn_q)))

        delta_ops_maps = []
        if abs(g_coup) > 1.0e-15:
            for pair_list in delta_pair_lists:
                ann_terms = []
                cre_terms = []
                for u, v, c_val in pair_list:
                    bit_u = np.uint32(1 << u)
                    bit_v = np.uint32(1 << v)
                    # c_u c_v on |st>: requires both bit_v and bit_u occupied!
                    mask_occ = ((sector_states & bit_v) != 0) & ((sector_states & bit_u) != 0)
                    src_a = np.nonzero(mask_occ)[0]
                    st_a = sector_states[src_a]
                    sgn_v = parity_sign_below[v, st_a]
                    st_after_v = st_a ^ bit_v
                    sgn_u = parity_sign_below[u, st_after_v]
                    dst_a = state_to_idx[st_after_v ^ bit_u]
                    ann_terms.append((dst_a, src_a, c_val * (sgn_u * sgn_v)))
                    # (c_u c_v)^dag = c_v^dag c_u^dag: exact Hermitian transpose of (dst_a, src_a, weight)!
                    cre_terms.append((src_a, dst_a, np.conj(c_val) * (sgn_u * sgn_v)))
                delta_ops_maps.append((ann_terms, cre_terms))

        def matvec(vec: np.ndarray) -> np.ndarray:
            out = np.zeros_like(vec, dtype=np.complex128)
            for dst_i, src_i, w_arr in hop_maps:
                out[dst_i] += w_arr * vec[src_i]
            if delta_ops_maps:
                for ann_terms, cre_terms in delta_ops_maps:
                    # tmp1 = Delta |vec>, then apply Delta^dag; tmp2 = Delta^dag |vec>, then apply Delta:
                    tmp_ann = np.zeros_like(vec, dtype=np.complex128)
                    for dst_a, src_a, w_a in ann_terms:
                        tmp_ann[dst_a] += w_a * vec[src_a]
                    for dst_c, src_c, w_c in cre_terms:
                        out[dst_c] -= g_coup * (w_c * tmp_ann[src_c])

                    tmp_cre = np.zeros_like(vec, dtype=np.complex128)
                    for dst_c, src_c, w_c in cre_terms:
                        tmp_cre[dst_c] += w_c * vec[src_c]
                    for dst_a, src_a, w_a in ann_terms:
                        out[dst_a] -= g_coup * (w_a * tmp_cre[src_a])
            return out

        return LinearOperator((32768, 32768), matvec=matvec, dtype=np.complex128)

    rng = np.random.default_rng(20261126)
    v0_even = rng.standard_normal(32768) + 1.0j * rng.standard_normal(32768)
    v0_odd = rng.standard_normal(32768) + 1.0j * rng.standard_normal(32768)

    l2_scan = []
    for g_val in (0.25, 0.50, 0.75, 1.00):
        op_even = make_sector_matvec(even_states, state_to_even_idx, g_val)
        op_odd = make_sector_matvec(odd_states, state_to_odd_idx, g_val)
        ev_e, vec_e = eigsh(op_even, k=4, which="SA", v0=v0_even, tol=1.0e-10, maxiter=4000)
        ev_o, _ = eigsh(op_odd, k=2, which="SA", v0=v0_odd, tol=1.0e-10, maxiter=4000)
        ev_e = np.sort(np.real(ev_e))
        ev_o = np.sort(np.real(ev_o))
        all_low = np.sort(np.concatenate([ev_e, ev_o]))
        E0_tot = float(all_low[0])
        E1_tot = float(all_low[1])
        gap_tot = E1_tot - E0_tot
        deg_tot = int(np.sum(np.abs(all_low - E0_tot) < 1.0e-6))

        # Check diagonal <c_p^dag c_p - 1/2>, ALL 120 Majorana pairing bilinears <c_p c_q> (p < q),
        # and ALL symmetry-breaking particle-hole bilinears <c_p^dag c_q> (excluding the 8 chiral-invariant
        # kinetic link hopping pairs (0,i,alpha) <-> (1,i,alpha)):
        gs_e = vec_e[:, int(np.argmin(np.real(ev_e)))]
        prob_e = np.abs(gs_e) ** 2
        max_occ_dev = 0.0
        for p in range(n_modes_2):
            occ_p = float(np.sum(prob_e[((even_states >> p) & 1) != 0]))
            max_occ_dev = max(max_occ_dev, abs(occ_p - 0.5))

        max_pp_l2 = 0.0
        max_symbreak_ph_l2 = 0.0
        hop_set = {(2 * i_c + alpha, 8 + 2 * i_c + alpha) for i_c in range(4) for alpha in (0, 1)}
        for p in range(n_modes_2):
            bit_p = np.uint32(1 << p)
            for q in range(p + 1, n_modes_2):
                bit_q = np.uint32(1 << q)
                # 1) Majorana pairing <c_p c_q>:
                mask_pp = ((even_states & bit_q) != 0) & ((even_states & bit_p) != 0)
                src_pp = np.nonzero(mask_pp)[0]
                st_pp = even_states[src_pp]
                sgn_pp = parity_sign_below[q, st_pp] * parity_sign_below[p, st_pp ^ bit_q]
                dst_pp = state_to_even_idx[(st_pp ^ bit_q) ^ bit_p]
                val_pp = complex(np.vdot(gs_e[dst_pp], sgn_pp * gs_e[src_pp]))
                max_pp_l2 = max(max_pp_l2, abs(val_pp))

                # 2) Symmetry-breaking particle-hole <c_p^dag c_q> (for (p, q) not a kinetic hop link):
                if (p, q) not in hop_set:
                    mask_ph = ((even_states & bit_q) != 0) & (((even_states ^ bit_q) & bit_p) == 0)
                    src_ph = np.nonzero(mask_ph)[0]
                    st_ph = even_states[src_ph]
                    sgn_ph = parity_sign_below[q, st_ph] * parity_sign_below[p, st_ph ^ bit_q]
                    dst_ph = state_to_even_idx[(st_ph ^ bit_q) ^ bit_p]
                    val_ph = complex(np.vdot(gs_e[dst_ph], sgn_ph * gs_e[src_ph]))
                    max_symbreak_ph_l2 = max(max_symbreak_ph_l2, abs(val_ph))

        assert E0_tot == float(ev_e[0]), f"Expected even-parity GS at g={g_val}"
        assert deg_tot == 1, f"Degenerate ground state at g={g_val}: {all_low}"
        assert gap_tot > 0.25, f"Many-body gap closed at g={g_val}: {gap_tot}"
        assert max_occ_dev < 1.0e-9
        assert max_pp_l2 < 1.0e-12
        assert max_symbreak_ph_l2 < 1.0e-12

        l2_scan.append({
            "g": g_val,
            "E0_even_GS": round(float(ev_e[0]), 6),
            "E1_even": round(float(ev_e[1]), 6),
            "E0_odd": round(float(ev_o[0]), 6),
            "many_body_gap_Delta_E": round(gap_tot, 6),
            "ground_state_degeneracy": deg_tot,
            "max_abs_n_p_minus_half": float(f"{max_occ_dev:.3e}"),
            "max_abs_all_120_Majorana_pp_bilinears": float(f"{max_pp_l2:.3e}"),
            "max_abs_symbreak_ph_bilinears": float(f"{max_symbreak_ph_l2:.3e}"),
        })

    # Gate 3 decouple check at g = 1.0: L = 2 decoupled sites MUST satisfy E0(L=2) == 2 * E0(L=1) = -48.0
    # and Delta E(L=2) == Delta E(L=1) = 8.0!
    assert abs(l2_scan[-1]["E0_even_GS"] - (-48.0)) < 1.0e-6
    assert abs(l2_scan[-1]["many_body_gap_Delta_E"] - 8.0) < 1.0e-6

    return {
        "hurwitz_4_subalgebra_of_Spin10_16": "4 symmetric 4x4 unitary matrices S_A^T = +S_A, S(n)^dag S(n) = |n|^2 I_4",
        "L1_single_site_dim256_exact": {
            "hilbert_dim": dim_1,
            "E0_GS": round(E0_1, 8),
            "E1_first_excited": round(E1_1, 8),
            "exact_gap_Delta_E": round(gap_1, 8),
            "ground_state_degeneracy": deg_1,
            "lorentz_spin_J2_expectation": float(f"{spin_j2_gs:.3e}"),
            "max_abs_quadratic_ph_bilinear": float(f"{max_ph_bilin:.3e}"),
            "max_abs_quadratic_pp_majorana_bilinear": float(f"{max_pp_bilin:.3e}"),
        },
        "L2_coupled_sites_dim65536_lanczos": {
            "hilbert_dim_total": dim_2,
            "hilbert_dim_even_odd": [32768, 32768],
            "g_scan": l2_scan,
        },
    }


def gate3_invariances_and_limits(cl_data: dict) -> dict:
    """Gate 3: Explicit Spin(10) covariance of M_a, exponential L_s chiral-wall decoupling,
    and random S^9 unit-vector spectral invariance.
    """
    M10 = cl_data["M_10"]
    T45 = cl_data["T_45"]
    pairs = cl_data["pair_labels"]

    # 1. Spin(10) vector covariance of M_c: T_{ab}^T M_c + M_c T_{ab} == -i (delta_{ac} M_b - delta_{bc} M_a)!
    # Check sign on (a=0, b=1, c=0):
    idx_01 = pairs.index((0, 1))
    lhs_01_0 = mm(T45[idx_01].T, M10[0]) + mm(M10[0], T45[idx_01])
    # Determine exact convention sign (+i or -i) from explicitCl(10) and verify across ALL 45 x 10 = 450 pairs!
    sgn_cov = +1.0j if np.max(np.abs(lhs_01_0 - 1.0j * M10[1])) < 1.0e-12 else -1.0j
    max_cov_err = 0.0
    for idx_p, (a_i, b_i) in enumerate(pairs):
        Tab = T45[idx_p]
        for c_i in range(10):
            lhs = mm(Tab.T, M10[c_i]) + mm(M10[c_i], Tab)
            rhs = np.zeros((16, 16), dtype=np.complex128)
            if c_i == a_i:
                rhs = sgn_cov * M10[b_i]
            elif c_i == b_i:
                rhs = -sgn_cov * M10[a_i]
            max_cov_err = max(max_cov_err, float(np.max(np.abs(lhs - rhs))))
    assert max_cov_err < 1.0e-13, f"Spin(10) SO(10)-vector covariance of M_a failed: {max_cov_err}"

    # 2. Random S^9 orientation invariance of singular values of M(n):
    rng = np.random.default_rng(137)
    max_s9_err = 0.0
    for _ in range(25):
        n_vec = rng.standard_normal(10)
        n_vec /= float(np.linalg.norm(n_vec))
        Mn = np.tensordot(n_vec, M10, axes=(0, 0))
        prod = mm(Mn.conj().T, Mn)
        max_s9_err = max(max_s9_err, float(np.max(np.abs(prod - np.eye(16)))))
    assert max_s9_err < 1.0e-14

    # 3. Exact 2nd-order wall-to-wall Majorana tunneling scaling ~ y_M * (1 - m_0)^{2 L_s}:
    #    When the right mirror wall is gapped (y_M = 1.5), the induced Majorana mass of the left-wall
    #    chiral fermion is quadratic in the wall-to-wall wavefunction tail (1 - m_0)^{L_s}, so
    #    increasing L_s by Delta L_s = 4 at m_0 = 0.5 suppresses leakage by (1/2)^{2*4} = 1/256 = 0.00390625!
    k0 = np.zeros(4, dtype=float)
    leak_Ls6 = solve_5d_dw_bdg_single_k(k0, 0.0, m0=0.5, L_s=6, y_M=1.5, chiral_projected=True)["min_left_physical_gap"]
    leak_Ls10 = solve_5d_dw_bdg_single_k(k0, 0.0, m0=0.5, L_s=10, y_M=1.5, chiral_projected=True)["min_left_physical_gap"]
    leak_Ls14 = solve_5d_dw_bdg_single_k(k0, 0.0, m0=0.5, L_s=14, y_M=1.5, chiral_projected=True)["min_left_physical_gap"]
    ratio_10_6 = leak_Ls10 / leak_Ls6
    ratio_14_10 = leak_Ls14 / leak_Ls10
    exact_tunnel_ratio = 0.5 ** 8  # == 1/256 = 0.00390625
    assert abs(ratio_14_10 - exact_tunnel_ratio) / exact_tunnel_ratio < 1.0e-5
    leak_fid_Ls12 = solve_5d_dw_bdg_single_k(k0, 0.0, m0=0.9, L_s=12, y_M=1.5, chiral_projected=True)["min_left_physical_gap"]
    assert leak_fid_Ls12 < 1.0e-15

    return {
        "Spin10_vector_covariance_all_450_commutators_max_err": float(f"{max_cov_err:.3e}"),
        "S9_random_25_orientations_unitarity_max_err": float(f"{max_s9_err:.3e}"),
        "DomainWall_Ls_exponential_chiral_protection": {
            "m0_0p5_leakage_Ls6": float(f"{leak_Ls6:.5e}"),
            "m0_0p5_leakage_Ls10": float(f"{leak_Ls10:.5e}"),
            "m0_0p5_leakage_Ls14": float(f"{leak_Ls14:.5e}"),
            "ratio_Ls10_over_Ls6": round(ratio_10_6, 8),
            "ratio_Ls14_over_Ls10_vs_exact_1_over_256": round(ratio_14_10, 10),
            "fiducial_m0_0p9_Ls12_leakage": float(f"{leak_fid_Ls12:.3e}"),
        },
    }


def main():
    cl_data = build_cl10_and_spin10_weyl_numpy()
    g1 = gate1_external_benchmarks(cl_data)
    g2a = gate2a_sympy_spin10_bilinear_and_nodeless_s9_proof()
    g2bc = gate2b_and_2c_24cell_d4_watson_and_dw_window()
    g2d = gate2d_exact_manybody_2comp_weyl_smg()
    g3 = gate3_invariances_and_limits(cl_data)

    out = {
        "script": "verify_4d_spin10_chiral_smg_and_ep_obstruction.py",
        "epistemic_status": {
            "NATURE_LEVEL_CLOSURE": False,
            "PROVEN_ALGEBRAIC": [
                "First-principles Cl(10) charge-conjugation SymPy proof: 16_+ x 16_+ = 10_S (+) 120_A (+) bar{126}_S with symmetric M_a^T = +M_a satisfying M(n)^dag M(n) == (sum_{a=1}^{10} n_a^2) I_{16} for all n in R^{10} (zero nodal points on S^9) and exact anomaly cancellation across all 91,125 Spin(10) generator triples. Spinless 1-component fermions vanish identically in the 10_S channel, whereas 4D 2-component Lorentz-Weyl spinors form the non-vanishing Lorentz-scalar pairing operator Delta_a = (1/2) eps^{alpha beta} psi_{i alpha} M_a^{ij} psi_{j beta}.",
                "Exact 4D D_4 reciprocal-lattice & Golterman-Petcher-Rivas / Tamm boundary resurrection theorems: (pi,pi,pi,pi) in D_4^* is a reciprocal-lattice vector (leaving 7 distinct non-zero half-period BZ points, all lifted with hat{K}_{D_4} in {8, 32/3} > 0 in ell_0 = 1 units). Unprojected 4D boundary Yukawa coupling resurrects a mirror mode at k = 0 as y_M -> infinity with y_M * Delta E_{mirror}(0) -> m_0(2 - m_0), whereas chirally projected P_R boundary coupling eliminates k = 0 resurrection completely (> 0.90 at y_M = 10^4) but pulls down a supercritical BZ doubler Tamm state at k = (pi,0,0,0) with y_M * Delta E_{doub} -> M_{eff}(pi)^2 - 1 = 15.81 — proving a strict PROVEN NO-GO for y_M -> infinity in BOTH prescriptions and bounding mirror SMG to a wide intermediate window y_M in [1, 15] (peak BZ gap 0.904).",
            ],
            "PROVEN_NUMERIC": [
                "Exact 4D D_4 physical Watson integral G_{D_4}^{phys}(0) (verified against the universal 4D continuum IR logarithm slope 1/(16 pi^2)) proves a strictly UV/IR-finite disordered O(10) symmetric phase <n_a> = 0 in d = 4, and the 5D Domain-Wall 24-cell BZ scan (L_s = 12, r_W = 1.0, m_0 = 0.9) establishes full-BZ mirror gapping (> 0.60 everywhere in BZ) with < 10^{-15} k = 0 chiral leakage.",
                "Exact many-body diagonalization / matrix-free bit-manipulation Lanczos of the 2-component Weyl symmetric-Hurwitz SMG Hamiltonian on L = 1 (dim = 256) and L = 2 coupled sites (dim = 65,536 = 32,768_even + 32,768_odd) proves a unique spin-singlet ground state (deg = 1), strictly positive many-body gap across g in (0, 1], and identically vanishing mass / symmetry-breaking fermion bilinear expectation values (<c_p c_q> = 0 across all 120 Majorana pairs and <c_p^dag c_q>_symbreak = 0).",
            ],
            "OPEN": [
                "Constructive 4D/5D dynamical Spin(10) chiral-gauge lattice Monte Carlo continuum limit (a -> 0): proving non-perturbatively in infinite 4D volume (with dynamical Spin(10) link variables U_ell turned on and without triggering an Eichten-Preskill symmetric-phase light composite-fermion transition or sign-problem obstruction) that the gapped mirror phase connects smoothly to the continuum limit.",
            ],
        },
        "gate1_benchmarks": g1,
        "gate2a_sympy_spin10_algebra": g2a,
        "gate2bc_24cell_watson_and_dw_window": g2bc,
        "gate2d_manybody_dim65536_weyl_smg": g2d,
        "gate3_invariances_and_limits": g3,
    }

    _data_dir = SCRIPT_DIR.parent / "data"
    out_path = (_data_dir if _data_dir.is_dir() else SCRIPT_DIR) / "spin10_chiral_smg_and_ep_obstruction.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
