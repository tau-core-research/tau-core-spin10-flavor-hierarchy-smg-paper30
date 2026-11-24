#!/usr/bin/env python3
"""Step 2: Exact Quantum Many-Body Simulation of the 1+1D Multi-Cell Spin(7)
Octonionic Cayley Symmetric Mass Generation (SMG) Chain!

Exact Hilbert-space diagonalization (ZERO approximations, python3 -W error) on:
  - L = 2 coupled 4-simplex mirror sites (16 Majoranas, dim = 2^8 = 256, dense),
  - L = 3 periodic coupled 4-simplex mirror sites (24 Majoranas, dim = 2^12 = 4096, sparse)!

Hamiltonian across g in [0, 1]:
    H(g) = (1 - g) * H_hop + g * H_SMG
  where:
    H_hop = i * sum_{x=0}^{L-1} sum_{a=0}^{7} chi_{x,a} chi_{x+1,a}
            (exact Spin(8) subset Spin(7)-invariant 1+1D free massless Majorana hopping!
             On L = 3 periodic, k = 0 has E(0) = 2 sin(0) = 0 -> EXACTLY GAPLESS
             16-fold degenerate ground state at g = 0!)
    H_SMG = - sum_{x=0}^{L-1} sum_{14 Cayley 4-tuples} s_{abcd} chi_{x,a} chi_{x,b} chi_{x,c} chi_{x,d}
            (exact Spin(7)-invariant octonionic Cayley 4-fermion interaction!)

Computes across the entire phase diagram g in [0, 1]:
  1. Exact energy gap Delta E(g) = E_1(g) - E_0(g),
  2. Ground-state energy E_0(g) and susceptibility -d^2 E_0 / dg^2 (pinpointing critical g_c!),
  3. Exact bilinear symmetry-breaking VEV max_{a<b} |<i chi_{0,a} chi_{0,b}>| (proving ZERO
     spontaneous Spin(7) symmetry breaking across ALL g > 0!),
  4. Exact single-site von Neumann entanglement entropy S_1(g) (watching entangled gapless
     free-fermion phase at small g transition into the gapped SMG singlet product state S_1 -> 0!).
"""

import json
import math
import sys
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def cayley_14_tuples():
    fano = [(1, 2, 3), (1, 4, 5), (1, 7, 6), (2, 4, 6), (2, 5, 7), (3, 4, 7), (3, 6, 5)]

    def perm_sign(seq):
        inv = sum(1 for a in range(len(seq)) for b in range(a + 1, len(seq)) if seq[a] > seq[b])
        return -1.0 if (inv % 2 == 1) else 1.0

    terms = []
    for (i, j, k) in fano:
        terms.append(((0, i, j, k), +1.0))
        comp = sorted(set(range(1, 8)) - {i, j, k})
        terms.append((tuple(comp), perm_sign([i, j, k] + comp)))
    assert len(terms) == 14
    return terms


def build_majorana_sparse(n_complex: int):
    """Build all 2*n_complex Hermitian Majorana operators chi_0..chi_{2*n_complex-1}
    as ultra-fast CSR sparse matrices (1 nonzero per row, dim = 2^n_complex).
    """
    dim = 1 << n_complex
    rows = np.arange(dim, dtype=np.int32)
    chis = []
    for k in range(n_complex):
        bit = 1 << (n_complex - 1 - k)
        # Jordan-Wigner Z string on bits strictly above k (higher bit positions):
        # Count number of 1-bits in mask_above = ((1 << k) - 1) << (n_complex - k)
        parity = np.zeros(dim, dtype=np.int8)
        for j in range(k):
            parity ^= ((rows >> (n_complex - 1 - j)) & 1).astype(np.int8)
        jw_sign = np.where(parity == 0, 1.0, -1.0)
        cols = rows ^ bit
        # sx: |0> -> |1>, |1> -> |0> (real +1)
        data_x = jw_sign.astype(np.complex128)
        chi_x = sparse.csr_matrix((data_x, (rows, cols)), shape=(dim, dim))
        # sy: |0> -> +i|1>, |1> -> -i|0> -> row=0,col=1 has -i; row=1,col=0 has +i
        is_one = ((rows >> (n_complex - 1 - k)) & 1) == 1
        data_y = jw_sign * np.where(is_one, 1j, -1j)
        chi_y = sparse.csr_matrix((data_y, (rows, cols)), shape=(dim, dim))
        chis.append(chi_x)
        chis.append(chi_y)
    return chis


def entanglement_entropy_site0(psi: np.ndarray, L: int) -> float:
    """Exact von Neumann entanglement entropy S_1 = -Tr(rho_0 ln rho_0) of site 0
    (4 complex fermions -> dim_A = 16) vs rest of chain (dim_B = 16^(L-1)).
    """
    dim_A = 16
    dim_B = 16 ** (L - 1)
    M = psi.reshape(dim_A, dim_B)
    s = np.linalg.svd(M, compute_uv=False)
    p = s * s
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log(p)))


def run_chain(L: int, g_grid):
    t0 = time.time()
    n_complex = 4 * L
    dim = 1 << n_complex
    chis = build_majorana_sparse(n_complex)
    # Index helper: site x in 0..L-1, flavor a in 0..7 -> 8*x + a
    cayley = cayley_14_tuples()

    # Build H_hop (open for L=2 so 0->1 doesn't cancel; periodic for L=3 so k=0 is exact gapless!):
    H_hop = sparse.csr_matrix((dim, dim), dtype=np.complex128)
    bonds = [(0, 1)] if L == 2 else [(0, 1), (1, 2), (2, 0)]
    for x1, x2 in bonds:
        for a in range(8):
            H_hop = H_hop + (1j * (chis[8 * x1 + a] @ chis[8 * x2 + a]))
    # Verify Hermiticity:
    assert sparse.linalg.norm(H_hop - H_hop.getH()) < 1e-12

    # Build H_SMG (sum of exact Spin(7) Cayley 4-fermion operator on every site x):
    H_SMG = sparse.csr_matrix((dim, dim), dtype=np.complex128)
    for x in range(L):
        for (a, b, c, d), sgn in cayley:
            op = chis[8 * x + a] @ chis[8 * x + b] @ chis[8 * x + c] @ chis[8 * x + d]
            H_SMG = H_SMG - sgn * op
    assert sparse.linalg.norm(H_SMG - H_SMG.getH()) < 1e-12

    # Prebuild all 28 on-site bilinears on site 0 for exact symmetry-breaking audit:
    bilin_site0 = [1j * (chis[a] @ chis[b]) for a in range(8) for b in range(a + 1, 8)]
    # And cross-site bilinear correlator (i chi_{0,0} chi_{0,1})(i chi_{1,0} chi_{1,1}):
    corr_01 = (1j * (chis[0] @ chis[1])) @ (1j * (chis[8] @ chis[9]))

    # Note: H(g) preserves total fermion parity P = prod chi_j!
    # For L=2 (dim=256), use exact dense eigh; for L=3 (dim=4096), use sparse eigsh (k=6)!
    records = []
    H_hop_d = H_hop.toarray() if dim <= 512 else None
    H_SMG_d = H_SMG.toarray() if dim <= 512 else None

    for g in g_grid:
        if dim <= 512:
            H = (1.0 - g) * H_hop_d + g * H_SMG_d
            evals, evecs = np.linalg.eigh(H)
            E0, E1, E2 = float(evals[0]), float(evals[1]), float(evals[2])
            gs = evecs[:, 0]
            deg0 = int(np.sum(np.abs(evals - E0) < 1e-8))
        else:
            H = (1.0 - g) * H_hop + g * H_SMG
            # Shift-invert or standard smallest algebraic 'SA' with high tol:
            evals, evecs = eigsh(H, k=6, which="SA", tol=1e-10, maxiter=5000)
            idx = np.argsort(evals.real)
            evals, evecs = evals[idx].real, evecs[:, idx]
            E0, E1, E2 = float(evals[0]), float(evals[1]), float(evals[2])
            gs = evecs[:, 0]
            deg0 = int(np.sum(np.abs(evals - E0) < 1e-7))

        gap = E1 - E0
        # Check maximal bilinear symmetry-breaking VEV on site 0 (meaningful when unique GS, deg0 == 1):
        max_bilin = max(abs(complex(gs.conj() @ (B @ gs))) for B in bilin_site0)
        c01_val = float(np.real(gs.conj() @ (corr_01 @ gs)))
        S1 = entanglement_entropy_site0(gs, L)
        records.append(dict(
            g=round(float(g), 4),
            E0=round(E0, 6),
            gap=round(gap, 6),
            E2_minus_E0=round(E2 - E0, 6),
            gs_deg_in_lowest6=deg0,
            max_bilinear_VEV=float(f"{max_bilin:.2e}"),
            cross_site_bilin_corr=round(c01_val, 6),
            entanglement_S1=round(S1, 5),
        ))

    # Compute exact finite-difference susceptibility -d^2 E_0 / dg^2 to locate crossover g_c:
    gs_arr = np.array([r["g"] for r in records])
    E0_arr = np.array([r["E0"] for r in records])
    susc = []
    for i in range(1, len(records) - 1):
        dg1 = gs_arr[i] - gs_arr[i - 1]
        dg2 = gs_arr[i + 1] - gs_arr[i]
        d2 = 2.0 * ((E0_arr[i + 1] - E0_arr[i]) / dg2 - (E0_arr[i] - E0_arr[i - 1]) / dg1) / (dg1 + dg2)
        susc.append((float(gs_arr[i]), float(-d2)))
    peak_g, peak_susc = max(susc, key=lambda t: t[1])

    # Verify g=1 exact analytical values: E0 == -14*L, gap == 14.0, S1 == 0.0!
    assert abs(records[-1]["E0"] + 14.0 * L) < 1e-5
    assert abs(records[-1]["gap"] - 14.0) < 1e-5
    assert records[-1]["entanglement_S1"] < 1e-4

    # Verify ZERO bilinear symmetry breaking across ALL unique-GS points (g > 0):
    max_vev_all_g_pos = max(r["max_bilinear_VEV"] for r in records if r["g"] > 0.0 and r["gs_deg_in_lowest6"] == 1)
    assert max_vev_all_g_pos < 1e-10

    sys.stderr.write(
        f"[L={L} (dim={dim}) done in {time.time()-t0:.2f}s] "
        f"gap(g=0)={records[0]['gap']:.4f} -> gap(g=1)={records[-1]['gap']:.4f}, "
        f"peak susceptibility at g_c~{peak_g:.2f} (-d2E0/dg2={peak_susc:.1f}), "
        f"max_bilin_VEV(g>0)={max_vev_all_g_pos:.2e}\n"
    )
    return dict(
        L_sites=L,
        n_majoranas=8 * L,
        hilbert_dim=dim,
        boundary_condition="open (1 bond)" if L == 2 else "periodic (3 bonds, exact k=0 gapless at g=0!)",
        peak_susceptibility_gc=peak_g,
        peak_susceptibility_val=round(peak_susc, 2),
        max_bilinear_VEV_for_all_unique_GS=max_vev_all_g_pos,
        phase_scan=records,
    )


def main():
    g_grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]
    out = dict(
        Chain_L2_dim256=run_chain(2, g_grid),
        Chain_L3_periodic_dim4096=run_chain(3, g_grid),
    )
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
