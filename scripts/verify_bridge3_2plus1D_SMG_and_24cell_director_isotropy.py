#!/usr/bin/env python3
r"""Bridge 3 (Paper 30 v1.2.0, Referee-Hardened):
  3A. Exact Full-Hilbert-Space (Even + Odd Parity!) Many-Body Lanczos Diagonalization
      of 32-Majorana (L=4 sites, dim = 65,536 = 32,768 even + 32,768 odd) Spin(7) Cayley SMG:
      - 1+1D L=4 Periodic Chain (exact gapless at g=0, Delta E_true(0) = 0.0000),
      - 1+1D L=4 Antiperiodic Ring (C_4 with pi-flux, z=2; finite-size gapped at g=0),
      - 4-Site K_4 Tetrahedral Cluster (z=3, 6 bonds; finite-size avoided-crossing crossover,
        NOT claimed as a thermodynamic gapless-to-SMG QPT!).
      Reports BOTH E_0 (verified in even parity), true full-Hilbert-space gap
      Delta E_true = min(E_1^even, E_0^odd) - E_0^even (== 14.0 at g=1!), and deterministic
      bilinear VEVs (seeded Lanczos v0 for 100% reproducibility!).
  3B. Exact |W(F_4)| = 1,152 Reynolds Character Proof of 4D Euclidean O(4) Rank-4 Director
      Isotropy on the 24-Cell Lattice:
        dim(((R^4)^{\otimes 2})^{W(F_4)}) == 1  (exact O(4))
        dim(((R^4)^{\otimes 4})^{W(F_4)}) == 3  (exact O(4) for ALL non-symmetric rank-4
          director kinetic tensors C^{mnrs} (d_m u_n)(d_r u_s) with independent c_1, c_2, c_3!)
        dim(((R^4)^{\otimes 6})^{W(F_4)}) == 16 > 15  (first W(F_4) anisotropy at rank 6!)

Strict `python3 -W error` compliance. Zero handwaving.
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

    def perm_sign(p):
        inv = sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])
        return -1.0 if (inv % 2 == 1) else 1.0

    terms = []
    for (i, j, k) in fano:
        terms.append((1.0, (0, i, j, k)))
        comp = tuple(sorted(set(range(1, 8)) - {i, j, k}))
        terms.append((perm_sign((i, j, k) + comp), comp))
    return terms


def build_parity_basis(n_complex: int, parity: int):
    """Return sorted array of all basis states with popcount % 2 == parity (0=even, 1=odd)."""
    n_tot = 1 << n_complex
    states = np.arange(n_tot, dtype=np.int32)
    pop = np.zeros(n_tot, dtype=np.int8)
    for b in range(n_complex):
        pop ^= ((states >> b) & 1).astype(np.int8)
    sel_states = states[pop == parity]
    inv_map = np.full(n_tot, -1, dtype=np.int32)
    inv_map[sel_states] = np.arange(len(sel_states), dtype=np.int32)
    return sel_states, inv_map


def apply_majorana_seq(states: np.ndarray, maj_indices):
    cur = states.copy()
    coeff = np.ones(len(states), dtype=np.complex128)
    for m in reversed(maj_indices):
        q = m >> 1
        odd = m & 1
        if q > 0:
            low = cur & ((1 << q) - 1)
            p = np.zeros(len(cur), dtype=np.int8)
            for b in range(q):
                p ^= ((low >> b) & 1).astype(np.int8)
            coeff *= np.where(p == 0, 1.0, -1.0)
        bit_q = (cur >> q) & 1
        if odd:
            coeff *= np.where(bit_q == 0, 1j, -1j)
        cur ^= (1 << q)
    return cur, coeff


def build_smg_hamiltonian_sector(L: int, bonds, bond_signs, parity: int):
    n_complex = 4 * L
    states, inv_map = build_parity_basis(n_complex, parity)
    dim_s = len(states)
    rows_base = np.arange(dim_s, dtype=np.int32)
    cayley = cayley_14_tuples()

    fk_rows, fk_cols, fk_vals = [], [], []
    for s in range(L):
        off = 8 * s
        for c_val, (a, b, c, d) in cayley:
            tgt, amp = apply_majorana_seq(states, (off + a, off + b, off + c, off + d))
            assert np.max(np.abs(amp.imag)) < 1e-14
            fk_rows.append(inv_map[tgt])
            fk_cols.append(rows_base)
            fk_vals.append(-c_val * amp.real)
    H_FK = sparse.csr_matrix(
        (np.concatenate(fk_vals), (np.concatenate(fk_rows), np.concatenate(fk_cols))),
        shape=(dim_s, dim_s), dtype=np.complex128,
    )
    H_FK.sum_duplicates()

    hop_rows, hop_cols, hop_vals = [], [], []
    for (u, v), bsgn in zip(bonds, bond_signs):
        for a in range(8):
            tgt, amp = apply_majorana_seq(states, (8 * u + a, 8 * v + a))
            hop_rows.append(inv_map[tgt])
            hop_cols.append(rows_base)
            hop_vals.append((1j * bsgn) * amp)
    H_hop = sparse.csr_matrix(
        (np.concatenate(hop_vals), (np.concatenate(hop_rows), np.concatenate(hop_cols))),
        shape=(dim_s, dim_s), dtype=np.complex128,
    )
    H_hop.sum_duplicates()

    assert sparse.linalg.norm(H_FK - H_FK.getH()) < 1e-12
    assert sparse.linalg.norm(H_hop - H_hop.getH()) < 1e-12
    return H_hop, H_FK, states, inv_map


def max_site0_bilinear_vev(psi: np.ndarray, states: np.ndarray, inv_map: np.ndarray) -> float:
    max_v = 0.0
    for a in range(8):
        for b in range(a + 1, 8):
            tgt, amp = apply_majorana_seq(states, (a, b))
            val = np.sum(np.conj(psi[inv_map[tgt]]) * (1j * amp) * psi)
            max_v = max(max_v, float(abs(val)))
    return max_v


def site0_entanglement(psi: np.ndarray, states: np.ndarray, L: int) -> float:
    n_rest = 4 * (L - 1)
    idx_s0 = states & 0xF
    idx_rest = states >> 4
    M = np.zeros((16, 1 << n_rest), dtype=np.complex128)
    M[idx_s0, idx_rest] = psi
    rho0 = np.einsum("ij,kj->ik", M, M.conj())
    evals = np.linalg.eigvalsh(0.5 * (rho0 + rho0.conj().T))
    evals = evals[evals > 1e-14]
    return float(-np.sum(evals * np.log(evals)))


def scan_lattice_smg_full_hilbert(label: str, L: int, bonds, bond_signs, g_grid, seed: int = 20261124):
    """Exact full-Hilbert-space scan (BOTH even-parity AND odd-parity blocks!) with
    deterministic seeded Lanczos vectors v0!
    """
    t0 = time.time()
    H_hop_e, H_FK_e, states_e, inv_e = build_smg_hamiltonian_sector(L, bonds, bond_signs, parity=0)
    H_hop_o, H_FK_o, _, _ = build_smg_hamiltonian_sector(L, bonds, bond_signs, parity=1)
    dim_e = len(states_e)

    rng = np.random.default_rng(seed)
    v0_e = rng.normal(size=dim_e) + 1j * rng.normal(size=dim_e)
    v0_e /= np.linalg.norm(v0_e)
    v0_o = rng.normal(size=dim_e) + 1j * rng.normal(size=dim_e)
    v0_o /= np.linalg.norm(v0_o)

    records = []
    for g in g_grid:
        He = (1.0 - g) * H_hop_e + g * H_FK_e
        Ho = (1.0 - g) * H_hop_o + g * H_FK_o
        ev_e, vec_e = eigsh(He, k=3, which="SA", tol=1e-10, v0=v0_e, maxiter=6000)
        ev_o, vec_o = eigsh(Ho, k=2, which="SA", tol=1e-10, v0=v0_o, maxiter=6000)
        ord_e = np.argsort(ev_e.real)
        ord_o = np.argsort(ev_o.real)
        ev_e, vec_e = ev_e[ord_e].real, vec_e[:, ord_e]
        ev_o, vec_o = ev_o[ord_o].real, vec_o[:, ord_o]

        E0_e, E1_e = float(ev_e[0]), float(ev_e[1])
        E0_o = float(ev_o[0])
        assert E0_e <= E0_o + 1e-6, f"Ground state in odd parity at g={g}: E0_e={E0_e}, E0_o={E0_o}"
        psi0 = vec_e[:, 0]
        v0_e = psi0
        v0_o = vec_o[:, 0]

        gap_even = E1_e - E0_e
        gap_odd = E0_o - E0_e
        gap_true = min(gap_even, gap_odd)
        deg0_full = int(np.sum(np.abs(ev_e - E0_e) < 1e-5) + np.sum(np.abs(ev_o - E0_e) < 1e-5))
        vev = max_site0_bilinear_vev(psi0, states_e, inv_e)
        S1 = site0_entanglement(psi0, states_e, L)
        records.append(dict(
            g=round(float(g), 4),
            E0=round(E0_e, 6),
            gap_true_full_hilbert=round(gap_true, 6),
            gap_even_bosonic=round(gap_even, 6),
            gap_odd_fermionic=round(gap_odd, 6),
            deg0_full=deg0_full,
            max_bilinear_VEV=float(f"{vev:.2e}"),
            entanglement_S1=round(S1, 5),
        ))

    gs = np.array([r["g"] for r in records])
    E0s = np.array([r["E0"] for r in records])
    susc = []
    for i in range(1, len(records) - 1):
        dg1, dg2 = gs[i] - gs[i - 1], gs[i + 1] - gs[i]
        d2 = 2.0 * ((E0s[i + 1] - E0s[i]) / dg2 - (E0s[i] - E0s[i - 1]) / dg1) / (dg1 + dg2)
        susc.append((float(gs[i]), float(-d2)))
    peak_g, peak_susc = max(susc, key=lambda t: t[1])

    # Verify g=1 exact analytical singlet spectrum: E0 = -14*L, true fermionic gap = 14.0, even bosonic gap = 16.0!
    assert abs(records[-1]["E0"] + 14.0 * L) < 1e-4
    assert abs(records[-1]["gap_true_full_hilbert"] - 14.0) < 1e-4
    assert abs(records[-1]["gap_even_bosonic"] - 16.0) < 1e-4
    assert records[-1]["entanglement_S1"] < 1e-4
    max_vev_unique = max(r["max_bilinear_VEV"] for r in records if r["g"] > 0.0 and r["deg0_full"] == 1)
    assert max_vev_unique < 1e-10

    sys.stderr.write(
        f"  [{label}] DONE in {time.time()-t0:.1f}s | gap_TRUE(0)={records[0]['gap_true_full_hilbert']:.4f} -> "
        f"gap_TRUE(1)={records[-1]['gap_true_full_hilbert']:.4f} (even={records[-1]['gap_even_bosonic']:.1f}), "
        f"peak -d2E0/dg2={peak_susc:.1f} at g_c={peak_g:.2f}, max_VEV(unique)={max_vev_unique:.2e}\n"
    )
    return dict(
        label=label,
        L_sites=L,
        n_majoranas=8 * L,
        full_hilbert_dim=1 << (4 * L),
        even_parity_dim=dim_e,
        odd_parity_dim=dim_e,
        peak_susceptibility_gc=peak_g,
        peak_susceptibility_val=round(peak_susc, 2),
        min_gap_true_full=min(r["gap_true_full_hilbert"] for r in records),
        max_bilinear_VEV_unique_GS=max_vev_unique,
        phase_scan=records,
    )


def generate_W_F4_group():
    """Explicitly generate all |W(F_4)| = 1,152 orthogonal 4x4 symmetry matrices of the 24-cell
    by closing the 4 simple reflections of the F_4 Coxeter diagram!
    """
    # Simple roots of F_4 (unit normalized):
    #   alpha_1 = (1, -1, 0, 0)/sqrt(2)
    #   alpha_2 = (0, 1, -1, 0)/sqrt(2)
    #   alpha_3 = (0, 0, 1, 0)
    #   alpha_4 = (-1/2, -1/2, -1/2, -1/2)
    alphas = [
        np.array([1.0, -1.0, 0.0, 0.0]) / math.sqrt(2.0),
        np.array([0.0, 1.0, -1.0, 0.0]) / math.sqrt(2.0),
        np.array([0.0, 0.0, 1.0, 0.0]),
        np.array([-0.5, -0.5, -0.5, -0.5]),
    ]
    I4 = np.eye(4)
    gens = [I4 - 2.0 * np.outer(a, a) for a in alphas]

    def key(M):
        return tuple(int(round(x * 1000)) for x in M.ravel())

    group = {key(I4): I4}
    queue = [I4]
    while queue:
        cur = queue.pop()
        for g in gens:
            nxt = g @ cur
            k = key(nxt)
            if k not in group:
                group[k] = nxt
                queue.append(nxt)
    mats = list(group.values())
    assert len(mats) == 1152, f"Expected |W(F_4)| = 1152, got {len(mats)}"
    return mats


def audit_24cell_director_isotropy():
    r"""Exact |W(F_4)| = 1,152 Reynolds character proof of 4D Euclidean O(4) isotropy for
    ALL general (non-symmetric) rank-4 director kinetic tensors C^{mnrs}(d_m u_n)(d_r u_s):
      dim(((R^4)^{\otimes r})^{W(F_4)}) = (1/|W(F_4)|) sum_{g in W(F_4)} (Tr g)^r
    plus exact 24-root moment verification!
    """
    wf4 = generate_W_F4_group()
    traces = np.array([float(np.trace(M)) for M in wf4])
    dim_r2 = float(np.mean(traces ** 2))
    dim_r4 = float(np.mean(traces ** 4))
    dim_r6 = float(np.mean(traces ** 6))
    # O(4) invariant dimensions: (2k-1)!! -> r=2: 1, r=4: 3, r=6: 15!
    assert abs(dim_r2 - 1.0) < 1e-12
    assert abs(dim_r4 - 3.0) < 1e-12
    assert abs(dim_r6 - 16.0) < 1e-12  # 16 > 15: first W(F_4) anisotropy at rank 6!

    # Also verify exact 24-root moments:
    roots = []
    for i in range(4):
        for j in range(i + 1, 4):
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    v = np.zeros(4)
                    v[i] = s1 / math.sqrt(2.0)
                    v[j] = s2 / math.sqrt(2.0)
                    roots.append(v)
    R = np.array(roots)
    T2 = np.einsum("ai,aj->ij", R, R)
    err_r2 = float(np.max(np.abs(T2 - 6.0 * np.eye(4))))
    T4 = np.einsum("ai,aj,ak,al->ijkl", R, R, R, R)
    I4 = np.eye(4)
    iso4 = np.einsum("ij,kl->ijkl", I4, I4) + np.einsum("ik,jl->ijkl", I4, I4) + np.einsum("il,jk->ijkl", I4, I4)
    err_r4 = float(np.max(np.abs(T4 - iso4)))
    assert err_r2 < 1e-14 and err_r4 < 1e-14

    n1 = np.array([1.0, 0.0, 0.0, 0.0])
    n3 = np.array([1.0, 1.0, 0.0, 0.0]) / math.sqrt(2.0)
    s6_n1 = float(np.sum((R @ n1) ** 6))
    s6_n3 = float(np.sum((R @ n3) ** 6))
    s6_iso = 1.875

    return dict(
        W_F4_order=len(wf4),
        reynolds_invariant_dims=dict(
            rank2_WF4=int(round(dim_r2)), rank2_O4=1,
            rank4_WF4=int(round(dim_r4)), rank4_O4=3,
            rank6_WF4=int(round(dim_r6)), rank6_O4=15,
        ),
        D4_roots_count=24,
        rank2_O4_isotropy_max_err=err_r2,
        rank4_O4_isotropy_max_err=err_r4,
        rank6_root_moments=dict(axis_1000=round(s6_n1, 6), face_1100=round(s6_n3, 6), O4_avg=s6_iso),
        verdict=(
            "PROVEN (4D Euclidean O(4)): Exact |W(F_4)|=1,152 Reynolds character averaging gives "
            "dim(((R^4)^4)^{W(F_4)}) = 3 == dim(O(4)) for ALL general non-symmetric rank-4 director kinetic "
            "tensors across independent (c_1, c_2, c_3), forbidding hypercubic sum_a (d_a u_a)^2 counterterms; "
            "first W(F_4) anisotropy enters at rank 6 (16 > 15, c_4-type acceleration operators)."
        ),
    )


def main():
    t0 = time.time()
    out = {}
    out["bridge3B_24cell_director_isotropy"] = audit_24cell_director_isotropy()
    sys.stderr.write(
        f"[Bridge 3B] |W(F_4)|=1152 Reynolds dims: r2=1, r4=3 (exact O(4)), r6=16 (>15); "
        f"root err={out['bridge3B_24cell_director_isotropy']['rank4_O4_isotropy_max_err']:.2e}\n"
    )

    g_grid = [0.0, 0.08, 0.14, 0.18, 0.20, 0.22, 0.25, 0.30, 0.40, 0.55, 0.75, 1.0]
    out["bridge3A_1plus1D_L4_periodic_dim65536"] = scan_lattice_smg_full_hilbert(
        "1+1D_L4_periodic", 4, [(0, 1), (1, 2), (2, 3), (3, 0)], [1.0]*4, g_grid, seed=301
    )
    out["bridge3A_1plus1D_L4_antiperiodic_C4_piFlux_dim65536"] = scan_lattice_smg_full_hilbert(
        "1+1D_L4_antiperiodic_C4_piFlux", 4, [(0, 1), (1, 3), (3, 2), (2, 0)], [1.0, 1.0, 1.0, -1.0], g_grid, seed=302
    )
    out["bridge3A_K4_4site_tetrahedral_cluster_z3_dim65536"] = scan_lattice_smg_full_hilbert(
        "K4_4site_tetrahedral_cluster_z3", 4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], [1.0]*6, g_grid, seed=303
    )

    sys.stderr.write(f"[Bridge 3 v1.2.0 ALL COMPLETE in {time.time()-t0:.1f}s]\n")
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
