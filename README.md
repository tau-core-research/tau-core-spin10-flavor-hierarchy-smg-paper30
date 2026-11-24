# Tau Core Paper 30 (`v1.3.0`)

**Four-Simplex $\mathrm{Spin}(10)$ Particle Genesis, 16-Parameter Parity--Hermitian Flavor Closure, Dimension-5 Proton Decay, and 24-Cell Chiral Lattice Symmetric Mass Generation**

- **Author:** Jozsef Olcsak (ORCID: [0009-0003-7311-6522](https://orcid.org/0009-0003-7311-6522))
- **Date:** 24 November 2026 (`v1.3.0`, complete multi-triplet $T_{10}+T_{126}+T_{120}$ $LLLL+RRRR$ $d=5$ proton-decay audit + exact $32\times 32$ 4-simplex Clifford Clebsch--Gordan theorem + simultaneous light-Higgs unitarity + 11-channel Pati--Salam DMM cancellation, building on `v1.2.0`'s Round-2 referee-hardened $P_{24}$ spectrum, $U(1)_R$-exact 1-loop SUSY Strong-CP, smooth $\delta_{\mathrm{CP}}^\ell$ profile, 2-loop MSSM DW unification, full even+odd 32-Majorana SMG scans, and exact $|W(F_4)|=1{,}152$ Reynolds character proof of 4D Euclidean $O(4)$ rank-4 director isotropy).
- **Manuscripts:**
  - English: `Tau_Core_Paper_30_Spin10_Flavor_Hierarchy_SMG_EN.pdf` (`_EN.tex`)
  - Hungarian: `Tau_Core_Paper_30_Spin10_Flavor_Hierarchy_SMG_HU.pdf` (`_HU.tex`)
- **Direct continuation of:** Paper 22 (`K_4` / 1-gen `Spin(10)`), Paper 25 (`10.5281/zenodo.24477423`, charged-fermion RG), Paper 26 (`10.5281/zenodo.24477427`, Strong-CP), and Paper 27 (`10.5281/zenodo.24477434`, 24-cell $D_4/F_4$ continuum lattice).

## Single-Command Reproduction (`python3 -W error`)

```sh
python3 -W error scripts/build_paper30_verification_and_figures.py
```

This executes all four pillars plus Bridge 1, Bridge 3, and Pillar 3B (`verify_complete_d5_proton_decay_multitriplet_LLLL_RRRR.py`) live in `~15 s`, verifies every algebraic, 1-/2-loop RG, $U(1)_R$-exact $6\times 6$ squark--gluino, smooth $\delta_{\mathrm{CP}}^\ell$ profile, complete multi-triplet $LLLL+RRRR$ $d=5$ proton-decay, partial-compositeness, $|W(F_4)|=1{,}152$ Reynolds director-isotropy, chiral-lattice, and full even+odd 32-Majorana (`dim=65,536`) many-body SMG invariant to machine precision, writes `data/paper30_master_verification_results.json`, `data/bridge1_results.json`, `data/bridge3_results.json`, and `data/complete_d5_proton_decay_results.json`, and generates all four 300-DPI figures in `figures/`.

## Core Verified Results (4-Tier Epistemic Ledger)

| Pillar | Status | Key Verified Numbers |
|---|---|---|
| **I. 4-Simplex $\Lambda^{\mathrm{even}}(\mathbb{C}^5)$ $\mathrm{Spin}(10)$ & $|\lambda|=1/4$** | `PROVEN` | 16 SM states, 45 gauge bosons (`12+9+24`), $Q$ err `0.0`, $p+e^-$ err `2.22e-16`, GJ `-3.0`, 3-gamma $\mathbf{120}_H$ antisymmetry `1.11e-16`; shared transfer eigenvalue $\lambda = -1/4$ on $P_{K_5}$ (mult 4) and 24-cell $P_{24}$ ($\{1^{(1)},\tfrac{1}{2}^{(4)},0^{(9)},-\tfrac{1}{4}^{(8)},-\tfrac{1}{2}^{(2)}\}$, $\mathrm{Tr}=0$, $\mathrm{Tr}\,P_{24}^2=3$); $\mathbb{Z}_6 \subset 2T$ texture zeros for `M16-U11` (distinct from `PH16_a13_h23`). |
| **II. 3-Gen Flavor No-Go, $\mathrm{PH16}_{a13,h23}$ & Bridge 1** | `PROVEN NO-GO` + `CONDITIONAL` | Modular $A_4$ (`chi2=10277`), spinorial $2T$ (`3523`), non-SUSY/SUSY $\mathbf{10}+\overline{\mathbf{126}}$ (`30.3` / `1259`) excluded; complementary 16-param `PH16_a13_h23` (`dof=2`, $\bar\theta_{\mathrm{QCD,tree}} \equiv 0$) achieves `chi2=0.0297` (non-SUSY) and `chi2=1.3478` (1-loop MSSM, $\Delta_b=0$). Smooth $\delta_{\mathrm{CP}}^\ell$ scan: perturbative everywhere, $[315^\circ, 90^\circ]$ $<2\sigma$, $270^\circ$ disfavored at $3.15\sigma$ (`chi2=11.28`), $180^\circ$ at $4.96\sigma$ (`chi2=25.93`). $U(1)_R$-exact (`<=1.35e-13`) 1-loop Strong-CP: universal Parity `4.09e-13–1.08e-12` (PASS); non-universal `3.21e-8` (`NO-GO`); $|\phi_{M_3}| \lesssim 4.2\times 10^{-11}$. |
| **III. Hierarchy, 9-Group PC & Complete Multi-Triplet $LLLL+RRRR$ $d=5$ $p$-Decay** | `PROVEN NO-GO` + `CONDITIONAL` | Conformal portal (`Delta_FT=3.1e17`), relaxion (`M_max=3.22e10`), $\mathbf{10}\otimes\mathbf{10}$ (`0` doublets), and 9-group complete-$\mathrm{Spin}(10)$ PC (`0/15` survivors) excluded. Exact $32\times 32$ Clifford Clebsches ($T_{10}\to 1$, $T_{126}(6,1,1)\to\sqrt{3}$, $T_{120}(6,1,1)\to$ pure $RRRR$ $\sqrt{2}$) + simultaneous light-Higgs unitarity ($\sum|U_u|^2=0.999, \sum|V_d|^2=0.987$) prove a sharp **`PROVEN NO-GO`** against un-fine-tuned $12\text{--}30\;\mathrm{TeV}$ Mini-Split SUSY $\mathrm{PH16}_{a13,h23}$: $RRRR$ beats $LLLL$ by $99\times$ even on isolated $T_{10}$ ($M_{\tilde q}^{\min}=255.2\;\mathrm{TeV}$ at $\mu=2\;\mathrm{TeV}$), and $T_{126}$ ($310.6\;\mathrm{TeV}$) + $T_{120}$ ($693.0\;\mathrm{TeV}$) bypass DW splitting of $\mathbf{10}_H$ alone (combined uncancelled $M_{\tilde q}^{\min}=596.4\;\mathrm{TeV}$, $\Delta_{\mathrm{FT}}=7.25\times 10^7$). Clean surviving routes: (i) Non-SUSY $\mathrm{PH16}_{a13,h23}$ (zero $d=5$), (ii) High-Scale Split SUSY ($\ge 600\;\mathrm{TeV}$), or (iii) $\sim 400\times$ fine-tuned 11-channel DMM cancellation ($\tau_K=8.70\times 10^{34}\;\mathrm{yr}$ real). |
| **IV. 24-Cell Chiral Fermions, $|W(F_4)|=1{,}152$ Isotropy & 32-Maj. SMG** | `PROVEN` + `OPEN` | Naive 24-cell `72=36+36` doublers; Neuberger Overlap GW err `2.78e-16`; 5D DW leakage `1.14e-5`; exact $|W(F_4)|=1{,}152$ Reynolds averaging proves $\dim((\mathbb{R}^4)^{\otimes 4})^{W(F_4)} = \dim((\mathbb{R}^4)^{\otimes 4})^{O(4)} = 3$ (4D Euclidean $O(4)$, first anisotropy at rank 6: $16>15$); full even+odd Hilbert-space (`dim=65,536`) 32-Majorana SMG scans converge to $\Delta E_{\mathrm{true}}=14.0$ on 1+1D periodic $L=4$ (peak `237.2`, non-monotonic vs $L=3$'s `442.6`), 1+1D antiperiodic $C_4$ (`471.9`), and 4-site $K_4$ cluster (`483.0` at $g_c=0.22$, VEV `<=2.40e-15`). Full 4D interacting chiral-gauge continuum limit remains `OPEN`; `NATURE_LEVEL_CLOSURE = false`. |

## Building the PDFs

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error Tau_Core_Paper_30_Spin10_Flavor_Hierarchy_SMG_EN.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error Tau_Core_Paper_30_Spin10_Flavor_Hierarchy_SMG_HU.tex
```
