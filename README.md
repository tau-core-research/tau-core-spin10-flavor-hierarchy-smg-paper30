# Tau Core Paper 30 (`v1.0.0`)

**Four-Simplex $\mathrm{Spin}(10)$ Particle Genesis, 16-Parameter Parity--Hermitian Flavor Closure, Dimension-5 Proton Decay, and 24-Cell Chiral Lattice Symmetric Mass Generation**

- **Author:** Jozsef Olcsak (ORCID: [0009-0003-7311-6522](https://orcid.org/0009-0003-7311-6522))
- **Date:** 24 November 2026
- **Manuscripts:**
  - English: `Tau_Core_Paper_30_Spin10_Flavor_Hierarchy_SMG_EN.pdf` (`_EN.tex`)
  - Hungarian: `Tau_Core_Paper_30_Spin10_Flavor_Hierarchy_SMG_HU.pdf` (`_HU.tex`)
- **Direct continuation of:** Paper 22 (`K_4` / 1-gen `Spin(10)`), Paper 25 (`10.5281/zenodo.24477423`, charged-fermion RG), Paper 26 (`10.5281/zenodo.24477427`, Strong-CP), and Paper 27 (`10.5281/zenodo.24477434`, 24-cell $D_4/F_4$ continuum lattice).

## Single-Command Reproduction (`python3 -W error`)

```sh
python3 -W error scripts/build_paper30_verification_and_figures.py
```

This executes all four pillars live in `~6.5 s`, verifies every algebraic, RG, flavor, proton-decay, partial-compositeness, chiral-lattice, and many-body SMG invariant to machine precision, writes `data/paper30_master_verification_results.json`, and generates all four 300-DPI figures in `figures/`.

## Core Verified Results (4-Tier Epistemic Ledger)

| Pillar | Status | Key Verified Numbers |
|---|---|---|
| **I. 4-Simplex $\Lambda^{\mathrm{even}}(\mathbb{C}^5)$ $\mathrm{Spin}(10)$ & $|\lambda|=1/4$** | `PROVEN` | 16 SM states, 45 gauge bosons (`12+9+24`), $Q$ err `0.0`, $p+e^-$ err `2.22e-16`, GJ `-3.0`, 3-gamma $\mathbf{120}_H$ antisymmetry `1.11e-16`; shared transfer eigenvalue $\lambda = -1/4$ on $K_5$ (mult 4) and 24-cell $P_{24}$ (mult 8); $\mathbb{Z}_6 \subset 2T$ texture zeros. |
| **II. 3-Gen Flavor No-Go & $\mathrm{PH16}_{a13,h23}$** | `PROVEN NO-GO` + `CONDITIONAL` | Modular $A_4$ (`chi2=10277`), spinorial $2T$ (`3523`), non-SUSY/SUSY $\mathbf{10}+\overline{\mathbf{126}}$ (`30.3` / `1259`) excluded; complementary 16-param `PH16_a13_h23` (`dof=2`, $\bar\theta_{\mathrm{QCD,tree}} \equiv 0$) achieves `chi2=0.0297` (`max_pull=0.13s`, `max|Y|=0.249`) in non-SUSY and `chi2=1.3478` (`max_pull=1.00s`, `max|Y|=0.534`, $\Delta_b=0$) in 1-loop MSSM. |
| **III. Hierarchy, 9-Group PC & $d=5$ Proton Decay** | `PROVEN NO-GO` + `CONDITIONAL` | Conformal portal (`Delta_FT=3.1e17`), relaxion (`M_max=3.22e10`), $\mathbf{10}\otimes\mathbf{10}$ (`0` doublets), 9-group complete-$\mathrm{Spin}(10)$ PC (`0/15` survivors), and naive TeV-SUSY $d=5$ $p$-decay (`3.81e27 yr << 5.9e33 yr` Super-K) excluded; Dimopoulos--Wilczek unique $M'=2.64\times 10^{14}\;\mathrm{GeV}$ ($\Delta\alpha_3^{-1}=0$, `39.25x` triplet suppression) puts Mini-Split squarks `12.0–18.4 TeV` (`Delta_FT=3.47e4`) inside Hyper-K's 10-year window. |
| **IV. 24-Cell Chiral Fermions & 1+1D SMG QPT** | `PROVEN` + `OPEN` | Naive 24-cell `72=36+36` doublers; Neuberger Overlap GW err `2.78e-16`; 5D DW leakage `1.14e-5`; 45-gen $\mathrm{Spin}(10)$ $\mathcal{O}_{10}$ err `0.0`; exact $L=2,3$ (`dim=256, 4096`) 1+1D $\mathrm{Spin}(7)$ SMG QPT at $g_c \approx 0.20\text{--}0.22$ (`-d^2 E_0/dg^2`: `191.8 -> 442.6`, bilinear VEV `<= 1.54e-14`). Full 4D interacting chiral-gauge lattice continuum limit remains `OPEN`; `NATURE_LEVEL_CLOSURE = false`. |

## Building the PDFs

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error Tau_Core_Paper_30_Spin10_Flavor_Hierarchy_SMG_EN.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error Tau_Core_Paper_30_Spin10_Flavor_Hierarchy_SMG_HU.tex
```
