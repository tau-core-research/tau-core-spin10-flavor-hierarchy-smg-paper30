# Tau Core Paper 30 (`v1.5.0`)

**Four-Simplex $\mathrm{Spin}(10)$ Particle Genesis, 16-Parameter Parity--Hermitian Flavor Closure, Dimension-5 & Dimension-6 Proton Decay, Cosmological Leptogenesis/DM Showdown, and 24-Cell Chiral Lattice Symmetric Mass Generation**

- **Author:** Jozsef Olcsak (ORCID: [0009-0003-7311-6522](https://orcid.org/0009-0003-7311-6522))
- **Date:** 24 November 2026 (`v1.5.0`, Round-3 hostile JHEP/PRD referee closure + Step 2 cosmological showdown: exact Dirac–Fierz $d=6$ rates, 1-loop Weinberg–Hall 2-loop non-SUSY unification ruling out minimal/modest $G_{422D}$ and 2-bidoublet $G_{422}$ while placing single-bidoublet $G_{422}$ squarely inside Hyper-K 10-yr reach, exact factor-of-3 $\mathbf{120}_H(15,2,2)$ Clebsch, true 2-loop Bagnaschi et al. Split-SUSY Higgs-mass RGEs + 3-stage Split-SUSY fermion targets + ATLAS-safe $M_{\tilde w}=700\;\mathrm{GeV}$ locking the **Low-$\tan\beta$ Split-SUSY Island** $\tan\beta\in[2.0,2.6]$, and thermal $N_1/N_2$ leptogenesis + wino/gravitino cosmology).
- **Manuscripts:**
  - English: `Tau_Core_Paper_30_Spin10_Flavor_Hierarchy_SMG_EN.pdf` (`_EN.tex`)
  - Hungarian: `Tau_Core_Paper_30_Spin10_Flavor_Hierarchy_SMG_HU.pdf` (`_HU.tex`)
- **Direct continuation of:** Paper 22 (`K_4` / 1-gen `Spin(10)`), Paper 25 (`10.5281/zenodo.24477423`, charged-fermion RG), Paper 26 (`10.5281/zenodo.24477427`, Strong-CP), and Paper 27 (`10.5281/zenodo.24477434`, 24-cell $D_4/F_4$ continuum lattice).

## Single-Command Reproduction (`python3 -W error`)

```sh
python3 -W error scripts/build_paper30_verification_and_figures.py
```

This executes all four pillars plus Bridge 1, Bridge 3, Pillar 3B (`verify_complete_d5_proton_decay_multitriplet_LLLL_RRRR.py`), and Pillar 3C (`verify_deep_proton_decay_d6_unification_splitsusy.py`) live, writes `data/paper30_master_verification_results.json`, `data/bridge1_results.json`, `data/bridge3_results.json`, `data/complete_d5_proton_decay_results.json`, and `data/deep_proton_decay_d6_splitsusy_results.json`, and generates all four 300-DPI figures in `figures/`.

## Core Verified Results (4-Tier Epistemic Ledger)

| Pillar | Status | Key Verified Numbers |
|---|---|---|
| **I. 4-Simplex $\Lambda^{\mathrm{even}}(\mathbb{C}^5)$ $\mathrm{Spin}(10)$ & $|\lambda|=1/4$** | `PROVEN` | 16 SM states, 45 gauge bosons (`12+9+24`), $Q$ err `0.0`, $p+e^-$ err `2.22e-16`, GJ `-3.0`, 3-gamma $\mathbf{120}_H$ antisymmetry `1.11e-16`; constructive $(X',Y')$ in $\mathcal{O}_2, \mathcal{O}_3$; shared transfer eigenvalue $\lambda = -1/4$ on $P_{K_5}$ and $P_{24}$ ($\mathrm{Tr}=0, \mathrm{Tr}\,P_{24}^2=3$); $\mathbb{Z}_6 \subset 2T$ texture zeros for `M16-U11`. |
| **II. 3-Gen Flavor No-Go, $\mathrm{PH16}_{a13,h23}$ & Bridge 1** | `PROVEN NO-GO` + `CONDITIONAL` | Modular $A_4$ (`10277`), spinorial $2T$ (`3523`), non-SUSY/SUSY $\mathbf{10}+\overline{\mathbf{126}}$ (`30.3` / `1259`) excluded; complementary 16-param `PH16_a13_h23` (`dof=2`, $\bar\theta_{\mathrm{QCD,tree}} \equiv 0$) achieves `chi2=0.0297` (non-SUSY), `1.3478` (MSSM $\tan\beta=50$), and `2.25–2.84` (true 3-stage Split-SUSY $\tan\beta=2.0\text{--}2.5$, `2.41` even at $270^\circ$). $U(1)_R$-exact 1-loop Strong-CP: universal Parity `4.09e-13–1.08e-12` (PASS); non-universal `3.21e-8` (`NO-GO`). |
| **III. Hierarchy, $d=5/d=6$ Proton Decay, Split-SUSY Island & Cosmology** | `PROVEN NO-GO` + `CONDITIONAL` | Conformal portal, relaxion, $\mathbf{10}\otimes\mathbf{10}$, 9-group PC (`0/15`), uncancelled $12\text{--}30\;\mathrm{TeV}$ Mini-Split ($1{,}137.5 / 308.5\;\mathrm{TeV}$ at $\tan\beta=50$), high-$\tan\beta$ ($\gtrsim 4$) Split SUSY ($m_h\ge 136.8\;\mathrm{GeV}$, $\sim 6\text{--}8\sigma$ with theory error), minimal/modest-threshold $G_{422D}$ ($\tau=1.16\text{--}1.83\times 10^{31}\;\mathrm{yr}$), 2-bidoublet $G_{422}$ ($\tau=4.43\text{--}5.49\times 10^{33}\;\mathrm{yr}$), and triplets at $M_{\mathrm{PS}}$ ($M_{T_{10}}>5.43\times 10^{13}, M_{T_{126}}>9.31\times 10^{13}, M_{T_{120}}>8.75\times 10^{11}\;\mathrm{GeV}$) are **`PROVEN NO-GO`**. Two clean zero-triplet-tuning routes survive: **(I) Single-bidoublet Non-SUSY $G_{422}$** ($\mathbf{210}_H$, $M_{\mathrm{PS}}=1.71\times 10^{11}$, $M_U=7.27\times 10^{15}\;\mathrm{GeV}$, $\tau(e^+\pi^0)=5.66\text{--}7.02\times 10^{34}\;\mathrm{yr}$ inside Hyper-K 10-yr reach at zero threshold, $\mathrm{BR}=0.901$; 100% AeST DM; thermal $N_1$ $\eta_B=+1.34\times 10^{-11}$, $46\times$ short; UV $D$-parity phase shield needed); **(II) Low-$\tan\beta$ Split-SUSY Island** ($\tan\beta\in[2.0,2.6]$, $M_{\tilde q}\in[131,214]\;\mathrm{TeV}$, ATLAS-safe $M_{\tilde w}=700\;\mathrm{GeV}$, true 2L Bagnaschi $m_h=125.25\;\mathrm{GeV}$, 2-loop $M_U=1.37\times 10^{16}\;\mathrm{GeV}$, binding $p\to\pi^+\bar\nu$ in Hyper-K/DUNE, subdominant wino $f_{\mathrm{DM}}=6.1\%$ with $270\times$ gamma suppression, $N_2$ $e$-flavor $\eta_B=+7.75\times 10^{-11}$, heavy gravitino $\tau_{3/2}=1.4\times 10^{-2}\;\mathrm{s}$ caps $T_{\mathrm{rh}}<2.7\times 10^8\;\mathrm{GeV}$ under exact $R$-parity, requiring tiny RPV or low-$T_{\mathrm{rh}}$ baryogenesis); plus **(III)** $\sim 400\times$ 11-ch.\ DMM. |
| **IV. 24-Cell Chiral Fermions, $|W(F_4)|=1{,}152$ Isotropy & 32-Maj. SMG** | `PROVEN` + `OPEN` | Naive 24-cell `72=36+36` doublers; Neuberger Overlap GW err `2.78e-16`; 5D DW leakage `1.14e-5`; exact $|W(F_4)|=1{,}152$ Reynolds averaging proves $\dim((\mathbb{R}^4)^{\otimes 4})^{W(F_4)} = \dim((\mathbb{R}^4)^{\otimes 4})^{O(4)} = 3$ (4D Euclidean $O(4)$, first anisotropy at rank 6: $16>15$); full even+odd Hilbert-space (`dim=65,536`) 32-Majorana SMG scans converge to $\Delta E_{\mathrm{true}}=14.0$ on 1+1D periodic $L=4$ (`237.2`), 1+1D antiperiodic $C_4$ (`471.9`), and 4-site $K_4$ cluster (`483.0` at $g_c=0.22$, VEV `<=2.40e-15`). Full 4D interacting chiral-gauge continuum limit remains `OPEN`; `NATURE_LEVEL_CLOSURE = false`. |

## Building the PDFs

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error Tau_Core_Paper_30_Spin10_Flavor_Hierarchy_SMG_EN.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error Tau_Core_Paper_30_Spin10_Flavor_Hierarchy_SMG_HU.tex
```
