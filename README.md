# Tau Core Paper 30 (`v1.6.0`)

**Four-Simplex $\mathrm{Spin}(10)$ Particle Genesis, 16-Parameter Parity--Hermitian Flavor Closure, Dimension-5 & Dimension-6 Proton Decay, Cosmological/UV/RPV Gate Closure, and 24-Cell Chiral Lattice Symmetric Mass Generation**

- **Author:** Jozsef Olcsak (ORCID: [0009-0003-7311-6522](https://orcid.org/0009-0003-7311-6522))
- **Date:** 24 November 2026 (`v1.6.0`, Pillar 3D hostile-referee-certified UV/leptogenesis/RPV gate closure on the two surviving proton-decay branches: `PROVEN NO-GO` on $\mathbb{Z}_2^{(210)}$ $D$-parity phase shielding [$d\bar\theta/d\varepsilon_d = -10.27 \implies |\varepsilon_d| < 1.27\times 10^{-11}$], exact 2-flavor Nardi ODE leptogenesis sign-flip/shortfall [$\eta_B = -1.13\times 10^{-12}$] & minimal Starobinsky `PROVEN NO-GO` on Branch I; and clean `5.48`-decade bilinear RPV window [$m_\nu^{\mathrm{RPV}} \in [1.19\times 10^{-14}, 3.56\times 10^{-9}]\;\mathrm{eV}$, BBN+washout+LHC+p-decay safe] restoring 100% AeST DM and lifting Split-SUSY gravitino [$\tau_{3/2}=9.56\;\mathrm{ms}$] hadronic BBN ceiling to $T_{\mathrm{rh}}^{\max} = 1.07\times 10^{11}\;\mathrm{GeV}$ above $N_2$ leptogenesis on Branch II).
- **Manuscripts:**
  - English: `Tau_Core_Paper_30_Spin10_Flavor_Hierarchy_SMG_EN.pdf` (`_EN.tex`)
  - Hungarian: `Tau_Core_Paper_30_Spin10_Flavor_Hierarchy_SMG_HU.pdf` (`_HU.tex`)
- **Direct continuation of:** Paper 22 (`K_4` / 1-gen `Spin(10)`), Paper 25 (`10.5281/zenodo.24477423`, charged-fermion RG), Paper 26 (`10.5281/zenodo.24477427`, Strong-CP), and Paper 27 (`10.5281/zenodo.24477434`, 24-cell $D_4/F_4$ continuum lattice).

## Single-Command Reproduction (`python3 -W error`)

```sh
python3 -W error scripts/build_paper30_verification_and_figures.py
python3 -W error scripts/verify_survivor_branches_uv_phase_leptogenesis_rpv.py
```

This executes all four pillars plus Bridge 1, Bridge 3, Pillar 3B (`verify_complete_d5_proton_decay_multitriplet_LLLL_RRRR.py`), Pillar 3C (`verify_deep_proton_decay_d6_unification_splitsusy.py`), and Pillar 3D (`verify_survivor_branches_uv_phase_leptogenesis_rpv.py`) live, writes all verification JSON ledgers in `data/`, and generates all four 300-DPI figures in `figures/`.

## Core Verified Results (4-Tier Epistemic Ledger)

| Pillar | Status | Key Verified Numbers |
|---|---|---|
| **I. 4-Simplex $\Lambda^{\mathrm{even}}(\mathbb{C}^5)$ $\mathrm{Spin}(10)$ & $|\lambda|=1/4$** | `PROVEN` | 16 SM states, 45 gauge bosons (`12+9+24`), $Q$ err `0.0`, $p+e^-$ err `2.22e-16`, GJ `-3.0`, 3-gamma $\mathbf{120}_H$ antisymmetry `1.11e-16`; constructive $(X',Y')$ in $\mathcal{O}_2, \mathcal{O}_3$; shared transfer eigenvalue $\lambda = -1/4$ on $P_{K_5}$ and $P_{24}$ ($\mathrm{Tr}=0, \mathrm{Tr}\,P_{24}^2=3$); $\mathbb{Z}_6 \subset 2T$ texture zeros for `M16-U11`. |
| **II. 3-Gen Flavor No-Go, $\mathrm{PH16}_{a13,h23}$ & Bridge 1** | `PROVEN NO-GO` + `CONDITIONAL` | Modular $A_4$ (`10277`), spinorial $2T$ (`3523`), non-SUSY/SUSY $\mathbf{10}+\overline{\mathbf{126}}$ (`30.3` / `1259`) excluded; complementary 16-param `PH16_a13_h23` (`dof=2`, $\bar\theta_{\mathrm{QCD,tree}} \equiv 0$) achieves `chi2=0.0297` (non-SUSY), `1.3478` (MSSM $\tan\beta=50$), and `2.25–2.84` (true 3-stage Split-SUSY $\tan\beta=2.0\text{--}2.5$, `2.41` even at $270^\circ$). $U(1)_R$-exact 1-loop Strong-CP: universal Parity `4.09e-13–1.08e-12` (PASS); non-universal `3.21e-8` (`NO-GO`). |
| **III. Hierarchy, $d=5/d=6$ Proton Decay, Split-SUSY Island & Pillar 3D UV/RPV Closure** | `PROVEN NO-GO` + `CONDITIONAL` | Conformal portal, relaxion, $\mathbf{10}\otimes\mathbf{10}$, 9-group PC (`0/15`), uncancelled $12\text{--}30\;\mathrm{TeV}$ Mini-Split ($1{,}137.5 / 308.5\;\mathrm{TeV}$ at $\tan\beta=50$), high-$\tan\beta$ ($\gtrsim 4$) Split SUSY ($m_h\ge 136.8\;\mathrm{GeV}$, $\sim 6\text{--}8\sigma$), minimal/modest-threshold $G_{422D}$ ($\tau=1.16\text{--}1.83\times 10^{31}\;\mathrm{yr}$), 2-bidoublet $G_{422}$ ($\tau=4.43\text{--}5.49\times 10^{33}\;\mathrm{yr}$), and triplets at $M_{\mathrm{PS}}$ are **`PROVEN NO-GO`**. Pillar 3D decisively tilts the two surviving branches toward Branch II: **(I) Single-bidoublet Non-SUSY $G_{422}$** ($\mathbf{210}_H$, $\tau(e^+\pi^0)=5.66\text{--}7.02\times 10^{34}\;\mathrm{yr}$ inside Hyper-K 10-yr, $\mathrm{BR}=0.901$; 100% AeST DM) hits a **`PROVEN NO-GO` on $\mathbb{Z}_2^{(210)}$ $D$-parity phase shielding** ($d\bar\theta/d\varepsilon_d = -10.27 \implies |\varepsilon_d| < 1.27\times 10^{-11}$ or PQ/axion) AND a 2-flavor Nardi ODE leptogenesis sign-flip/shortfall ($\eta_B = -1.13\times 10^{-12}$, `540.4x`; minimal Starobinsky **`PROVEN NO-GO`**, max $2.97\times 10^{-15}$; needs $\delta_{12} = 6.18\times 10^{-4}$); **(II) Low-$\tan\beta$ Split-SUSY Island** ($\tan\beta\in[2.0,2.6]$, $M_{\tilde q}\in[131,214]\;\mathrm{TeV}$, ATLAS-safe $M_{\tilde w}=700\;\mathrm{GeV}$, true 2L Bagnaschi $m_h=125.25\;\mathrm{GeV}$, binding $p\to\pi^+\bar\nu$, right-sign $N_2$ $\eta_B=+7.75\times 10^{-11}$) resolves its gravitino crisis cleanly via a **`5.48`-decade bilinear RPV window** ($m_\nu^{\mathrm{RPV}}\in[1.19\times 10^{-14}, 3.56\times 10^{-9}]\;\mathrm{eV}$, passing BBN, Split-SUSY gauge washout $<5.21\times 10^{-4}\;\mathrm{eV}$, ATLAS/CMS Run-2 LLP $c\tau>100\;\mathrm{m}$, and zero RPV proton decay), giving **100% AeST DM** and raising the Split-SUSY gravitino ($\tau_{3/2}=9.56\;\mathrm{ms}$) Kawasaki et al. (2018) hadronic BBN ceiling to **$T_{\mathrm{rh}}^{\max} = 1.07\times 10^{11}\;\mathrm{GeV}$** cleanly above $N_2$ leptogenesis! |
| **IV. 24-Cell Chiral Fermions, $|W(F_4)|=1{,}152$ Isotropy & 32-Maj. SMG** | `PROVEN` + `OPEN` | Naive 24-cell `72=36+36` doublers; Neuberger Overlap GW err `2.78e-16`; 5D DW leakage `1.14e-5`; exact $|W(F_4)|=1{,}152$ Reynolds averaging proves $\dim((\mathbb{R}^4)^{\otimes 4})^{W(F_4)} = \dim((\mathbb{R}^4)^{\otimes 4})^{O(4)} = 3$ (4D Euclidean $O(4)$, first anisotropy at rank 6: $16>15$); full even+odd Hilbert-space (`dim=65,536`) 32-Majorana SMG scans converge to $\Delta E_{\mathrm{true}}=14.0$ on 1+1D periodic $L=4$ (`237.2`), 1+1D antiperiodic $C_4$ (`471.9`), and 4-site $K_4$ cluster (`483.0` at $g_c=0.22$, VEV `<=2.40e-15`). Full 4D interacting chiral-gauge continuum limit remains `OPEN`; `NATURE_LEVEL_CLOSURE = false`. |

## Building the PDFs

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error Tau_Core_Paper_30_Spin10_Flavor_Hierarchy_SMG_EN.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error Tau_Core_Paper_30_Spin10_Flavor_Hierarchy_SMG_HU.tex
```
