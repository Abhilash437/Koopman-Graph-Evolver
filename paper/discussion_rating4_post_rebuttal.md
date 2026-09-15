# Discussion Reply — Reviewer t7BM (Rating 4 / Post-Rebuttal Update)

**Manuscript updates:** (i) abstract + Contribution 3 no longer claim “structural-energy”; (ii) Appendix D: `L_collapse`/`L_iso`=0 ablation; (iii) Appendix E: G-GRU pushforward/noise (Q1); (iv) Appendix F: decoded RDF pilot (Q3).

We thank the reviewer for the post-rebuttal update and for highlighting W6 and the loss-attribution concern.

## W6 — “structural-energy drift”

We agree the previous phrasing was unsupported. Wilcoxon Table 1 covers bond, angle, torsion, and `|R_norm−1|` only. No Hamiltonian/potential-energy drift is defined in Sec. 4.1 or tabulated in Tables 2/7/8. Fig. 2 row 4 is a DHA **latent-norm** subplot and cannot support a 14-system energy claim. The manuscript now matches Table 1: bond, angle, torsion, and latent-norm-ratio metrics.

## Loss ablation (`L_collapse`, `L_iso`)

We agree shared training terms alone do not settle attribution when `L_collapse`/`L_iso` relate to latent motion and bonds. The requested noreg run (`λ_collapse=λ_iso=0`, seeds `{42,1337,2026}`, 29-step horizon, KGE vs G-GRU) is in **Appendix D** (aspirin full vs noreg; noreg on malonaldehyde, benzene, ethanol, dha, at-at; no N-body).

Headline: under noreg, aspirin KGE drifts rise vs full loss (bond 0.0045→0.015 Å; angle 0.091→0.311°), so those terms affect margins; yet `R_norm=1` for KGE on all reported seeds, and seed-averaged angle (and on most pairs bond/torsion/`|R_edge−1|`) remain lower for KGE than G-GRU, while MSE stays lower for G-GRU. Full tables are in Appendix D.

## W1 — scope / physics motivation

We agree the paper studies a temporal-transition inductive bias on standard MD17/MD22/N-body rollouts, not a new dataset or domain finding. `SO(n)` is a **latent** constraint (not Liouville/Hamiltonian decoded mechanics).

## W2 — MSE, SEGNO, alternative stabilizers

KGE often has higher rollout MSE; that is the stated MSE paradox. The primary claim is topology-grounded drift / `R_norm` / `R_edge`, not MSE minimization. SEGNO is compared only under Sec. 4.8 (MD22 subset). **Q1** (pushforward + training noise on G-GRU; no symplectic) is now in **Appendix E**.

## W3 — EGNN and external force models

EGNN divergence on 4/14 systems is unbounded latent-norm growth under the shared loss/protocol. NequIP and MACE are cited as spatial encoders; wrapping them as multi-step simulators is outside this paper’s isolation design (KGE vs G-GRU / Flat-K).

## W4 — wall-clock / memory

The evaluated model is the graph-aware Kronecker operator (`O(N)`), not dense `(Nd)²`. Appendix C: 3–5 min small systems; 15–35 min large MD22 / 100 epochs. Paired peak-memory vs G-GRU not in the MS; we can add if requested.

## W5 — latent guarantees vs decoded fidelity

Agreed. Limitation 1 already states Theorem 1 applies in latent space only; decoded bond/angle/torsion gains are empirical, not proven transfers of `SO(n)`.

## Answers to questions

**Q1.** We compare longer pushforward and latent training noise on G-GRU (same full loss as KGE; no symplectic). The main-text / vanilla G-GRU baseline already uses a 4-step dyn unroll (H=4, σ=0); we evaluate pf-only (H=16), noise-only (H=4, σ=0.01), and both (H=16, σ=0.01) on **8 MD17 + 4 MD22**, seed 42, vs fixed KGE full checkpoints (**Appendix E**). Noise alone ≈ vanilla G-GRU; pushforward helps some molecules and hurts others; combined ≈ pushforward. **No arm closes the structural gap** (0/12 molecules with bond/angle/torsion all within 2× of KGE). Under this protocol, SO(n) is not replaced by these G-GRU stabilizers.

**Q2.** EGNN used the same composite loss/protocol as other baselines (no EGNN-only clipping, noise, or special `Δt`). Under that protocol it hits NaN/overflow on 4/14 systems. A checkpoint can look finite at `k=29` while longer unconstrained rollout stays unstable (Sec. 4.6). We do not claim an exhaustive EGNN retune that removes all divergence.

**Q3.** Spectra are not reported; “structural-energy” is removed (W6). Decoded RDF pilot (**Appendix F**): full-loss KGE vs G-GRU, 8 MD17 + 4 MD22, seeds `{42,1337,2026}`, 36 runs. Fixed-time L1 usually favors G-GRU (absolute RDF 30/36; bonded 25/36 at t=29); **ΔL1 favors KGE** (29/36 absolute; 30/36 bonded). We do **not** claim better absolute RDFs—stability nuance only; primary claims remain bond/angle/torsion / `R_norm` / `R_edge`.

**Q4.** Appendix C: about 15–35 min per 100-epoch large MD22 run. No paired peak-memory table yet.

**Q5.** `A_skew + εS` is not evaluated. Controlled dissipation remains future work (Limitation 2).

We hope Appendices D–F and the W6 wording fix address the remaining concerns.
