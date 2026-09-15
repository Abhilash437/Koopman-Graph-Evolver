# Author Rebuttal — Submission 39
**"Beyond MSE: Geometry-Preserving Latent Dynamics for Long-Horizon Graph Simulation"**

We thank all three reviewers for their constructive and rigorous evaluations. Reviewer **vp7p** recognized the paper's clarity, topology-grounded metrics beyond MSE, and geometry-preserving latent transitions (**Accept, 8/10**). We thank Reviewer **1LAu** (**Rating 4**) for architectural clarification, loss formalization, numerical standardization, and the critical $\mathcal{L}_{\text{collapse}}$/$\mathcal{L}_{\text{iso}}$ attribution question. We thank Reviewer **t7BM** (**Rating 4**) for pressing on claim precision (W6), alternative stabilizers, and distributional probes. All requested clarifications and new experiments are incorporated into the revised manuscript (Appendices **C–F**).

---

# Response to Reviewer vp7p (Rating: 8 / Accept)

### Q1: Figure 2 Caption & Subplot Ratios

**Comment:** *Subplots 2–4 are ratios where closer to 1 is better; SEGNO appears slightly closer to 1 on coordinate retention than E-GKN.*

**Response:** We agree ideal retention $=1.0$. On MD22 DHA, SEGNO attains marginally closer pairwise coordinate retention ($\approx 0.982$ vs. $\approx 0.970$ for E-GKN) via steerable harmonic message passing and bond-distance features. E-GKN achieves lower cumulative rollout MSE and bounds latent collapse ($\rho(K)=1.0$) by construction without engineered edge features.

**Revision:** Fig. 2 caption and Sec. 4.8: top = rollout MSE; rows 2–4 = pairwise coordinate / node-embedding / **latent-norm** ratios (not physical energy; DHA-only subplot, not a 14-system energy table).

### Q2: Equations for Latent Collapse & Coordinate Retention

Added in Sec. 4.8 (Eq. 8). With pairwise distance matrices $D_X(t)$, $D_H(t)$ from decoded coordinates / latents and Frobenius norm $\|\cdot\|_F$:

$$\text{Coord Retention}(T)=\frac{\|D_X(T)\|_F}{\|D_X(0)\|_F},\qquad \text{Latent Collapse}(T)=1-\frac{\|D_H(T)\|_F}{\|D_H(0)\|_F}.$$

$R_{\text{norm}}$ and $R_{\text{edge}}$ are defined in Appendix G.

---

# Response to Reviewer 1LAu (Rating: 4)

### W1: Architecture & Parameter Scaling

All reported experiments use **Graph-Aware Kronecker** (`GraphAwareKoopmanNet`):

$$A_{\text{glob}}=I_N\otimes A_{\text{self}}+\alpha\,P_{\text{sym}}\otimes A_{\text{edge}},\quad A_{\text{self}},A_{\text{edge}}\in\mathbb{R}^{64\times 64}.$$

1. **Koopman Transition:** $2\times 64^2+1=\mathbf{8{,}193}$ tensor entries ($2\times\frac{64\times 63}{2}+1=\mathbf{4{,}033}$ independent skew DoFs), constant in $N$.
2. **Encoder:** $\mathbf{25{,}536}$.
3. **Decoder MLP:** $\mathrm{Linear}(64N\to 128)+\mathrm{Linear}(128\to 128)+\mathrm{Linear}(128\to 3N)=\mathbf{8{,}579N+16{,}640}$.
4. **Total:** $\mathbf{50{,}369+8{,}579N}$ — aspirin ($N{=}21$): $\mathbf{230{,}528}$; stachyose ($N{=}87$): $\mathbf{796{,}742}$ (exact Table 6 match).

Sec. 3.2 now states $O(N)$ scaling (not $O(N^2 d^2)$).

### W2: Loss Specification

Original Eq. 2 was a 2-term draft. Sec. 3.2 now states the KGE trainer (`compute_loss`) 4-term loss with **fixed** weights (no annealed $\lambda$):

$$\mathcal{L}_{\text{total}}=10\,\mathcal{L}_{\text{recon}}+1\,\mathcal{L}_{\text{dyn}}+2\,\mathcal{L}_{\text{collapse}}+5\,\mathcal{L}_{\text{iso}},$$

- $\mathcal{L}_{\text{recon}}$: teacher-forced autoencoder (encode current graph $\to$ decode $\to$ score vs same-timestep coordinates). **Not** next-frame prediction.
- $\mathcal{L}_{\text{dyn}}$: one-step latent consistency ($\K\mathbf{s}_t$ vs encoder $\mathbf{s}_{t+1}$).
- $\mathcal{L}_{\text{collapse}}=\max(0,0.05-\rho_{\text{rel}})$: encoder anti-freeze hinge. Does **not** train $R_{\text{norm}}$.
- $\mathcal{L}_{\text{iso}}$: bonded-distance MSE on decoded coordinates; **does** push bond margins.

$R_{\text{norm}}=1$ is architectural $SO(n)$ from $\K=\exp(\mathbf{A}_{\mathrm{glob}})$ with implicit $\Delta t=1$. Baseline loss asymmetries (identical-objective bake-off) are **not** claimed here.

### W2 Follow-up: $\lambda_{\text{collapse}}=\lambda_{\text{iso}}=0$

We agree parity alone does not settle attribution when those terms relate to latent motion and bonds. We retrain under $\mathcal{L}_{\text{noreg}}=10\,\mathcal{L}_{\text{recon}}+1\,\mathcal{L}_{\text{dyn}}$, seeds $\{42,1337,2026\}$, horizon 29, MD17/MD22 only (**Appendix D**).

**Aspirin.** Noreg raises KGE drifts (bond $0.0045\to 0.015$ Å; angle $0.091\to 0.311^\circ$), so margins are partly loss-dependent. Under noreg, $R_{\text{norm}}=1$ for KGE on all seeds; seed-averaged angle (and on most pairs bond/torsion/$|R_{\text{edge}}-1|$) still favor KGE, while MSE favors G-GRU.

**Expand** (malonaldehyde, benzene, ethanol, dha, at-at; $15$ pairs): KGE wins bond $13/15$, angle $14/15$, torsion $11/15$, $|R_{\text{norm}}-1|$ $15/15$, $|R_{\text{edge}}-1|$ $13/15$. With aspirin ($18$): $15/18$, $17/18$, $13/18$, $18/18$, $15/18$. Benzene is mixed (bonds nearly tied; torsion lower for G-GRU). Seed-averaged MSE favors G-GRU on every system here.

**Takeaway.** Removing the two terms changes margins, but under noreg the structural edge largely survives; the MSE paradox remains.

### W3: Number Reconciliation

**W3.1:** Springs G-GRU MSE standardized to multi-seed $0.0531\pm 0.0015$ (not intermediate $T{=}15$ / single-seed values). **W3.2:** Table 2 = 3-seed aggregate; Tables 7–8 = single-seed sweep (captions fixed). **W3.3:** Removed unsupported $33\times$; cite verified $109.5\times$ on unconstrained `ac-ala3-nhme` (Sec. 4.6). **W3.4:** $4.9\times 10^{27}$ = unbounded EGNN instability; Table 3 = standardized $k{=}29$ ($4.40\times 10^{-2}$).

### W4: Hyperparameters & Compute (Appendix C)

Tuned on aspirin/DHA validation (LR $\{10^{-4},5\times 10^{-4},10^{-3},2\times 10^{-3}\}$, $d\in\{32,64,128\}$, layers $\{2,4,6\}$, batch $\{16,32\}$, loss weights), then **fixed** across all 14 systems. Hardware: A100 (80GB) / RTX 3090 (24GB). Runtime: $\approx 3$–$5$ min/100 epochs small MD17/$N$-body; $\approx 15$–$35$ min large MD22; eval $<2$ min/system.

---

# Response to Reviewer t7BM (Rating: 4)

Manuscript updates: (i) abstract + Contribution 3 drop “structural-energy”; (ii) Appendix D noreg ablation; (iii) Appendix E G-GRU pushforward/noise; (iv) Appendix F RDF pilot.

**W1.** Scope is a temporal inductive bias on standard MD17/MD22/$N$-body rollouts, not a new domain finding. $SO(n)$ is a latent-cell constraint (Remark 1); we do not claim Liouville/Hamiltonian decoded mechanics.

**W2.** Higher KGE MSE is the stated MSE paradox; primary claims are bond/angle/torsion / $R_{\text{norm}}$ / $R_{\text{edge}}$. SEGNO is Sec. 4.8 (MD22 subset). Stabilizers: Appendix E / Q1.

**W3.** EGNN NaNs on $4/14$ = unbounded latent growth under the shared protocol. NequIP/MACE as multi-step simulators are outside this isolation design (KGE vs G-GRU / Flat-K).

**W4.** Evaluated model is Kronecker $O(N)$, not dense $(Nd)^2$. Runtimes in Appendix C; no paired peak-memory table yet (can add if requested).

**W5.** Agreed: Theorem 1 is latent-only (Limitation 1); decoded bond/angle/torsion gains are empirical.

**W6.** “Structural-energy drift” was unsupported. Wilcoxon Table 1 covers bond, angle, torsion, and $|R_{\text{norm}}-1|$ only; Fig. 2 row 4 is a DHA **latent-norm** subplot and cannot support a 14-system energy claim.

**Loss ablation.** Same as 1LAu / Appendix D: under noreg, structural metrics largely still favor KGE; MSE favors G-GRU.

### Answers to questions

**Q1 (pushforward / training noise / symplectic).** Vanilla G-GRU already uses $H{=}4$, $\sigma{=}0$. We retrain G-GRU only under the same full loss as KGE (**no symplectic**): pf-only ($H{=}16$), noise-only ($H{=}4$, $\sigma{=}0.01$), and both ($H{=}16$, $\sigma{=}0.01$) on **8 MD17 + 4 MD22**, seed 42, vs fixed KGE full checkpoints (**Appendix E**; bond/angle and torsion tables). Noise alone ≈ vanilla G-GRU; pushforward helps some molecules and hurts others; combined ≈ pushforward. **No arm closes the structural gap** ($0/12$ with bond, angle, and torsion all within $2\times$ of KGE). Under this protocol, $SO(n)$ is not replaced by these stabilizers.

**Q2 (EGNN).** Same composite loss/protocol as other baselines (no EGNN-only clipping, noise, or special $\Delta t$). NaN/overflow on $4/14$; a finite $k{=}29$ checkpoint can still mask longer instability (Sec. 4.6). No exhaustive EGNN retune claimed.

**Q3 (RDFs / spectra / energy).** Spectra not reported; energy claim removed (W6). Decoded RDF pilot (**Appendix F**): full-loss KGE vs vanilla G-GRU, 12 molecules × 3 seeds ($36$ runs). Fixed-time L1 usually favors G-GRU (absolute RDF $30/36$; bonded $25/36$ at $t{=}29$); **$\Delta$L1 favors KGE** ($29/36$ absolute; $30/36$ bonded). Stability nuance only—not better absolute $g(r)$; primary claims remain bond/angle/torsion / $R_{\text{norm}}$ / $R_{\text{edge}}$.

**Q4.** Appendix C: $\approx 15$–$35$ min per 100-epoch large MD22 run. No paired peak-memory table yet.

**Q5.** $A_{\text{skew}}+\varepsilon S$ not evaluated; controlled dissipation remains future work (Limitation 2).

---

We hope the clarifications for Reviewers vp7p and 1LAu, together with Appendices D–F and the W6 wording fix for Reviewer t7BM, address the remaining concerns and make the evidence base fully transparent.
