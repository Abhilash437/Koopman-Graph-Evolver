# Post-rebuttal paper↔code audit — LoG 2026 #39

**Paper:** Beyond MSE: Geometry-Preserving Latent Dynamics for Long-Horizon Graph Simulation  
**Forum:** https://openreview.net/forum?id=rWPwsq3M8c  
**Code branch audited:** `dev/rebuttal-1ALu` (exact name present; HEAD `42bac82`)  
**Manuscript:** post-rebuttal PDF (18 pp; anonymous author block)  
**Scope:** investigation only. No new experiments were run; PyTorch is not installed in this audit environment, so parameter counts are from closed-form enumeration of the modules, not `model.parameters()`.

**Verdict (hypothesis check).** The residual integrity risk is **loss-attribution, metric definition, and soft-physics language**, not a hidden node-wise 64×64 operator. The CLI default *is* Graph-Aware Kronecker `GraphAwareKoopmanNet`; Table 6 matches that architecture. The SO(n) construction is implemented as claimed. What still would not survive a methods-conference re-review (or an honest arXiv reader with the repo open) is: (i) headline bond/angle/torsion being trained (via \(L_{\mathrm{iso}}\)) and evaluated under a different reference than Sec. 4.1 writes; (ii) “strict loss parity” being false for G-GRU / Flat-K / SEGNO; (iii) leftover Liouville / energy / pooling language; (iv) incomplete noreg coverage.

---

## 0. Branch and artifact map

| Item | Location |
|------|----------|
| Training entrypoint | `koopman_evolver/cli.py` (`kge train --model koopman` → `GraphAwareKoopmanNet`) |
| Default KGE | `koopman_evolver/models/koopman_net.py` (`GraphAwareKoopmanNet`) |
| Legacy node-wise 64×64 | same file, class `GraphKoopmanNet` (**not** wired into CLI) |
| Loss weights | `koopman_evolver/utils/loss_weights.py` |
| G-GRU / Flat-K / EGNN / SEGNO | `koopman_evolver/models/baselines.py` |
| Trainer (checkpoint on val \(R^2\)) | `koopman_evolver/training/trainer.py` (`GraphAwareTrainer`) |
| Bond/angle/torsion + \(R_{\mathrm{norm}}\)/\(R_{\mathrm{edge}}\) | `koopman_evolver/evaluation/physics_eval.py` |
| Noreg scripts | `run_rebuttal_aspirin.sh`, `run_rebuttal_expand.sh` |
| Q1 pushforward/noise | `scripts/run_q1_pushforward.sh` |
| RDF | `scripts/rdf_from_ckpt.py`, `scripts/run_rdf_full.sh`, `scripts/run_rdf_train_full.sh` |
| README still markets the paper | `README.md` (no `paper/` tree in this repo despite README + `pyproject.toml` claiming `paper/main.tex`) |

---

## 1. Training loss: code vs Sec. 3.2 / Appendix C

### 1.1 Coefficients — match

Revised Eq. (3) / Table 9:

\[
\mathcal{L}_{\mathrm{total}} = 10\,\mathcal{L}_{\mathrm{recon}} + 1\,\mathcal{L}_{\mathrm{dyn}} + 2\,\mathcal{L}_{\mathrm{collapse}} + 5\,\mathcal{L}_{\mathrm{iso}}.
\]

Code defaults (`koopman_evolver/utils/loss_weights.py`):

```python
DEFAULT_LOSS_WEIGHTS = {
    "dyn": 1.0,
    "recon": 10.0,
    "collapse": 2.0,
    "iso": 5.0,
}
```

CLI flags `--lambda-dyn/recon/collapse/iso` default to those values (`cli.py` L105–128). `GraphAwareTrainer` writes the same dict into checkpoints. There is **no** annealed \(\lambda\) anywhere (`grep` finds no scheduler / cosine / anneal). That original-review mismatch is **fixed in the revised PDF and in this branch**, for the models that actually call `get_loss_weights`.

`GraphAwareKoopmanNet.compute_loss` (the reported KGE):

```python
l_dyn = F.mse_loss(h_pred, h_tgt)
...
l_collapse = torch.relu(0.05 - relative_change)
...
l_recon = F.mse_loss(coords_pred, coords_true)
l_iso = F.mse_loss(d_pred, d_true)
...
total_loss = w["dyn"]*l_dyn + w["collapse"]*l_collapse + w["recon"]*l_recon + w["iso"]*l_iso
```

Noreg Eq. (8) is implemented: `--lambda-collapse 0 --lambda-iso 0` keeps \(10\,\mathcal{L}_{\mathrm{recon}}+1\,\mathcal{L}_{\mathrm{dyn}}\) (`run_rebuttal_aspirin.sh` L9–10, L147–148).

### 1.2 Term *definitions* — several paper↔code gaps remain

These are more dangerous than the old 2-term vs 4-term lie, because the revised paper now *names* the four terms but still does not describe the tensors the code actually uses.

| Term | Paper (Sec. 3.2) | Code (KGE) | Integrity impact |
|------|------------------|------------|------------------|
| \(\mathcal{L}_{\mathrm{dyn}}\) | \(\|K_{\mathrm{glob}} s_t - s_{t+1}\|_F^2\) (one-step, concatenated \(s\in\mathbb{R}^{Nd}\)) | One-step MSE on **per-node** latents after `transition_step`. Same spirit. | Low. \(\Delta t\) is **not** a tensor: `K_global = exp(A_glob)` with implicit \(\Delta t=1\) (`koopman_net.py` L98–107; `safe_matrix_exp` has no \(\Delta t\)). |
| \(\mathcal{L}_{\mathrm{recon}}\) | \(\|\hat x_{t+1}-x_{t+1}\|_F^2\) (reads as **next-step** coord prediction) | Decoder applied to the **full encoded window**, vs true coords at those same times — an autoencoder recon of teacher-forced frames, **not** Koopman-rollout \(\hat x_{t+1}\). | Medium. Notation overclaims what is supervised. |
| \(\mathcal{L}_{\mathrm{collapse}}\) | \(\max(0, 0.05-\rho_{\mathrm{rel}})\) with \(\rho_{\mathrm{rel}}\) on **concatenated** \(\|s_{t+1}-s_t\|_2/\|s_t\|_2\) | Hinge on **mean per-node** \(\|h_{t+1}-h_t\|/\|h_t\|\) of **encoder outputs**, not of \(K h_t\), and not of rollout \(R_{\mathrm{norm}}\). | High for the reviewer story (see §3). |
| \(\mathcal{L}_{\mathrm{iso}}\) | mean over edges of squared **bond-length** errors | `F.mse_loss` on bonded distances of **autoencoded** coords vs GT, using `self.edge_index` (distance cutoff graph, not necessarily covalent topology). | High for bond-drift attribution (see §3). |

Skew parameterization: paper writes \(A^\top=-A\). Code stores dense `A_self`, `A_edge` and uses `A - A.T` at forward time. Equivalent manifold; 8,193 tensor entries vs 4,033 independent DoFs, matching the rebuttal arithmetic.

### 1.3 “Strict parity across all baselines” — false

Sec. 3.2: “All baseline models (G-GRU, FLAT-K, EGNN, and SEGNO) are trained under this exact same composite loss formulation for strict parity.”

| Model | Same 4-term weighted sum? | Same \(\mathcal{L}_{\mathrm{dyn}}\)? | Notes |
|-------|---------------------------|--------------------------------------|-------|
| KGE `GraphAwareKoopmanNet` | yes (default weights) | 1-step | Reported method |
| G-GRU `GraphAwareGRUNet` | yes | **No.** Default `unroll_steps=4` averages 4-step pushforward MSE (`baselines.py` L84–86, L142–163). Appendix E later *admits* \(H=4\) as “vanilla,” which **contradicts** main-text parity. |
| Flat-K | **omits \(\mathcal{L}_{\mathrm{iso}}\)** (`baselines.py` L377–383) | 1-step on flat \(Nd\) | Logged `lambda_iso` is unused |
| EGNN (`baselines.EGNNDynamicsNet`) | yes if CLI-trained | 1-step | Duplicate **hardcoded** copy in `models/blocks.py` L269 (`total_loss = l_dyn + 2.0*l_collapse + 10.0*l_recon + 5.0*l_iso`) is **not** CLI-wired but is a landmine |
| E-GKN | yes | 1-step on \([x;h]\) | DummyDecoder means \(\mathcal{L}_{\mathrm{recon}}/\mathcal{L}_{\mathrm{iso}}\) act on **encoder-updated** \(x\) (first 3 channels), not an MLP decode |
| SEGNO | **forces `l_collapse = 0.0`** (`baselines.py` L1031–1033) | 1-step | Comment: “historically omitted l_collapse” |

So the revised paper fixed the *coefficients* and then overstated *identity of the objective*.

### 1.4 Optimizer / schedule vs Appendix C Table 9

| Table 9 claim | Code |
|---------------|------|
| Adam (\(\beta_1=0.9,\beta_2=0.999\)) | `torch.optim.AdamW(..., lr=args.lr, weight_decay=1e-4)` (`cli.py` L308). Default WD is undocumented in the paper. |
| LR \(10^{-3}\), cosine decay to \(10^{-5}\) | Constant `lr=1e-3`. **No scheduler exists** in the repo. |
| Batch 32 (16 MD22) | CLI default `--batch-size 16`. Rebuttal / `run_all.sh` / RDF scripts *do* use 32/16. A bare `kge train --md17 aspirin` does not match Table 9. |
| 100 epochs, \(d=64\), horizon 29, seeds \(\{42,1337,2026\}\) | Defaults match except seeds: CLI `--seed` default is `None`; `run_all.sh` never passes seeds. |
| “GNN / EGNN layers = 4” | `GraphEncoder` = Linear + **2** `EdgeConditionedConv` layers. E-GKN encoder = **2** EGNN layers; EGNN baseline = **3**. Tuned over \(\{2,4,6\}\) and “selected 4” is not what is instantiated. |
| “4-layer GCN” (Sec. 3.2) | Edge-conditioned MPNN, not GCN. |

---

## 2. Architecture default vs Table 6

### 2.1 What actually trains

`kge train --model koopman` instantiates **`GraphAwareKoopmanNet`** only (`cli.py` L256–261). Eval loads the same class (`cli.py` L420). `models/__init__.py` aliases `KoopmanNet = GraphAwareKoopmanNet`.

Kronecker generator (`koopman_net.py` L98–107):

```python
A_self_skew = self.A_self - self.A_self.T
A_edge_skew = self.A_edge - self.A_edge.T
P_sym = 0.5 * (self.P + self.P.T)
A_glob = kron(I_N, A_self_skew) + alpha * kron(P_sym, A_edge_skew)
return safe_matrix_exp(A_glob)
```

This is exactly revised Eqs. (1)–(2) with \(\Delta t=1\). \(P\) is row-normalized adjacency from `edge_index`. Transition acts on concatenated \(s\in\mathbb{R}^{Nd}\) via `h_flat @ K_glob.T`.

**Legacy `GraphKoopmanNet`** (L31–66) is a **shared node-wise** \(K=\exp(A_{\mathrm{raw}}-A_{\mathrm{raw}}^\top)\in\mathrm{SO}(64)\) applied independently per node. It is documented `OBSOLETE`, has **no `compute_loss`**, and is **not** in the CLI. Reviewer 1LAu W1 is **addressed for reported experiments**, provided nobody retrains via a notebook that still constructs the old class.

### 2.2 Parameter counts vs Table 6

Closed form for `GraphAwareKoopmanNet` with \(d=64\):

| Block | Count |
|-------|-------|
| Encoder (`GraphEncoder`) | 25,536 |
| Koopman tensors \(A_{\mathrm{self}},A_{\mathrm{edge}},\alpha\) | \(2\times 64^2+1=8{,}193\) (4,033 skew DoFs) |
| Decoder `Linear(64N→128) + Linear(128→128) + Linear(128→3N)` | \(8{,}579N + 16{,}640\) |
| **Total** | **\(50{,}369 + 8{,}579N\)** |

Aspirin \(N=21\): \(230{,}528\). Stachyose \(N=87\): \(796{,}742\). Matches Table 6 “Graph Koopman” and the rebuttal W1 arithmetic.

G-GRU extras vs KGE (same encoder/decoder): `msg_proj` \(64\times 64\) + `GRUCell(64,64)` + \(\alpha\) = \(+20{,}864\) → aspirin \(251{,}392\). Matches Table 6 “Graph GRU.”

**Table 6 is the Kronecker GraphAware story, not legacy 64×64.** A legacy-only operator would be \(230{,}528-8{,}193+4{,}096=226{,}431\), which appears nowhere in the tables.

Flat-K uses a dense \(Nd\times Nd\) generator (`A_raw` of size `latent_dim`); that *does* scale \(\sim n_{\mathrm{atoms}}^2\) and is consistent with the 2.86M (aspirin) / 34.9M (stachyose) column.

### 2.3 Paper text that still fights the code

- **Limitation 3** (p. 10): “the pooled graph-level latent vector discards per-node temporal variation.” The encoder docstring and implementation explicitly **do not pool** (`blocks.py` L47–49). Concatenation is not pooling. This is leftover copy from an older design and will look like a hallucination next to Eqs. (1)–(2).
- **Sec. 4.6:** “augmenting SE(3)-equivariant GNNs with **node-local** Koopman transitions.” E-GKN in code is the **same Kronecker \(K_{\mathrm{glob}}\in\mathrm{SO}(N(d-3))\)** plus an EGNN coordinate updater (`koopman_net.py` L208–229, L234–255). Appendix A is correct; 4.6’s “node-local” is not.
- **Duplicate E-GKN / EGNN / DummyDecoder** in `baselines.py` *and* `koopman_net.py`/`blocks.py`. CLI E-GKN is `koopman_net.EquivariantKoopmanNet`. Dead twins invite future drift.
- `EquivariantGraphEncoder.forward` **hardcodes** `z.view(B, T, V, 64)` (`blocks.py` L310) even if `hidden_dim≠64`.
- Obsolete class docstrings still claim GraphAware models “strictly enforce pairwise distances and **graph energy conservation** during the latent rollout” (`koopman_net.py` L35–36; `baselines.py` L35–36). They do not. \(L_{\mathrm{iso}}\) is a training penalty; there is no conservation law in the rollout.

---

## 3. Do \(L_{\mathrm{collapse}}\) / \(L_{\mathrm{iso}}\) still train the headline metrics?

### 3.1 What the headlines actually are

Main claim (abstract / Table 1): bond, angle, torsion, \(|R_{\mathrm{norm}}-1|\). Also sold: \(R_{\mathrm{edge}}\).

### 3.2 \(R_{\mathrm{norm}}\) — mostly *not* trained by \(L_{\mathrm{collapse}}\)

Eval \(R_{\mathrm{norm}}\) is mean-squared latent-norm ratio of a **rollout** (`physics_eval.py` `_compute_energy_ratio`: `mean(\|h\|^2)` at \(t\) over \(t=0\)). For KGE this is **Theorem 1**, \(\approx 1\) by construction of \(\exp(A_{\mathrm{skew}})\). Zeroing \(\lambda_{\mathrm{collapse}}\) cannot break that (Appendix D: \(R_{\mathrm{norm}}=1.0000\) on every noreg KGE seed).

\(L_{\mathrm{collapse}}\) is an **encoder anti-freeze hinge** on consecutive **teacher-forced encodings**. Related quantity, different tensor, different time. Reviewer t7BM’s phrase “directly regularize the latent-collapse ratio” is **too strong for \(R_{\mathrm{norm}}\)**, and only loosely true for Table 5’s pairwise latent-collapse score (which is a **rollout** pairwise-distance contraction, never appearing in `compute_loss`).

A separate trainer guard rejects checkpoints if encoded val relative change \(<10^{-4}\) (`trainer.py` L284–297). That *is* collapse-related, but it is model-selection, not the loss.

### 3.3 Bond / angle / torsion — **yes, partly trained**, and eval ≠ paper definition

**Training.** \(L_{\mathrm{iso}}\) is MSE on **bonded distances of autoencoded frames vs GT**. It directly supervises the same decoder used at eval. Appendix D already shows the numerical dependence: aspirin KGE bond \(0.0045\to 0.0149\) Å, angle \(0.0908\to 0.3114^\circ\) when the two terms are dropped.

**Evaluation.** `ThreeWayAblationEvaluator._compute_drifts` and `PhysicsEval` compute drift **from decoded \(t=0\) of the predicted trajectory**, not vs ground-truth topology:

```python
# physics_eval.py
b_drift = mean(|b_vals - b_vals[:, 0:1, :]|)
# print: "Physical drifts @ t=S (lower better; drift from decoded t=0)"
```

Sec. 4.1 writes: “Mean absolute error in inter-atomic bond lengths **relative to ground-truth molecular topology**.”

That mismatch is load-bearing. Appendix F already documents a **decode-collapse** on stachyose KGE (mean bonded length \(0.19\) Å vs GT \(1.28\) Å). Drift-from-own-\(t=0\) can look excellent while absolute structure is wrong. RDF absolute L1 favoring G-GRU (30/36) is the same phenomenon. **Headline topology wins are not the same functional as \(L_{\mathrm{iso}}\), but they share the decoder and are easy to over-read as “bonds vs chemistry.”**

Angle/torsion are **not** in the loss. Any angle/torsion noreg residual is the interesting SO(n) (plus encoder) story — and is exactly why Appendix D still matters.

### 3.4 Where noreg lives, and what is missing

| Asset | Coverage |
|-------|----------|
| `run_rebuttal_aspirin.sh` | aspirin × seeds \(\{42,1337,2026\}\) × `{koopman,gru,flat?}` × `{full,noreg}` |
| `run_rebuttal_expand.sh` | **noreg only**. Default Outcome B: malonaldehyde, benzene, ethanol, dha, at-at. Outcome A adds remaining MD17 + stachyose, ac-ala3-nhme. **No N-body.** |
| `run_rebuttal_aspirin_eval_only.sh` | eval from existing ckpts |
| Appendix D tables | aspirin full vs noreg + 5-molecule noreg expand — matches the B-path script, **not** Outcome A |
| Q1 `scripts/run_q1_pushforward.sh` | G-GRU \(H\in\{4,16\}\), \(\sigma\in\{0,0.01\}\) under **full** loss; KGE **not** retrained; no N-body; no symplectic |
| RDF scripts | **full-loss** KGE vs G-GRU, MD17+MD22, 3 seeds |

**Missing for a clean attribution story**

1. Noreg on **all 14** Table 1 systems (N-body charged/springs absent).  
2. Noreg for **E-GKN vs EGNN** and vs SEGNO.  
3. Factorial ablation: \(\lambda_{\mathrm{iso}}=0\) alone vs \(\lambda_{\mathrm{collapse}}=0\) alone (currently bundled).  
4. Noreg **and** GT-referenced bond/angle/torsion (not only \(t=0\)-self drift).  
5. Pushforward/noise under **noreg** (Appendix E still uses full loss, so it cannot isolate SO(n) from \(L_{\mathrm{iso}}\)).  
6. Public checkpoints + seed logs that hash to Tables 2, 7–8, 10–15. None are in this repo (`checkpoints/`, `eval_logs/` are runtime dirs, not committed).  
7. Wilcoxon Table 1 recomputed on the **noreg** 14-system (or explicitly shrunk) set.

---

## 4. Remaining paper↔code mismatches and reproducibility gaps

### 4.1 Numbers still colliding (revised PDF did not fully kill 1LAu W3)

Captions now say Table 2 = 3-seed aggregate, Tables 7–8 = “benchmark sweep.” Readers still see the same cells with different numbers, and the protocol is not internally consistent:

| Quantity | Table 2 (claimed 3-seed) | Table 7/8 |
|----------|--------------------------|-----------|
| aspirin KGE bond | \(0.0045\pm 0.0042\) | \(0.0070\) |
| aspirin G-GRU MSE | \(0.1388\pm 0.0307\) | \(0.1388\) (looks like the multi-seed mean copied into the “sweep” table) |
| AT-AT G-GRU angle | \(17.98^\circ\) | \(24.94^\circ\) |
| springs G-GRU MSE | \(0.0531\pm 0.0015\) | \(0.0531\) |

If 7–8 were a single seed, aspirin G-GRU MSE would not identically equal the 3-seed mean. The “we fixed captions” patch is still a credibility risk.

Sec. 4.2 cites malonaldehyde G-GRU bond \(0.059\) vs KGE \(0.071\) (Table 7 single-ish numbers) while Table 2’s 3-seed means are \(0.098\) vs \(0.090\) and **flip the exception**. Main text and tables disagree on which model wins that bond.

### 4.2 Soft physics / energy language (hypothesis: still the claim problem)

Abstract/Contribution 3 **dropped** “structural-energy.” That original AC issue is **text-fixed**. Residual surfaces:

- Remark 1 still analogizes Liouville / Hamiltonian volume. Caveat exists; abstract+conclusion still read as “geometry-preserving **physical** simulation.”
- Conclusion: tradeoff “favorable for **physical fidelity**.” Appendix F absolute RDF mostly **loses**. That sentence is not evidence-matched.
- Fig. 2 / eval code still plot **“Graph Energy Ratio”**, which is latent \(\mathbb{E}\|h\|^2\) ratio (`_graph_energy_retention`), not a Hamiltonian. Easy to re-inflate W6.
- README contribution 3: “Physical Regularization in 3D Space… latent orthogonality strongly regularizes spatial decoding.” Overclaim vs Limitation 1.
- `pyproject.toml` keywords include `physics-informed-machine-learning`.

### 4.3 Data protocol (undocumented in Appendix C)

| Knob | Code default | Paper |
|------|--------------|-------|
| MD17 window / subsample / train frac / bond cutoff | 150 / 2 / 0.8 / 1.6 Å | not in Table 9 |
| MD22 subsample | **10** | not stated |
| Alignment | SVD principal axes per window | not stated |
| Graph | distance cutoff on **first raw frame** | “fixed topology \(E\)” |
| Checkpoint split | `trainer.fit(train_split, test_split)` — **test is the validation \(R^2\)** | “validation \(R^2\)”; no third split |
| N-body | `NBodyAdapter.load()` returns **`(split_obj, split_obj)`** — train file used as train *and* “test”; 2D padded to 6D; **fully connected** \(E\) | presented as a physical system alongside MD; bond/angle/torsion on a complete graph is a different object |

Using the reported test windows to pick the best epoch is a silent leakage. N-body identity of train/test is worse.

### 4.4 Other gaps

- Appendix H still points at `anonymous.4open.science/r/Koopman-Graph-Evolver-4EB2`. Contribution 4 “we open-source…” vs forum with **no durable public link**. This GitHub repo + PyPI `koopman-graph-evolver==0.1.1` exist, but the PDF does not cite them.
- PDF author block still Anonymous — wrong for arXiv.
- `paper/main.tex` referenced by README/`pyproject.toml` **is not in this tree**.
- SEGNO import depends on a sibling `SEGNO/` checkout (`baselines.py` L906–913). Not vendored; not in this repo.
- Traffic/METR-LA still in CLI though the paper dropped it.
- No peak-memory table (authors already conceded).
- No seeds in `run_all.sh`; no table-reproduction entrypoint that prints Wilcoxon + Tables 2/7–8 from ckpts.
- E-GKN DummyDecoder: rollout “coordinates” for recon/iso during training are latent \(x\)-channels, while KGE uses an MLP decoder — isolation vs EGNN is fairer than vs G-GRU, but the shared-loss paragraph hides that.

---

## 5. Hypothesis verdict

**Confirmed, with one extra metric-definition finding.**

- **SO(n) operator:** implemented, default, Table-6-consistent, Kronecker, \(O(N)\). Legacy 64×64 is dead code. This is **not** the remaining reject driver.
- **Loss-attribution:** still live. \(L_{\mathrm{iso}}\) trains GT bonds on reconstructions; eval bond drift is **self-drift from decoded \(t=0\)**; noreg shrinks margins; Table 1 was not rebuilt under noreg; Q1 stabilizers still use the full 4-term loss.
- **Soft-physics language:** energy claim removed from abstract, but Liouville remark, “physical fidelity” conclusion, “graph energy” plots, and “energy conservation” docstrings remain.
- **New (this audit):** Sec. 4.1 GT-referenced topology vs code \(t=0\)-referenced drift; G-GRU 4-step \(\mathcal{L}_{\mathrm{dyn}}\) vs “strict parity”; Table 9 cosine/4-layer/Adam vs AdamW/constant LR/2-layer MPNN; N-body train=test.

---

## 6. Concrete file-level fix lists

### Track A — venue salvage (code + experiments + rewrite)

Do not resubmit the current PDF. Minimum experiment bar is **attribution on the same protocol as Table 1**, plus honest metrics.

**Code (make one objective, one architecture, one metric suite)**

1. `koopman_evolver/models/koopman_net.py` — delete or hard-gate `GraphKoopmanNet`; strip “energy conservation” docstring.  
2. `koopman_evolver/models/baselines.py` — delete duplicate `EquivariantKoopmanNet` / `EGNN_Layer` / `DummyDecoder` (keep one module). Force Flat-K and SEGNO through the **same** four-term API or **document** the exceptions in Eq. (3).  
3. `koopman_evolver/models/blocks.py` — remove dead `EGNNDynamicsNet` hardcoded loss; fix `view(..., 64)`.  
4. `koopman_evolver/models/baselines.py` `GraphAwareGRUNet` — add `--unroll-steps 1` as the **parity** setting; keep \(H=4\) only as an Appendix E arm, not as “vanilla = same loss.”  
5. `koopman_evolver/evaluation/physics_eval.py` — implement **both** (a) drift vs **GT** topology and (b) drift vs predicted \(t=0\); rename every “energy ratio” print/plot to \(R_{\mathrm{norm}}\) / \(R_{\mathrm{edge}}\). Stop using “graph energy.”  
6. `koopman_evolver/training/trainer.py` — real **val** split (e.g. last 10% of train windows); never checkpoint on test. Constant vs cosine: pick one and log it.  
7. `koopman_evolver/cli.py` — default batch 32; require `--seed`; drop traffic or mark experimental. Optimizer: either implement cosine Adam as Table 9 or change Table 9.  
8. `koopman_evolver/data/nbody_adapter.py` — load distinct train/test NRI files; do not return the same object twice.  
9. `koopman_evolver/data/md17_adapter.py` + `md22_adapter.py` — log/serialize `window_len`, `sub_sampling`, `bond_cutoff`, `edge_index` hash into checkpoints.  
10. Scripts: one `scripts/reproduce_tables.sh` that trains/evals all 14 × 3 seeds × `{full,noreg}` and writes CSV for Wilcoxon. Extend `run_rebuttal_expand.sh` to N-body + E-GKN/EGNN noreg. Add iso-only / collapse-only tags. Re-run Q1 under **noreg**.  
11. Vendor or submodule SEGNO; pin commits.  
12. `README.md` — deanonymize, match the rewritten claims, link tables scripts, state LoG 2026 reject.

**Experiments (mandatory vs optional)**

- **Mandatory:** noreg KGE vs G-GRU on all 14 systems × 3 seeds; report GT-referenced *and* \(t=0\)-referenced topology; recompute Wilcoxon on noreg; factorial iso vs collapse on aspirin + one MD22; publish ckpts.  
- **If claiming E-GKN:** noreg E-GKN vs EGNN on the Table 3 systems.  
- **Optional for venue, not honesty:** symplectic baseline, NequIP/MACE, peak memory, spectra.

**Manuscript rewrite**

- Lead with **temporal inductive bias + MSE-paradox diagnostic**, not Liouville.  
- Eq. (1) = actual `compute_loss` tensors (autoencode recon; per-node collapse hinge; G-GRU \(H\)).  
- Put noreg in **main** Table 1 / Table 2. Shrink 14-system claims to the noreg set if compute is incomplete.  
- Own higher MSE and RDF absolute losses in abstract/conclusion.  
- Delete Limitation 3 pooling sentence or replace with concatenation/decoder \(O(N)\) bottleneck.  
- Unify Table 2 vs 7–8 (one protocol, or move 7–8 to a clearly labeled single-seed supplement with a seed column).  
- Fix malonaldehyde bond-exception paragraph to the table that is actually cited.  
- Deanonymize; cite this GitHub SHA; drop anonymous.4open.

### Track B — arXiv-only honesty (no big new runs)

Goal: claims ⊆ evidence already in the PDF + this repo. Do **not** pretend Appendix D closed attribution.

**Paper (priority order)**

1. Deanonymize; replace Appendix H with this repo + commit SHA; mention LoG 2026 reject / revision.  
2. Abstract/conclusion: no “physical fidelity” as a general win; keep MSE paradox + topology **and** RDF humility.  
3. Rewrite Eq. (3) bullets to match code (recon = autoencode; collapse = encoder hinge; iso = recon bonds; \(\Delta t=1\); G-GRU \(H=4\)). Point to Appendix D in the **main** loss paragraph, not only at the end.  
4. Sec. 4.1: say bond/angle/torsion are **decoded-trajectory self-drift from \(t=0\)** if that is what Tables 2/7–8 are (or recompute — that would be Track A).  
5. Table 9: AdamW, no cosine, 2 MPNN layers, batch-size caveat, WD \(10^{-4}\), `unroll_steps=4` for G-GRU.  
6. Strike Limitation 3 pooling; strike “node-local” in 4.6; keep Theorem 1 latent-only.  
7. Remark 1: one sentence, or cut. Do not analogize Liouville in the abstract.  
8. Add a short “open attribution” box: noreg is MD17/MD22 subset, no N-body, no E-GKN; iso changes margins; \(R_{\mathrm{norm}}=1\) is architectural.  
9. Tables 2 vs 7–8: one legend block in **both** captions + a footnote in 4.4 that they are **not** interchangeable; fix the malonaldehyde exception to Table 2 or Table 7 explicitly.  
10. Contribution 4: “code at …” not “we open-source” with a dead anonymous URL.

**Code (minimal, so readers who clone `dev/rebuttal-1ALu` do not re-discover W1/W2)**

1. `docs/` — this audit (done).  
2. `README.md` — default architecture = GraphAware Kronecker; paste Eq. (3) weights; document `--lambda-collapse 0 --lambda-iso 0`; document G-GRU \(H=4\); document drift-from-\(t=0\); link rebuttal scripts.  
3. `koopman_evolver/utils/loss_weights.py` — already honest; reference it from README.  
4. Rename eval prints “LATENT ENERGY RATIO” → \(R_{\mathrm{norm}}\) (already partly done in 3-way summary); “GRAPH ENERGY” → latent \(\|h\|^2\) ratio.  
5. Comment-gate or delete `GraphKoopmanNet` / `GraphGRUNet` / `blocks.EGNNDynamicsNet` so `grep KoopmanNet` cannot revive W1.  
6. One `LOSS.md` (short) quoting `compute_loss` and listing parity exceptions (Flat iso, SEGNO collapse, G-GRU unroll).  
7. Do **not** silently change defaults (batch, \(H\), layers) without relabeling tables — that would create a *new* mismatch with published numbers.

**Explicitly out of Track B:** full 14-system noreg, NequIP, symplectic, cosine-schedule retrain, 4-layer GNN retrain.

---

## 7. Suggested decision

| Product | Honest next step |
|---------|------------------|
| arXiv | Track B, then post. Current revised PDF is **still not** post-as-is. |
| Venue | Track A. Same PDF at another GNN venue will hit the same AC failure mode (objective + energy/physics overclaim + numbers). |
| Split | Diagnostic note (MSE paradox + springs/MD topology suite) can ship faster than the method paper. |

The interesting scientific residue after this audit is: **under noreg, \(R_{\mathrm{norm}}=1\) is free from SO(n), and angle drift often still favors KGE**, while bonds are partly \(L_{\mathrm{iso}}\) and absolute RDFs often favor G-GRU. That is a publishable, narrower claim. It is not what the abstract currently sells.
