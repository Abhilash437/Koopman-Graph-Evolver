# Plan: Q3 RDF pilot → then Q1 feasibility

## Goal
Answer the rating-4 reviewer with evidence, not vague “future work.”
Order: **Q3 first** (eval-only, uses existing ckpts) → then decide **Q1** (needs retrain).

---

## Phase A — Q3 RDF pilot (try now)

**Why first:** No new training. Need decoded coordinate rollouts + ground-truth frames → histogram of pairwise distances.

**Inputs (on GCP):**
- Prefer rebuttal aspirin ckpts: `checkpoints/rebuttal_1LAu/graph_aware_{koopman,gru}_aspirin_seed{42,1337,2026}_{full,noreg}_best.pt`
- Fallback local: `checkpoints/phase1/graph_aware_{koopman,gru}_ethanol_best.pt` (smoke only)

**Protocol (minimal):**
1. Load KGE + G-GRU, run 29-step rollout on held-out trajectories (same split as eval).
2. For each time step (or final step + average over horizon), compute all-pair (or bonded+nonbonded) distances for pred vs GT.
3. RDF-style histogram: `g(r)` with fixed bins (e.g. 0–6 Å, 0.05 Å).
4. Report: overlay plot + simple error (L1/L2 between pred and GT `g(r)` at t=0 vs t=29).

**Success criteria for discussion:**
- If KGE `g(r)` stays closer to GT than G-GRU at long horizon on aspirin → add 2–3 sentences + one figure to discussion.
- If mixed/null → report honestly; do not force a win.

**Effort:** ~0.5–1 day (script + GCP eval). No RDF code in repo today—add a small `scripts/rdf_from_ckpt.py` (or notebook) that reuses `PhysicsEval` rollout/decode path.

**Not in scope for pilot:** spectra / VACF (needs velocities or finite-diff design); full 14 systems.

---

## Phase B — Q1 only if Phase A is clean

**Question:** Do pushforward and/or training noise on G-GRU close the structural gap vs KGE?

**Why harder:** Existing ckpts were **not** trained with long pushforward or noise. Must retrain G-GRU only (keep KGE full fixed). **Skip symplectic** (different modeling interface).

**Minimal fair test:**
1. Paper G-GRU already has `unroll_steps=4`, noise=0. Raise pushforward and add latent training noise under the same full loss.
2. Primary arm (**combined**): `UNROLL=16` + `TRAIN_NOISE_STD=0.01`.
3. Optional **matrix** for attribution: pf-only (16/0), noise-only (4/0.01), combined (16/0.01).
4. Train **all paper MD17+MD22** (no N-body), seeds `{42}` matrix smoke then expand winning arm to `{42,1337,2026}`.
5. Compare to existing KGE full on bond/angle/torsion/`R_norm`/`R_edge`.
6. Interpret:
   - If G-GRU+stabilizers ≈ KGE on structure → SO(n) is not uniquely necessary here.
   - If gap remains → SO(n) still adds something under this protocol.

**Script (order):**  
1. Smoke matrix all mols: `DEVICE=cuda SEEDS=42 ./scripts/run_q1_pushforward.sh` (~12×3 GRU trains)  
2. Expand seeds on winning arm only, e.g.  
   `DEVICE=cuda SEEDS="42 1337 2026" MODE=combined UNROLL=16 TRAIN_NOISE_STD=0.01 ./scripts/run_q1_pushforward.sh`  
   Single-mol override: `MOL=aspirin ...`


**Effort:** ~1–3 days GCP after hook exists (done: `--unroll-steps`, `--train-noise-std`).


---

## Decision gate

| After Q3 pilot | Action |
|---|---|
| Clear structural story in `g(r)` | Post RDF snippet in discussion; optionally start Q1 smoke |
| Ambiguous RDF | Still post honest Q3 wording; **do not** rush Q1 before deadline |
| Blocked (no GCP ckpts / data) | Keep current Q3 text (“computable, running pilot”); restore ckpts from VM first |

---

## Immediate next commands (GCP)

```bash
cd ~/github-main/Koopman-Graph-Evolver
git pull origin dev/rebuttal-1ALu   # after this script is pushed

ls checkpoints/rebuttal_1LAu/graph_aware_*aspirin*seed42*full* | head

python3 scripts/rdf_from_ckpt.py \
  --md17 aspirin \
  --koopman-ckpt checkpoints/rebuttal_1LAu/graph_aware_koopman_aspirin_seed42_full_best.pt \
  --gru-ckpt checkpoints/rebuttal_1LAu/graph_aware_gru_aspirin_seed42_full_best.pt \
  --device cuda \
  --out-dir results/rdf_pilot/aspirin_full_seed42

# optional: noreg + other seeds
for SEED in 42 1337 2026; do
  for TAG in full noreg; do
    python3 scripts/rdf_from_ckpt.py --md17 aspirin \
      --koopman-ckpt checkpoints/rebuttal_1LAu/graph_aware_koopman_aspirin_seed${SEED}_${TAG}_best.pt \
      --gru-ckpt checkpoints/rebuttal_1LAu/graph_aware_gru_aspirin_seed${SEED}_${TAG}_best.pt \
      --device cuda --out-dir results/rdf_pilot/aspirin_${TAG}_seed${SEED}
  done
done
```

**Readout:** compare `L1_KGE_vs_GT_t29` vs `L1_GRU_vs_GT_t29` (and deltas from t=0). If KGE stays closer to GT `g(r)` at long horizon, we have a downstream structural win to cite.
