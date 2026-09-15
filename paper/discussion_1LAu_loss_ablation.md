# Discussion Reply to Reviewer 1LAu — Loss Ablation (L_collapse, L_iso)

**Scope:** MD17/MD22 only (no N-body). **Seeds:** {42, 1337, 2026}. **Horizon:** 29 steps. **Models:** KGE vs G-GRU. Values are mean ± std. Bold = lower drift, or closer to 1 for R_norm / R_edge.

We thank the reviewer for pressing on causal attribution. We disable L_collapse / L_iso while keeping L_recon and L_dyn fixed (weights are the trainer defaults; no annealed λ):

```
L_full  = 10*L_recon + 1*L_dyn + 2*L_collapse + 5*L_iso
L_noreg = 10*L_recon + 1*L_dyn   (lambda_collapse = lambda_iso = 0)
```

L_recon is a teacher-forced autoencoder (encode current → decode → same-timestep coords), not next-frame prediction. L_collapse is an encoder anti-freeze hinge and does not train R_norm (R_norm=1 is architectural SO(n)). L_iso is bonded-distance MSE and does push bond margins. Bond/angle/torsion columns below are drift from decoded t=0, not vs GT.

Paired full vs noreg on MD17 aspirin; noreg expansion on malonaldehyde, benzene, ethanol, dha, at-at. Same protocol as the paper appendix.

**Takeaway.** On aspirin, disabling the two terms increases KGE bond/angle/torsion drift and moves R_edge farther from 1, so they affect margins. Under noreg, R_norm = 1.0000 for KGE on every seed; seed-averaged angle drift (and on most systems bond/torsion and |R_edge−1|) stay lower for KGE than G-GRU. MSE is lower for G-GRU in nearly all runs.

## Table 1 — Aspirin full vs noreg

| Setting | Model | MSE | Bond (Å) | Angle (°) | Torsion (°) | R_norm | R_edge |
|---|---|---:|---:|---:|---:|---:|---:|
| full | KGE | 0.241±0.003 | **0.0045±0.005** | **0.091±0.024** | **0.148±0.070** | **1.000** | **0.997±0.005** |
| full | G-GRU | **0.139±0.038** | 0.069±0.014 | 5.50±1.40 | 6.42±0.77 | 0.821±0.030 | 0.959±0.019 |
| noreg | KGE | 0.222±0.008 | **0.015±0.013** | **0.311±0.101** | **0.731±0.336** | **1.000** | **0.986±0.013** |
| noreg | G-GRU | **0.062±0.012** | 0.026±0.004 | 1.43±0.32 | 1.54±0.45 | 0.941±0.018 | 0.983±0.005 |

Aspirin noreg (of 3): lower bond for KGE on 2/3; angle 3/3; torsion 2/3; |R_norm−1| 3/3; |R_edge−1| 2/3. Seed 42: G-GRU lower bond/torsion and |R_edge−1|.

## Table 2 — Expand set (noreg only)

| System | Model | MSE | Bond | Angle | Torsion | R_norm | R_edge |
|---|---|---:|---:|---:|---:|---:|---:|
| malonaldehyde | KGE | 0.371±0.018 | **0.015±0.005** | **1.34±0.68** | **2.79±1.41** | **1.000** | **0.980±0.010** |
| malonaldehyde | G-GRU | **0.362±0.004** | 0.085±0.009 | 3.44±0.16 | 4.80±1.23 | 0.886±0.029 | 0.940±0.007 |
| benzene | KGE | 0.425±0.013 | 0.016±0.013 | **0.590±0.143** | 0.899±0.072 | **1.000** | **0.992±0.012** |
| benzene | G-GRU | **0.030±0.001** | **0.016±0.005** | 0.705±0.265 | **0.526±0.100** | 0.973±0.016 | 0.989±0.004 |
| ethanol | KGE | 0.236±0.002 | **0.012±0.006** | **0.624±0.097** | **2.55±1.68** | **1.000** | **0.987±0.011** |
| ethanol | G-GRU | **0.167±0.002** | 0.084±0.023 | 3.14±1.55 | 4.36±2.22 | 0.885±0.034 | 0.931±0.023 |
| dha | KGE | 3.11±0.54 | **0.040±0.024** | **1.57±0.31** | **2.99±1.10** | **1.000** | **0.962±0.029** |
| dha | G-GRU | **1.15±0.07** | 0.112±0.031 | 9.22±2.61 | 16.0±4.2 | 0.867±0.045 | 0.927±0.029 |
| at-at | KGE | 6.29±1.00 | **0.048±0.028** | **0.804±0.441** | **2.18±1.57** | **1.000** | **0.966±0.020** |
| at-at | G-GRU | **2.33±0.18** | 0.266±0.096 | 16.3±5.4 | 24.6±9.7 | 0.772±0.043 | 0.882±0.053 |

Expand (15 seeds): bond 13/15, angle 14/15, torsion 11/15, |R_norm−1| 15/15, |R_edge−1| 13/15 for KGE; MSE 1/15. With aspirin (18): 15/18, 17/18, 13/18, 18/18, 15/18. Benzene: bonds nearly tied; torsion lower for G-GRU.

## Conclusion

1. Removing L_collapse / L_iso changes aspirin numbers (KGE bond 0.0045→0.015 Å; angle 0.091→0.311°): margins are partly loss-dependent.
2. Under noreg, R_norm=1 for KGE on all seeds; angle drift lower for KGE on every system above; bond/torsion/|R_edge−1| lower for KGE on most pairs (exceptions: benzene, aspirin seed 42).
3. Seed-averaged MSE lower for G-GRU on every system here.
4. N-body / EGNN–E-GKN noreg not run; happy to add if useful. Tables also in the revised appendix.
