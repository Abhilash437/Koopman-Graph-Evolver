# Koopman Graph Evolver: Technical Documentation

## 1. Mathematical Foundation

### 1.1 Stochastic Block Model (SBM)
The SBM is a generative model for random graphs that exhibits community structure. Nodes are partitioned into $k$ blocks.
The probability of an edge between node $u \in B_i$ and $v \in B_j$ is given by a matrix $P_{ij}$.

In this implementation:
- $P_{ii} = p_{intra}$ (High density within community)
- $P_{ij} = p_{inter}$ for $i \neq j$ (Low density between communities)

### 1.2 Graph Evolution Dynamics
The graph evolves discretely as $G_t \rightarrow G_{t+1}$ using two independent Bernoulli processes:
1. **Edge Removal**: For $e \in G_t$, $P(e \notin G_{t+1}) = p_{remove}$
2. **Edge Addition**: For $e \notin G_t$, $P(e \in G_{t+1}) = p_{add}$

The stationary edge density $\rho$ is governed by:
$$\rho_{\infty} = \frac{p_{add}}{p_{add} + p_{remove}}$$

### 1.3 Koopman Operator Theory
The Koopman operator $\mathcal{K}$ is an infinite-dimensional linear operator that acts on functions of the state space (observables) $\psi(x)$, such that:
$$\mathcal{K}\psi(x_t) = \psi(x_{t+1})$$

In our latent space, we approximate this with a finite-dimensional matrix $K \in \mathbb{R}^{d \times d}$:
$$z_{t+1} = K z_t$$
Where $z = \text{Encoder}(G)$.

---

## 2. Model Architectures

### 2.1 GCN Encoder
The encoder $\phi: \mathcal{G} \rightarrow \mathbb{R}^d$ uses Graph Convolutional Networks (GCN). The layer-wise update rule is:
$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$$
where $\tilde{A} = A + I$ is the adjacency matrix with self-loops, and $\tilde{D}$ is the degree matrix.

### 2.2 Transition Models
- **MLP Transition**: A non-linear mapping $z_{t+1} = \text{MLP}(z_t)$.
- **Koopman Transition**: A linear mapping $z_{t+1} = z_t K^\top$.
    - **Spectral Penalty**: To ensure stability, we penalize eigenvalues $\lambda$ of $K$ where $|\lambda| > 1$:
    $$\mathcal{L}_{spectral} = \text{mean}(\max(0, |\lambda(K)| - 1))$$

---

## 3. Training & Loss Functions

### 3.1 Adjacency Reconstruction
Since graph adjacency matrices are often sparse, we use **BCE with Logits Loss** combined with a `pos_weight` to balance the loss contribution of edges vs. non-edges.
$$\mathcal{L}_{recon} = -\frac{1}{E} \sum_{i=1}^E [w \cdot y_i \log \sigma(\hat{y}_i) + (1-y_i) \log(1-\sigma(\hat{y}_i))]$$
where $w = \frac{\text{count(zeros)}}{\text{count(ones)}}$.

---

## 4. Key Terminology

### 4.1 Ablation Study
An **Ablation Study** involves removing or modifying specific components of a system to understand their individual contribution to the overall performance.
In this project, we perform an ablation on the **Latent Dimension** ($d \in \{16, 32, 64\}$) to justify our architectural choices.

### 4.2 Multi-Step Rollout
Instead of predicting just one step ahead ($t \rightarrow t+1$), a **Rollout** involves recursively applying the transition model:
$$\hat{z}_{t+k} = T(T(...T(z_t)...))$$
This tests the **accumulation of error** over time. Linear Koopman operators often show better stability in long-term rollouts.

### 4.3 Paired T-Test (Statistical Validation)
To prove that Model A is significantly better than Model B, we compare their performance across multiple random seeds using a **Paired T-Test**.
- **Null Hypothesis ($H_0$)**: The mean difference in performance is zero.
- **P-value**: If $p < 0.05$, we reject $H_0$ and conclude the improvement is statistically significant.

---

## 5. Implementation Details

### Node Features
In the absence of raw node data, we use **Node Degree** as an initial feature vector $x_i \in \mathbb{R}^1$. This provides the GNN with basic structural information about each node's local connectivity.

### Symmetry in Decoder
The decoder predicts $N(N-1)/2$ edges (upper triangle). We then mirror these to the lower triangle to ensure the reconstructed adjacency matrix is perfectly symmetric:
$$A_{ij} = A_{ji}$$

---

## 6. Empirical Results (Prototype Study)

### 6.1 Performance Comparison
The prototype study compared a non-linear MLP transition model against a linear Koopman operator across 15 rollout steps.

| Metric | MLP Transition | Koopman Transition |
| :--- | :---: | :---: |
| **Parameters** | 2,128 | **256** (8.3x fewer) |
| **Step-1 Accuracy** | 91.3% | 91.3% |
| **Step-15 Accuracy** | 55.4% | **64.9%** |
| **Accuracy Drop (1→15)** | 35.8% | **26.4%** |
| **Mean Advantage (Steps 3-15)** | - | **+10.0%** |

### 6.2 Stability Analysis
The learned Koopman operator $K$ was analyzed for spectral stability. A discrete-time linear system $z_{t+1} = K z_t$ is stable if all eigenvalues $\lambda$ of $K$ satisfy $|\lambda| \leq 1$.
- **Observed Max $|\lambda|$**: 0.7260
- **Mean $|\lambda|$**: 0.3240
- **Conclusion**: The learned operator is strictly stable, explaining its superior performance in long-term rollouts compared to the MLP, which lacks such explicit stability guarantees.

### 6.3 Key Findings
1. **Efficiency**: The Koopman model achieves superior rollout stability with an order of magnitude fewer parameters.
2. **Rollout Robustness**: While both models perform identically on single-step prediction (91.3% accuracy), the Koopman model's accuracy degrades significantly slower, maintaining a ~10% lead by step 15.
3. **Identity Convergence**: Interestingly, F1 scores for both models converge to similar values (approx 0.33) at step 15, suggesting that while the Koopman model preserves global structure better, fine-grained edge identity remains challenging for both architectures in long-term predictions.

---

## 7. Future Work
1. **Higher-Dimensional Latent Spaces**: Expanding `latent_dim` from 16 to 64.
2. **Positional Encodings**: Adding Laplacian Eigenmaps or Node2Vec features to provide richer initial node embeddings.
3. **Real-World Datasets**: Testing on the ZINC molecular dataset or protein-protein interaction (PPI) networks.
4. **Advanced Metrics**: Moving beyond edge-level accuracy to graph-theoretic metrics like degree distribution stability and clustering coefficient preservation.

