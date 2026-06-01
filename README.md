# Reassembling a Shuffled Neural Network

### Jane Street's "Dropped a Neural Net" Challenge

> A 10^122 search space collapsed to 2,304 pairing scores + 66 swaps. Full recovery to floating-point-zero error.

[Jane Street](https://www.janestreet.com/) posted a puzzle on [Hugging Face](https://huggingface.co/janestreet/droppedaneuralnet): a trained 48-block residual network shattered into 97 individual linear layers, with only raw weight matrices and 10,000 rows of historical predictions. Recover the exact assembly order.

Solved by [Auric Hardcastle](https://github.com/AuricHardcastle) and [Keegan Ratner](https://github.com/keeganratner).

---

## The Problem

The 97 weight matrices split by shape into three groups:

| Shape | Count | Role |
|-------|-------|------|
| (96, 48) | 48 | Input projections |
| (48, 96) | 48 | Output projections |
| (1, 48) | 1 | Prediction head |

Each residual block combines one input and one output projection:

```
x -> x + W_out * ReLU(W_in * x + b_in) + b_out
```

The search space factors into two subproblems:
1. **Pairing:** which input layer goes with which output layer (48! possibilities)
2. **Ordering:** what sequence do the 48 assembled blocks go in (48! possibilities)

Combined: **(48!)^2 = 10^122 configurations.** More than atoms in the observable universe.

---

## Solution

### Subproblem 1: Pairing (The Negative-Diagonal Fingerprint)

Naive approaches failed. Norm-based correlation scored 0.29. Data-driven greedy matching scored 0.69 but degraded catastrophically in later steps because mispaired blocks corrupted the residual stream.

**The breakthrough:** For each candidate pair (i, j), compute the matrix product M = W_out x W_in. Training via dynamic isometry leaves a structural fingerprint. Correctly paired layers produce M with a strongly negative diagonal:

```
score(i, j) = |trace(W_out * W_in)| / ||W_out * W_in||_F
```

| | Score Range |
|---|---|
| Correct pairs | 1.76 - 3.23 |
| Incorrect pairs | 0.00 - 0.58 |
| **Separation gap** | **1.18** |

The Hungarian algorithm trivially solves the assignment from these scores.

**Why it works:** For a residual block with Jacobian J_f = I + W_out * D(x) * W_in, dynamic isometry requires E[J_f^T * J_f] = I, which forces trace(W_out * W_in) < 0. Incorrect pairings have no such constraint. Their products behave like random matrices.

### Subproblem 2: Ordering (Commutativity-Seeded Hill Climbing)

Residual blocks are near-identity transforms: `x -> x + r(x)` where r(x) is small relative to x. This makes adjacent blocks approximately commutative. Swapping two neighbors changes output by O(||r_A|| * ||r_B||). A rough ordering is already close to correct.

**Seed:** Sort blocks by ascending ||W_out||_F (deeper blocks develop larger weight norms during training).

**Then:** Adjacent-swap hill climbing. Scan all neighboring pairs, swap if MSE decreases, repeat:

```
Round 1:  0.076    -> 0.006    (30 swaps)
Round 2:  0.006    -> 0.003    (17 swaps)
Round 3:  0.003    -> 0.001    (10 swaps)
Round 4:  0.001    -> 0.0005   (6 swaps)
Round 5:  0.0005   -> 0.0002   (2 swaps)
Round 6:  0.0002   -> 7.8e-15  (1 swap)
```

**66 total swaps. Floating-point zero. Puzzle solved.**

---

## Multi-Agent Workflow

This wasn't one model reasoning end-to-end. Three agents worked off a shared scratchpad:

- **Claude Code** explored the data, tried and failed with naive approaches, established that pairing was the real bottleneck
- **Codex** proposed the two-stage decomposition, worked through the dynamic isometry theory, validated the pairing signal on the actual puzzle files
- **Us:** orchestrated the pipeline, directed focus, killed dead ends, wrote the final solver specification

The shared scratchpad eliminated duplicated work and context loss between handoffs.

---

## Key Insight

The same properties that make neural networks trainable (dynamic isometry, residual structure, near-commutativity) leave exploitable signatures in the learned weights. The math that enables gradient flow also enables forensic reconstruction.

---

## Repository Structure

```
├── solve_final.py          # Streamlined solver using verified pair map
├── solve_ordering.py       # Full pipeline: pairing -> seeding -> hill climbing
├── solution.txt            # Recovered 97-element permutation
├── fig_before_after.png    # Prediction accuracy before/after solving
└── fig_search_space.png    # Search space visualization
```

## Running

```bash
# 1. Download puzzle data from HuggingFace:
#    https://huggingface.co/janestreet/droppedaneuralnet
#    Place pieces/ directory and historical_data.csv in this directory.

# 2. Install dependencies
pip install torch numpy pandas

# 3. Run the solver
python solve_final.py
```

## License

MIT

## Authors

**Auric Hardcastle** · [LinkedIn](https://linkedin.com/in/auric-hardcastle) · [GitHub](https://github.com/AuricHardcastle)

**Keegan Ratner** · [GitHub](https://github.com/keeganratner)
