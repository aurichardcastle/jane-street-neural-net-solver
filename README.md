# Reassembling a Shuffled Neural Network

Solving [Jane Street's neural network puzzle](https://huggingface.co/janestreet/droppedaneuralnet): recover the exact assembly order of a 48-block residual network shattered into 97 individual linear layers.

**Search space:** (48!)^2 ~ 10^122 possible assemblies. **Result:** Full recovery to floating-point-zero error (MSE ~ 7.8 x 10^-15).

Solved with [Keegan Ratner](https://github.com/keeganratner).

## The Problem

Jane Street took a trained 48-block residual network, ripped out every layer, shuffled them, and posted the raw weight matrices + 10,000 rows of historical predictions on Hugging Face. The 97 pieces split by shape into:
- 48 input projections (48 -> 96)
- 48 output projections (96 -> 48)
- 1 prediction head (48 -> 1)

Each residual block: `x -> x + W_out * ReLU(W_in * x + b_in) + b_out`

Two subproblems: **(1)** which input pairs with which output, and **(2)** what order do the 48 assembled blocks go in.

## Subproblem 1: Pairing (The Hard Part)

Naive approaches failed -- norm-based correlation scored 0.29, data-driven greedy matching scored 0.69 but degraded catastrophically in later steps.

**The breakthrough:** For each candidate pair (i, j), compute the matrix product M = W_out x W_in. Training via dynamic isometry leaves a structural fingerprint -- correctly paired layers produce M with a strongly negative diagonal:

```
score(i,j) = |trace(W_out * W_in)| / ||W_out * W_in||_F
```

On the actual puzzle files: correct pairs scored 1.76-3.23, incorrect pairs scored 0.00-0.58. A separation gap of 1.18. The Hungarian algorithm trivially solves the assignment.

**Why it works:** For a residual block with Jacobian J_f = I + W_out * D(x) * W_in, dynamic isometry requires E[J_f^T * J_f] ~ I, which forces trace(W_out * W_in) < 0. Incorrect pairings have no such constraint -- their products behave like random matrices.

## Subproblem 2: Ordering (Easier Than It Looks)

Residual blocks are near-identity transforms: `x -> x + r(x)` where r(x) is small relative to x. This makes adjacent blocks approximately commutative -- swapping two neighbors changes output by O(||r_A|| * ||r_B||). A rough ordering is already close to correct.

**Seed:** Sort blocks by ascending ||W_out||_F (deeper blocks develop larger weight norms during training). This gives MSE = 0.076 -- already in the right neighborhood.

**Then:** Adjacent-swap hill climbing:

```
Round 1: MSE 0.076 -> 0.006 (30 swaps)
Round 2: 0.006 -> 0.003 (17 swaps)
Round 3: 0.003 -> 0.001 (10 swaps)
Round 4: 0.001 -> 0.0005 (6 swaps)
Round 5: 0.0005 -> 0.0002 (2 swaps)
Round 6: 0.0002 -> 7.8e-15 (1 swap)
```

66 total swaps. Floating-point zero. Puzzle solved.

## Multi-Agent Workflow

We used Claude Code, Codex, and Anti Gravity as a three-stage pipeline off a shared scratchpad:

- **Claude Code** explored the data, tried and failed with naive approaches, established that pairing was the bottleneck
- **Codex** proposed the two-stage decomposition, worked through the theory, validated the pairing signal on the actual files
- **We** orchestrated -- directed focus, killed dead ends, wrote the final solver spec

The shared scratchpad was critical. Each agent could see what others had tried, what failed, and what was verified.

## Key Insight

The same properties that make neural networks trainable -- dynamic isometry, residual structure, near-commutativity -- leave exploitable signatures in the learned weights. The math that enables gradient flow also enables forensic reconstruction.

A 10^122 search space collapsed to 2,304 pairing scores + 66 swaps.

## Files

- `solve_ordering.py` -- Full solver pipeline: pairing verification, Frobenius norm seeding, adjacent-swap + insertion + pair-swap hill climbing
- `solve_final.py` -- Streamlined final solver using verified pair map
- `solution.txt` -- The recovered 97-element permutation

## Running

```bash
# Download the puzzle pieces from HuggingFace first:
# https://huggingface.co/janestreet/droppedaneuralnet

pip install torch numpy pandas
python solve_final.py
```

Requires the `pieces/` directory and `historical_data.csv` from the HuggingFace repo.
