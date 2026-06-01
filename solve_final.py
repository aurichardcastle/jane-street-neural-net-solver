"""
Final solver using Codex's verified pair map + W_out Frobenius norm seed + adjacent-swap hill climbing.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

torch.set_grad_enabled(False)

# Verified pair map from Codex's paper analysis
PAIR_MAP = [
    (0, 75), (1, 40), (2, 51), (3, 53), (4, 17), (5, 21), (10, 20), (13, 7),
    (14, 33), (15, 67), (16, 83), (18, 25), (23, 46), (27, 76), (28, 12), (31, 36),
    (35, 24), (37, 6), (39, 38), (41, 92), (42, 55), (43, 34), (44, 82), (45, 19),
    (48, 9), (49, 72), (50, 66), (56, 30), (58, 78), (59, 52), (60, 93), (61, 80),
    (62, 79), (64, 70), (65, 22), (68, 47), (69, 89), (73, 11), (74, 90), (77, 32),
    (81, 8), (84, 63), (86, 57), (87, 71), (88, 54), (91, 29), (94, 96), (95, 26),
]
LAST_PIECE = 85

# Load data
print("Loading data...")
data = pd.read_csv('historical_data.csv')
X_all = torch.tensor(data.iloc[:, :48].values, dtype=torch.float32)
pred_all = torch.tensor(data['pred'].values, dtype=torch.float32)

# Load pieces and build blocks
print("Loading pieces...")
pieces = {}
for i in range(97):
    pieces[i] = torch.load(f'pieces/piece_{i}.pth', map_location='cpu', weights_only=True)

# Build block data: list of (inp_W, inp_b, out_W, out_b, inp_idx, out_idx)
blocks = []
for inp_idx, out_idx in PAIR_MAP:
    iW = pieces[inp_idx]['weight']  # (96, 48)
    ib = pieces[inp_idx]['bias']     # (96,)
    oW = pieces[out_idx]['weight']  # (48, 96)
    ob = pieces[out_idx]['bias']     # (48,)
    blocks.append((iW, ib, oW, ob, inp_idx, out_idx))

last_W = pieces[LAST_PIECE]['weight']  # (1, 48)
last_b = pieces[LAST_PIECE]['bias']    # (1,)

relu = nn.ReLU()

def apply_block(x, block):
    iW, ib, oW, ob, _, _ = block
    h = x @ iW.T + ib
    h = relu(h)
    r = h @ oW.T + ob
    return x + r

def forward_pass(x, order):
    for idx in order:
        x = apply_block(x, blocks[idx])
    return (x @ last_W.T + last_b).squeeze()

def compute_mse(order, x=None, p=None):
    if x is None: x = X_all
    if p is None: p = pred_all
    return ((forward_pass(x.clone(), order) - p)**2).mean().item()

# ============================================================
# Step 1: Seed order by ascending ||W_out||_F
# ============================================================
print("\n=== Seed: ascending ||W_out||_F ===")
out_frob_norms = []
for i, (iW, ib, oW, ob, inp_idx, out_idx) in enumerate(blocks):
    out_frob_norms.append(torch.norm(oW, 'fro').item())

seed_order = sorted(range(48), key=lambda i: out_frob_norms[i])
seed_mse = compute_mse(seed_order)
print(f"Seed MSE: {seed_mse:.10f}")

# ============================================================
# Step 2: Adjacent-swap hill climbing
# ============================================================
print("\n=== Adjacent-swap hill climbing ===")

current_order = list(seed_order)
current_mse = seed_mse
total_swaps = 0

for round_num in range(100):
    improved = False
    swaps_this_round = 0

    for i in range(47):
        new_order = list(current_order)
        new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
        new_mse = compute_mse(new_order)

        if new_mse < current_mse:
            current_order = new_order
            current_mse = new_mse
            improved = True
            swaps_this_round += 1
            total_swaps += 1

    print(f"  Round {round_num}: MSE = {current_mse:.12f}, swaps = {swaps_this_round}")

    if not improved:
        print(f"  Converged after {round_num} rounds, {total_swaps} total swaps")
        break

# ============================================================
# Step 3: Try non-adjacent swaps too
# ============================================================
print("\n=== General swap hill climbing ===")
for round_num in range(50):
    improved = False
    for i in range(48):
        for j in range(i+1, 48):
            new_order = list(current_order)
            new_order[i], new_order[j] = new_order[j], new_order[i]
            new_mse = compute_mse(new_order)
            if new_mse < current_mse:
                current_order = new_order
                current_mse = new_mse
                improved = True
                total_swaps += 1
                print(f"  Swap ({i},{j}): MSE = {current_mse:.12f}")

    if not improved:
        print(f"  Converged after round {round_num}")
        break

# ============================================================
# Validation
# ============================================================
print("\n=== Full Validation ===")
final_pred = forward_pass(X_all.clone(), current_order)
mse = ((final_pred - pred_all)**2).mean().item()
max_err = (final_pred - pred_all).abs().max().item()
corr = torch.corrcoef(torch.stack([final_pred, pred_all]))[0, 1].item()
print(f"MSE:      {mse:.15f}")
print(f"Max err:  {max_err:.15f}")
print(f"Corr:     {corr:.15f}")

# Show first few predictions
print("\nFirst 5 predictions vs expected:")
for i in range(5):
    print(f"  pred={final_pred[i].item():.10f}, expected={pred_all[i].item():.10f}, diff={abs(final_pred[i].item()-pred_all[i].item()):.2e}")

# ============================================================
# Build permutation
# ============================================================
permutation = []
for block_idx in current_order:
    inp_idx, out_idx = PAIR_MAP[block_idx]
    permutation.append(inp_idx)
    permutation.append(out_idx)
permutation.append(LAST_PIECE)

print(f"\nPermutation ({len(permutation)} elements):")
result = ','.join(map(str, permutation))
print(result)

with open('solution.txt', 'w') as f:
    f.write(result)
print("\nSaved to solution.txt")

# Sanity check
assert len(permutation) == 97
assert len(set(permutation)) == 97
assert all(0 <= p <= 96 for p in permutation)
print("Sanity check passed: 97 unique indices, all in [0, 96]")
