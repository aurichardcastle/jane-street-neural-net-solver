"""
48-block ordering solver.
Fixed pairing from AGENT_CHAT.md. Fixed last piece = 85.
Seed by ascending ||W_out||_F, then adjacent-swap hill climbing.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

torch.set_grad_enabled(False)

# === Fixed pair map from AGENT_CHAT.md ===
PAIR_MAP = [
    (0, 75), (1, 40), (2, 51), (3, 53), (4, 17), (5, 21), (10, 20), (13, 7),
    (14, 33), (15, 67), (16, 83), (18, 25), (23, 46), (27, 76), (28, 12), (31, 36),
    (35, 24), (37, 6), (39, 38), (41, 92), (42, 55), (43, 34), (44, 82), (45, 19),
    (48, 9), (49, 72), (50, 66), (56, 30), (58, 78), (59, 52), (60, 93), (61, 80),
    (62, 79), (64, 70), (65, 22), (68, 47), (69, 89), (73, 11), (74, 90), (77, 32),
    (81, 8), (84, 63), (86, 57), (87, 71), (88, 54), (91, 29), (94, 96), (95, 26),
]
LAST_PIECE = 85

# === Load data ===
print("Loading data...")
data = pd.read_csv('historical_data.csv')
X_all = torch.tensor(data.iloc[:, :48].values, dtype=torch.float32)
pred_all = torch.tensor(data['pred'].values, dtype=torch.float32)
N = len(pred_all)
print(f"Data: {N} rows, 48 features")

# === Load pieces, build blocks ===
print("Loading pieces...")
pieces = {}
for i in range(97):
    pieces[i] = torch.load(f'pieces/piece_{i}.pth', map_location='cpu', weights_only=True)

class Block:
    def __init__(self, inp_idx, out_idx):
        self.inp_idx = inp_idx
        self.out_idx = out_idx
        self.win = pieces[inp_idx]['weight']   # (96, 48)
        self.bin = pieces[inp_idx]['bias']      # (96,)
        self.wout = pieces[out_idx]['weight']   # (48, 96)
        self.bout = pieces[out_idx]['bias']     # (48,)
        self.wout_f = torch.norm(self.wout, 'fro').item()

blocks = [Block(inp, out) for inp, out in PAIR_MAP]

w_last = pieces[LAST_PIECE]['weight']  # (1, 48)
b_last = pieces[LAST_PIECE]['bias']    # (1,)

relu = nn.ReLU()

# === Forward pass and MSE ===
def forward(order, x):
    for idx in order:
        b = blocks[idx]
        h = x @ b.win.T + b.bin
        h = relu(h)
        r = h @ b.wout.T + b.bout
        x = x + r
    return (x @ w_last.T + b_last).squeeze()

def mse(order):
    return ((forward(order, X_all.clone()) - pred_all) ** 2).mean().item()

# === Step 1: Seed by ascending ||W_out||_F ===
order = sorted(range(48), key=lambda i: blocks[i].wout_f)
seed_mse = mse(order)
print(f"\nSeed MSE (ascending ||W_out||_F): {seed_mse:.16f}")
print(f"Expected:                         0.07585128396749496")
print(f"Match: {'YES' if abs(seed_mse - 0.07585128396749496) < 1e-6 else 'NO — check forward pass!'}")

# === Step 2: Adjacent-swap hill climbing ===
print("\n=== Adjacent-swap hill climbing ===")
current_mse = seed_mse
round_num = 0

while True:
    improved = False
    swaps = 0
    for i in range(47):
        cand = list(order)
        cand[i], cand[i+1] = cand[i+1], cand[i]
        cand_mse = mse(cand)
        if cand_mse < current_mse - 1e-12:
            order = cand
            current_mse = cand_mse
            improved = True
            swaps += 1
    round_num += 1
    print(f"  Round {round_num}: MSE = {current_mse:.16f}, swaps = {swaps}")
    if not improved:
        break

print(f"After adjacent swaps: MSE = {current_mse:.16f}")

# === Step 3: If stalled, try insertion moves ===
if current_mse > 1e-10:
    print("\n=== Insertion moves ===")
    ins_round = 0
    while True:
        improved = False
        for i in range(48):
            for j in range(48):
                if i == j or i == j - 1:
                    continue
                cand = list(order)
                block = cand.pop(i)
                cand.insert(j if j < i else j, block)
                cand_mse = mse(cand)
                if cand_mse < current_mse - 1e-12:
                    order = cand
                    current_mse = cand_mse
                    improved = True
                    print(f"  Insert block from pos {i} to {j}: MSE = {current_mse:.16f}")
        ins_round += 1
        if not improved:
            break
        # Re-run adjacent swaps after any insertion
        print("  Re-running adjacent swaps...")
        while True:
            adj_improved = False
            for i in range(47):
                cand = list(order)
                cand[i], cand[i+1] = cand[i+1], cand[i]
                cand_mse = mse(cand)
                if cand_mse < current_mse - 1e-12:
                    order = cand
                    current_mse = cand_mse
                    adj_improved = True
            if not adj_improved:
                break
            print(f"  After adj repair: MSE = {current_mse:.16f}")

    print(f"After insertions: MSE = {current_mse:.16f}")

# === Step 4: If still stalled, try all pair swaps ===
if current_mse > 1e-10:
    print("\n=== All pair swaps ===")
    while True:
        improved = False
        for i in range(48):
            for j in range(i+1, 48):
                cand = list(order)
                cand[i], cand[j] = cand[j], cand[i]
                cand_mse = mse(cand)
                if cand_mse < current_mse - 1e-12:
                    order = cand
                    current_mse = cand_mse
                    improved = True
                    print(f"  Swap ({i},{j}): MSE = {current_mse:.16f}")
        if not improved:
            break
        # Re-run adjacent swaps
        while True:
            adj_improved = False
            for i in range(47):
                cand = list(order)
                cand[i], cand[i+1] = cand[i+1], cand[i]
                cand_mse = mse(cand)
                if cand_mse < current_mse - 1e-12:
                    order = cand
                    current_mse = cand_mse
                    adj_improved = True
            if not adj_improved:
                break
            print(f"  After adj repair: MSE = {current_mse:.16f}")

    print(f"After pair swaps: MSE = {current_mse:.16f}")

# === Final validation ===
print(f"\n=== FINAL RESULT ===")
final_mse = mse(order)
print(f"MSE: {final_mse:.16e}")

final_pred = forward(order, X_all.clone())
max_err = (final_pred - pred_all).abs().max().item()
print(f"Max err: {max_err:.16e}")

print("\nFirst 5 predictions vs expected:")
for i in range(5):
    print(f"  got={final_pred[i].item():.12f}, want={pred_all[i].item():.12f}, diff={abs(final_pred[i].item()-pred_all[i].item()):.2e}")

# === Emit permutation ===
permutation = []
for idx in order:
    inp_idx, out_idx = PAIR_MAP[idx]
    permutation.append(inp_idx)
    permutation.append(out_idx)
permutation.append(LAST_PIECE)

assert len(permutation) == 97
assert len(set(permutation)) == 97
assert all(0 <= p <= 96 for p in permutation)

result = ','.join(map(str, permutation))
print(f"\nPermutation:\n{result}")

with open('solution.txt', 'w') as f:
    f.write(result)
print("\nSaved to solution.txt")
