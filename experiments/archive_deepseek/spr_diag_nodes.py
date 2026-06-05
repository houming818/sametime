"""Diagnose: what do heap tree nodes learn after InfoNCE?"""
import torch, torch.nn as nn, torch.nn.functional as F, sentencepiece as spm
device = 'cuda' if torch.cuda.is_available() else 'cpu'; d = 128
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V = sp.get_piece_size()
def ok(ids): return all(x != 0 for x in ids)

# Load the tree_nce checkpoint
ckpt = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt', map_location=device)

# Rebuild the tree structure (depth=5)
td = 5
t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
# Load from ckpt — but ckpt only has L0 and t_merge, not t_nodes
# Need to load from the actual model state

print("Note: anchor_tree_nce.pt doesn't save t_nodes weights.")
print("Running diagnostic on random-init tree nodes instead.")
print("This shows the STRUCTURE, not the learned values.\n")

L0 = nn.Embedding(V, d).to(device)
L0.load_state_dict(ckpt['L0'])
L0.eval()

for tn in t_nodes:
    nn.init.normal_(tn.weight, 0, 0.1)

# Build routing
print(f"=== Tree Node Structure (depth={td}) ===")
total_nodes = sum(2 ** i for i in range(td))
print(f"Total nodes: {total_nodes}")
print(f"Nodes per level: {' + '.join([str(2**i) for i in range(td)])}")
print()

# Token assignment per node (hardcoded routing)
for level in range(td):
    n = 2 ** level
    print(f"Level {level} ({n} nodes, each dim={d}):")
    for ni in range(min(3, n)):
        # Which tokens go to this node?
        tok_range_start = ni * (V // n)
        tok_range_end = (ni + 1) * (V // n) - 1
        # Show sample tokens
        samples = []
        for tid in range(tok_range_start, min(tok_range_end + 1, V), max(1, V // (n * 5))):
            piece = sp.id_to_piece(tid)
            if len(piece) > 2 and piece not in ('<unk>', '<s>', '</s>'):
                samples.append(piece)
            if len(samples) >= 5: break
        print(f"  node[{ni}]: tokens {tok_range_start}-{tok_range_end}, samples: {', '.join(samples[:5])}")
    print()

# What a node actually STORES: a d-dimensional vector
# After training, nodes at deeper levels should develop specialized representations
print("=== Node vector properties (random init) ===")
for level in range(td):
    w = t_nodes[level].weight  # [n_nodes, d]
    norms = w.norm(dim=-1)
    print(f"Level {level}: norm mean={norms.mean():.4f} std={norms.std():.4f}")

# Node-to-node cosine at each level
print("\n=== Inter-node cosine (per level) ===")
for level in range(1, td):
    w = t_nodes[level].weight
    n = 2 ** level
    if n > 1:
        cos = F.cosine_similarity(w[0:1], w[1:2])
        print(f"Level {level}: cos(node0, node1) = {cos.item():.4f}")
