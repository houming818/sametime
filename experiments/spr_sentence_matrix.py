"""
SPR Sentence-Level Echo — Test Matrix
16 combinations: {Gumbel,Soft} × {Mean, Cumsum} × {MLP, GRU} × {M=2, M=4}
100 toy sentences, d=32, depth=3, 50 epochs each
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, json
from collections import Counter

# ──── Toy Data Generator ────
def gen_toy_sentences(n_sents=100, vocab_size=200, max_len=10, seed=42):
    """Generate random sentences from synthetic vocab (fixed-length for testing)"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    sentences = []
    for _ in range(n_sents):
        L = np.random.randint(3, max_len + 1)
        ids = list(np.random.randint(1, vocab_size, size=L))
        sentences.append(ids)
    return sentences

# ──── Hash Functions ────
def hash_mean(E, ids, W_hash):
    T = len(ids)
    pos = torch.arange(T, device=ids.device).unsqueeze(1).float()
    d_out = E.shape[1]
    phase = pos / (10000 ** (torch.arange(0, d_out, 2, device=ids.device).float() / d_out))
    pos_emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
    emb = E[ids] + 0.1 * pos_emb
    return torch.tanh(W_hash @ emb.mean(dim=0))

def hash_cumsum(E, ids, W_hash):
    T = len(ids)
    pos = torch.arange(T, device=ids.device).unsqueeze(1).float()
    d_out = E.shape[1]
    phase = pos / (10000 ** (torch.arange(0, d_out, 2, device=ids.device).float() / d_out))
    pos_emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
    emb = E[ids] + 0.1 * pos_emb
    return torch.tanh(W_hash @ emb.cumsum(dim=0)[-1] / T)

# ──── SPR Tree ────
class SPRTree(nn.Module):
    def __init__(self, d, depth, use_gumbel=True, tau=0.5):
        super().__init__()
        self.depth = depth
        self.n_leaves = 1 << depth
        self.use_gumbel = use_gumbel
        self.tau = tau
        self.W_proj = nn.Parameter(torch.randn(d, d) * 0.02)
        self.W_split = nn.ParameterList([
            nn.Parameter(torch.randn(d, d) * 0.02)
            for _ in range(depth)
        ])
        self.E_leaf = nn.Parameter(torch.randn(self.n_leaves, d) * 0.02)

    def forward(self, x):
        """x: [d] → leaf vector [d]; gradient flows via soft-weighted sum of E_leaf"""
        x = self.W_proj @ x  # per-tree projection → look at different aspects
        leaf_probs = torch.ones(self.n_leaves, device=x.device)
        for level in range(self.depth):
            logit = (x @ self.W_split[level]).sum()
            if self.use_gumbel and self.training:
                logits_2 = torch.stack([logit, -logit])
                choice = F.gumbel_softmax(logits_2, tau=self.tau, hard=True)
                left_prob, right_prob = choice[0], choice[1]
            else:
                left_prob = torch.sigmoid(logit)
                right_prob = 1.0 - left_prob
            bit_pos = self.depth - 1 - level
            mask = ((torch.arange(self.n_leaves, device=x.device) >> bit_pos) & 1).float()
            level_prob = mask * right_prob + (1.0 - mask) * left_prob
            leaf_probs = leaf_probs * level_prob
        return (leaf_probs.unsqueeze(1) * self.E_leaf).sum(dim=0)

# ──── Decoder ────
class MLPDecoder(nn.Module):
    def __init__(self, d_in, d, V, max_len=12):
        super().__init__()
        self.max_len = max_len
        self.V = V
        self.proj = nn.Sequential(
            nn.Linear(d_in, d * 4),
            nn.Tanh(),
            nn.Linear(d * 4, max_len * d),
        )

    def forward(self, leaf_concat, E, gold_ids=None):
        """leaf_concat: [M*d], output: logits [max_len, V]"""
        h = self.proj(leaf_concat)  # [max_len * d]
        h = h.view(self.max_len, -1)  # [max_len, d]
        logits = h @ E.T  # [max_len, V]
        return logits

class GRUDecoder(nn.Module):
    def __init__(self, d_in, d, V, max_len=12):
        super().__init__()
        self.max_len = max_len
        self.V = V
        self.init_h = nn.Linear(d_in, d)
        self.gru = nn.GRUCell(d, d)
        self.W_out = nn.Linear(d, V)

    def forward(self, leaf_concat, E, gold_ids=None):
        h = self.init_h(leaf_concat)
        logits_list = []
        prev = torch.zeros(1, device=leaf_concat.device).long()
        for t in range(self.max_len):
            if self.training and gold_ids is not None and t < len(gold_ids):
                inp = E[gold_ids[t]].unsqueeze(0)
            else:
                inp = E[prev].unsqueeze(0)
            h = h.unsqueeze(0) if h.dim() == 1 else h
            h = self.gru(inp, h).squeeze(0)
            logits_list.append(self.W_out(h))
            if not (self.training and gold_ids is not None):
                prev = logits_list[-1].argmax(dim=-1)
        return torch.stack(logits_list, dim=0)

# ──── Sentence SPR Model ────
class SentenceSPR(nn.Module):
    def __init__(self, V, d=32, depth=3, M=4, use_gumbel=True, enc_type='mean',
                 dec_type='mlp', max_len=12):
        super().__init__()
        self.d = d
        self.max_len = max_len
        self.enc_type = enc_type
        self.dec_type = dec_type

        self.E = nn.Parameter(torch.randn(V, d) * 0.02)
        self.W_hash = nn.Parameter(torch.randn(d, d) * 0.02)
        self.trees = nn.ModuleList([SPRTree(d, depth, use_gumbel) for _ in range(M)])

        d_in = M * d
        if dec_type == 'mlp':
            self.decoder = MLPDecoder(d_in, d, V, max_len)
        else:
            self.decoder = GRUDecoder(d_in, d, V, max_len)

    def forward(self, ids):
        T = len(ids)
        if self.enc_type == 'mean':
            sent_emb = hash_mean(self.E, ids, self.W_hash)
        else:
            sent_emb = hash_cumsum(self.E, ids, self.W_hash)

        leaf_vecs = [tree(sent_emb) for tree in self.trees]
        leaf_concat = torch.cat(leaf_vecs, dim=0)  # [M*d]

        logits = self.decoder(leaf_concat, self.E, ids)
        return logits, sent_emb


# ──── Metrics ────
def ng(t, n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]

def compute_bleu(refs, hyps):
    C = Counter
    ps = []
    for n in range(1, 5):
        mch, ttl = 0, 0
        for r, h in zip(refs, hyps):
            rc = C(ng(r, n)); hc = C(ng(h, n))
            ttl += sum(hc.values())
            mch += sum(min(hc[k], rc.get(k, 0)) for k in hc)
        ps.append(mch / max(ttl, 1) if ttl > 0 else 1.0)
    bpv = [1 - len(r) / max(len(h), 1) for r, h in zip(refs, hyps) if len(h) > 0]
    bp = min(1.0, math.exp(max(bpv) if bpv else 0))
    return bp * math.exp(sum(math.log(max(p, 1e-10)) for p in ps) / 4) * 100


# ──── Run One Combo ────
def run_combo(use_gumbel, enc_type, dec_type, M, n_sents=100, vocab_size=200,
              max_len=10, d=32, depth=3, epochs=50, lr=0.01):
    label = f"gumbel={use_gumbel}|enc={enc_type}|dec={dec_type}|M={M}"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    np.random.seed(42)

    # Data
    train_sents = gen_toy_sentences(n_sents, vocab_size, max_len, 42)
    val_sents = train_sents[:20]  # use same for val (echo task = reconstruction)

    model = SentenceSPR(vocab_size, d, depth, M, use_gumbel, enc_type, dec_type, max_len).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'loss': [], 'bleu': []}
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss, total_tok = 0, 0
        for ids in train_sents:
            ids_t = torch.tensor(ids, device='cuda', dtype=torch.long)
            logits, _ = model(ids_t)
            T = len(ids)
            targets = ids_t
            if T > logits.shape[0]:
                logits = logits[:T]
            loss = F.cross_entropy(logits[:T], targets, ignore_index=-1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * T
            total_tok += T

        avg_loss = total_loss / max(total_tok, 1)
        history['loss'].append(avg_loss)

        if epoch % 25 == 0 or epoch == epochs - 1:
            # Eval BLEU
            model.eval()
            refs, hyps = [], []
            with torch.no_grad():
                for ids in val_sents[:10]:
                    ids_t = torch.tensor(ids, device='cuda', dtype=torch.long)
                    logits, _ = model(ids_t)
                    pred_ids = logits[:len(ids)].argmax(dim=-1).cpu().tolist()
                    refs.append(ids[:len(pred_ids)])
                    hyps.append(pred_ids)
            b = compute_bleu(refs, hyps)
            history['bleu'].append(b)
            elapsed = time.time() - t0
            print(f"  epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | BLEU={b:.1f} | {elapsed:.1f}s")

    final_loss = history['loss'][-1]
    final_bleu = history['bleu'][-1]
    grad_norms = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    elapsed = time.time() - t0

    result = {
        'label': label,
        'use_gumbel': use_gumbel,
        'enc_type': enc_type,
        'dec_type': dec_type,
        'M': M,
        'final_loss': final_loss,
        'final_bleu': final_bleu,
        'grad_norm': grad_norms,
        'time_s': elapsed,
    }
    print(f"  => loss={final_loss:.4f} BLEU={final_bleu:.1f} grad={grad_norms:.2f} time={elapsed:.0f}s")
    return result


# ──── Main ────
if __name__ == '__main__':
    print("SPR Sentence Echo — Test Matrix")
    print("=" * 60)

    combos = []
    for use_gumbel in [True, False]:
        for enc_type in ['mean', 'cumsum']:
            for dec_type in ['mlp', 'gru']:
                for M in [2, 4]:
                    combos.append((use_gumbel, enc_type, dec_type, M))

    print(f"\nTotal: {len(combos)} combinations\n")

    results = []
    for i, (g, e, d, m) in enumerate(combos):
        print(f"\n[{i+1}/{len(combos)}]", end="", flush=True)
        try:
            r = run_combo(
                use_gumbel=g, enc_type=e, dec_type=d, M=m,
                n_sents=100, vocab_size=200, max_len=10,
                d=128, depth=6, epochs=200, lr=0.01
            )
            results.append(r)
        except Exception as ex:
            print(f"  ERROR: {ex}")
            results.append({'label': f"gumbel={g}|enc={e}|dec={d}|M={m}", 'error': str(ex)})

    # ──── Summary ────
    print(f"\n\n{'='*60}")
    print("SUMMARY — Best performing combinations")
    print(f"{'='*60}")

    valid = [r for r in results if 'final_bleu' in r]
    sorted_by_bleu = sorted(valid, key=lambda x: -x['final_bleu'])

    print(f"\n{'Rank':<5} {'BLEU':>6} {'Loss':>6} {'Grad':>8} {'Time':>6}  Params")
    print("-" * 80)
    for i, r in enumerate(sorted_by_bleu):
        print(f"  {i+1:<3}  {r['final_bleu']:5.1f}  "
              f"{r['final_loss']:5.3f}  {r['grad_norm']:7.2f}  {r['time_s']:5.0f}s  "
              f"gumbel={r['use_gumbel']} enc={r['enc_type']} dec={r['dec_type']} M={r['M']}")

    # Save
    with open('/tmp/spr_matrix_results.json', 'w') as f:
        json.dump([{k: v for k, v in r.items() if not isinstance(v, bool)} for r in results], f, indent=2)
    print(f"\nSaved to /tmp/spr_matrix_results.json")

    # Best combo
    best = sorted_by_bleu[0]
    print(f"\n🏆 BEST: gumbel={best['use_gumbel']} enc={best['enc_type']} "
          f"dec={best['dec_type']} M={best['M']} BLEU={best['final_bleu']:.1f}")
