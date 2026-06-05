"""Diagnose: can Q map anchor words? Single-word EN→ZH test."""
import torch, torch.nn as nn, torch.nn.functional as F
import sentencepiece as spm
device = 'cuda' if torch.cuda.is_available() else 'cpu'

sp = spm.SentencePieceProcessor()
sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V, d = sp.get_piece_size(), 128

class BiGRU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.ep = nn.Linear(d, d)
        self.dec = nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.dp = nn.Linear(d, d)

    def fe(self, x):
        return self.ep(self.enc(x)[0])

    def fd(self, x):
        return self.dp(self.dec(x)[0])

L0 = nn.Embedding(V, d).to(device)
L1 = BiGRU(d).to(device)
ckpt = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_auto.pt', map_location=device)
L0.load_state_dict(ckpt['L0'])
L1.load_state_dict(ckpt['L1'])
L0.eval()
L1.eval()

bridge = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_bridge.pt', map_location=device)
Q = bridge['Q'].to(device)

def ok(ids):
    return all(x != 0 for x in ids)

ANCHOR_WORDS = [
    ('one', '一'), ('two', '二'), ('three', '三'), ('four', '四'), ('five', '五'),
    ('six', '六'), ('seven', '七'), ('eight', '八'), ('nine', '九'), ('ten', '十'),
    ('big', '大'), ('small', '小'), ('good', '好'), ('bad', '坏'),
    ('water', '水'), ('fire', '火'), ('sun', '太阳'), ('earth', '地'),
    ('book', '书'), ('door', '门'), ('go', '去'), ('come', '来'),
    ('eat', '吃'), ('see', '看'), ('love', '爱'), ('hate', '恨'),
    ('man', '男人'), ('woman', '女人'), ('child', '孩子'), ('boy', '男孩'),
    ('girl', '女孩'), ('red', '红'), ('white', '白'), ('black', '黑'),
    ('hot', '热'), ('cold', '冷'), ('new', '新'), ('old', '老'),
    ('long', '长'), ('short', '短'), ('fast', '快'), ('slow', '慢'),
    ('fish', '鱼'), ('bird', '鸟'), ('horse', '马'), ('cow', '牛'),
    ('i', '我'), ('you', '你'), ('he', '他'), ('she', '她'),
    ('we', '我们'), ('they', '他们'), ('war', '战争'), ('peace', '和平'),
    ('year', '年'), ('day', '天'), ('night', '夜'), ('week', '周'),
    ('morning', '早上'), ('evening', '晚上'), ('yesterday', '昨天'), ('today', '今天'),
    ('mother', '母亲'), ('father', '父亲'), ('friend', '朋友'), ('enemy', '敌人'),
    ('say', '说'), ('write', '写'), ('read', '读'), ('sing', '唱'),
    ('happy', '幸福'), ('sad', '悲伤'), ('angry', '愤怒'),
    ('king', '国王'), ('queen', '女王'),
    ('gold', '金'), ('iron', '铁'), ('stone', '石'), ('wood', '木'),
    ('meat', '肉'), ('bread', '面包'), ('apple', '苹果'), ('tea', '茶'),
]

test_pairs = [(sp.encode_as_ids(w), sp.encode_as_ids(z)) for w, z in ANCHOR_WORDS]
test_pairs = [(e, z) for e, z in test_pairs if ok(e) and ok(z)]
print(f"Testing {len(test_pairs)} anchor word pairs:\n")

correct_dec = 0
correct_dir = 0
total = 0
with torch.no_grad():
    for en_ids, zh_ids in test_pairs:
        e_en = L0(torch.tensor(en_ids, device=device)).unsqueeze(0)
        e_zh = L0(torch.tensor(zh_ids, device=device)).unsqueeze(0)
        h_en = L1.fe(e_en).squeeze(0).mean(dim=0)
        h_zh = L1.fe(e_zh).squeeze(0).mean(dim=0)

        cos_before = F.cosine_similarity(h_en.unsqueeze(0), h_zh.unsqueeze(0)).item()
        h_pred = h_en @ Q
        cos_after = F.cosine_similarity(h_pred.unsqueeze(0), h_zh.unsqueeze(0)).item()

        # Decoder path
        dz = L1.fd(h_pred.unsqueeze(0).unsqueeze(0))
        logits_dec = dz.squeeze() @ L0.weight.T
        pred_id_dec = logits_dec.argmax().item()
        pred_tok_dec = sp.decode_ids([pred_id_dec])

        # Direct path (no decoder)
        logits_dir = h_pred @ L0.weight.T
        pred_id_dir = logits_dir.argmax().item()
        pred_tok_dir = sp.decode_ids([pred_id_dir])

        gold = sp.decode_ids(zh_ids)
        en = sp.decode_ids(en_ids)
        ok_dec = pred_id_dec == zh_ids[0]
        ok_dir = pred_id_dir == zh_ids[0]
        correct_dec += ok_dec
        correct_dir += ok_dir
        total += 1
        print(f"{en:10s} → gold={gold:10s}  dec={pred_tok_dec:15s}({'Y' if ok_dec else 'N'}) dir={pred_tok_dir:15s}({'Y' if ok_dir else 'N'})  cos_bef={cos_before:.3f} cos_aft={cos_after:.3f}")

print(f"\nAnchor word acc (decoder): {correct_dec}/{total} = {100 * correct_dec / total:.1f}%")
print(f"Anchor word acc (direct):  {correct_dir}/{total} = {100 * correct_dir / total:.1f}%")
