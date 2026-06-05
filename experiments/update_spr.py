import re

with open('/home/nio/log/holds/SameTime/experiments/spr_l2_train.py', 'r') as f:
    content = f.read()

args_code = """
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--rep_penalty', type=float, default=1.2)
parser.add_argument('--data_path', type=str, default='/mnt/nas/datasets/wmt17/train.zh-en')
parser.add_argument('--bpe_model', type=str, default='/mnt/nas/datasets/wmt17/sp_bpe.model')
parser.add_argument('--ckpt_l1', type=str, default='/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt')
args = parser.parse_args()

D_MODEL_L1 = 128
D_MODEL_L2 = args.d_model
TD = 5
VOCAB_SIZE = 16000
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.lr
MAX_LEN = 80
CKPT_L1 = args.ckpt_l1
BPE_MODEL = args.bpe_model
DATA_PATH = args.data_path
REP_PENALTY = args.rep_penalty
"""

# Replace hypers
content = re.sub(r'import argparse.*?REP_PENALTY = args\.rep_penalty\n', args_code, content, flags=re.DOTALL)

with open('/home/nio/log/holds/SameTime/experiments/spr_l2_train.py', 'w') as f:
    f.write(content)
