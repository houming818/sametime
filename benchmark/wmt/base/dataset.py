"""
base/dataset.py — IWSLT14 De-En 数据加载 + word-level Vocab

从原始 XML 文件解析 IWSLT TED 数据，不依赖 HuggingFace datasets。
"""

import os
import re
import xml.etree.ElementTree as ET
import urllib.request
import tarfile
import torch
from torch.utils.data import DataLoader
from collections import Counter


# ═══════════════════════════════════════════
# 数据下载与解析
# ═══════════════════════════════════════════

IWSLT14_URL = "https://wit3.fbk.eu/archive/2016-01//texts/de/en/de-en.tgz"

# 训练/验证/测试的文件名匹配规则
SPLIT_FILES = {
    "train": ["train.tags.de-en.de", "train.tags.de-en.en"],
    "valid": ["IWSLT14.TED.dev2010.de-en.de.xml", "IWSLT14.TED.dev2010.de-en.en.xml"],
    "test":  ["IWSLT14.TED.tst2010.de-en.de.xml", "IWSLT14.TED.tst2010.de-en.en.xml"],
}


def extract_segments(text):
    """从 XML 片段中提取 <seg id="N"> 内的文本"""
    segs = []
    for line in text.split("\n"):
        m = re.search(r'<seg id="\d+">(.*?)</seg>', line)
        if m:
            segs.append(m.group(1).strip())
    return segs


def clean_train_line(line):
    """从训练数据的 tags 文件中提取纯文本"""
    # 去掉 XML 标签行
    if line.startswith("<") and ">" in line:
        return None
    line = line.strip()
    return line if line else None


def parse_iwslt14_tgz(tgz_path, data_dir):
    """下载并解析 IWSLT14 数据，返回 {split: [(de, en), ...]}"""
    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(tgz_path):
        print("  downloading iwslt14 de-en...")
        urllib.request.urlretrieve(IWSLT14_URL, tgz_path)

    result = {}
    with tarfile.open(tgz_path, "r:gz") as tf:
        # 提取所有文件名
        names = tf.getnames()
        prefix = "de-en" if names[0].startswith("de-en") else ""

        for split, (de_name, en_name) in SPLIT_FILES.items():
            de_path = f"{prefix}/{de_name}"
            en_path = f"{prefix}/{en_name}"

            with tf.extractfile(de_path) as f:
                de_raw = f.read().decode("utf-8")
            with tf.extractfile(en_path) as f:
                en_raw = f.read().decode("utf-8")

            if split == "train":
                # train 数据: 纯文本文件，过滤 XML 标签行
                de_lines = [l.strip() for l in de_raw.split("\n")]
                en_lines = [l.strip() for l in en_raw.split("\n")]
                pairs = []
                for d, e in zip(de_lines, en_lines):
                    if d and e and not d.startswith("<"):
                        pairs.append((d, e))
            else:
                # 验证/测试: XML 文件
                de_segs = extract_segments(de_raw)
                en_segs = extract_segments(en_raw)
                pairs = list(zip(de_segs, en_segs))

            # 保存为 tab 分隔文件，方便重复读取
            out_path = os.path.join(data_dir, f"iwslt14.{split}.de-en")
            if not os.path.exists(out_path):
                with open(out_path, "w", encoding="utf-8") as f:
                    for d, e in pairs:
                        f.write(f"{d}\t{e}\n")

            result[split] = pairs

    return result


def load_iwslt14(split="train", data_dir="/data/datasets/iwslt14"):
    split_key = {"validation": "valid"}.get(split, split)
    cache_path = os.path.join(data_dir, f"iwslt14.{split_key}.de-en")

    if os.path.exists(cache_path):
        pairs = []
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "\t" in line:
                    de, en = line.split("\t", 1)
                    pairs.append({"de": de, "en": en})
        return pairs

    # 首次：下载并解析
    tgz_path = os.path.join(data_dir, "de-en.tgz")
    all_data = parse_iwslt14_tgz(tgz_path, data_dir)
    return [{"de": d, "en": e} for d, e in all_data[split_key]]


# ═══════════════════════════════════════════
# 词表
# ═══════════════════════════════════════════

class Vocab:
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self, sentences, min_freq=2):
        counter = Counter()
        for s in sentences:
            counter.update(s.split())
        self.stoi = {"<pad>": 0, "<sos>": 1, "</s>": 2, "<unk>": 3}
        self.itos = {v: k for k, v in self.stoi.items()}
        for word, freq in counter.items():
            if freq >= min_freq:
                idx = len(self.stoi)
                self.stoi[word] = idx
                self.itos[idx] = word

    def __len__(self):
        return len(self.stoi)

    def encode(self, sentence, max_len=50):
        tokens = sentence.split()[: max_len - 2]
        ids = [self.SOS] + [self.stoi.get(t, self.UNK) for t in tokens] + [self.EOS]
        return ids

    def decode(self, ids):
        words = []
        for i in ids:
            if i == self.EOS:
                break
            if i not in (self.PAD, self.SOS):
                words.append(self.itos.get(i, "<unk>"))
        return " ".join(words)


# ═══════════════════════════════════════════
# DataLoader
# ═══════════════════════════════════════════

def collate_fn(batch):
    src_ids, tgt_ids = zip(*batch)
    src_len = [len(s) for s in src_ids]
    tgt_len = [len(t) for t in tgt_ids]
    max_src = max(src_len)
    max_tgt = max(tgt_len)
    src_padded = torch.full((len(batch), max_src), Vocab.PAD, dtype=torch.long)
    tgt_padded = torch.full((len(batch), max_tgt), Vocab.PAD, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_ids, tgt_ids)):
        src_padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        tgt_padded[i, : len(t)] = torch.tensor(t, dtype=torch.long)
    return src_padded, tgt_padded, torch.tensor(src_len), torch.tensor(tgt_len)


def build_dataloader(split, vocab_src, vocab_tgt, batch_size=64, shuffle=True, max_len=50, data_dir="/data/datasets/iwslt14"):
    raw = load_iwslt14(split, data_dir)
    data = []
    for pair in raw:
        src_ids = vocab_src.encode(pair["de"], max_len)
        tgt_ids = vocab_tgt.encode(pair["en"], max_len)
        data.append((src_ids, tgt_ids))
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
