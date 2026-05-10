"""
base/dataset.py — IWSLT14 De-En 数据加载 + word-level Vocab

从原始 XML 文件解析 IWSLT TED 数据，不依赖 HuggingFace datasets。
"""

import json
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
import urllib.request
import tarfile
import torch
from torch.utils.data import DataLoader, Dataset
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
    src_len = [s.size(0) for s in src_ids]
    tgt_len = [t.size(0) for t in tgt_ids]
    max_src = max(src_len)
    max_tgt = max(tgt_len)
    src_padded = torch.full((len(batch), max_src), Vocab.PAD, dtype=torch.long)
    tgt_padded = torch.full((len(batch), max_tgt), Vocab.PAD, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_ids, tgt_ids)):
        src_padded[i, :s.size(0)] = s
        tgt_padded[i, :t.size(0)] = t
    return src_padded, tgt_padded, torch.tensor(src_len), torch.tensor(tgt_len)


def build_dataloader(split, vocab_src, vocab_tgt, batch_size=64, shuffle=True, max_len=50, data_dir="/data/datasets/iwslt14"):
    raw = load_iwslt14(split, data_dir)
    data = []
    for pair in raw:
        src_ids = vocab_src.encode(pair["de"], max_len)
        tgt_ids = vocab_tgt.encode(pair["en"], max_len)
        data.append((src_ids, tgt_ids))
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


# ═══════════════════════════════════════════
# WMT14 De-En
# ═══════════════════════════════════════════
# WMT14 De-En — streaming tokenization + sharded .pt cache
# ═══════════════════════════════════════════


def load_wmt14_de_en(split="train", data_dir="/data/datasets/wmt14"):
    """
    Download WMT14 from HF datasets, write tab-separated cache file.
    Returns list of {"de": str, "en": str} (used only by word-level Vocab).
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, f"wmt14.{split}.de-en")

    if os.path.exists(cache_path):
        return _read_tab(cache_path)

    print(f"  downloading WMT14 De-En ({split})...", flush=True)
    from datasets import load_dataset
    split_map = {"train": "train", "validation": "validation", "valid": "validation", "test": "test"}
    hf_split = split_map.get(split, split)
    ds = load_dataset("wmt14", "de-en", split=hf_split, trust_remote_code=True)

    pairs = []
    with open(cache_path, "w", encoding="utf-8") as f:
        for ex in ds:
            de = ex["translation"]["de"]
            en = ex["translation"]["en"]
            pairs.append({"de": de, "en": en})
            f.write(f"{de}\t{en}\n")
    print(f"  wmt14 {split}: {len(pairs)} pairs", flush=True)
    return pairs


def _read_tab(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "\t" in line:
                de, en = line.split("\t", 1)
                pairs.append({"de": de, "en": en})
    return pairs


def tokenize_and_cache_wmt14(split, vocab_src, vocab_tgt, max_len=128, shard_size=500000, data_dir="/data/datasets/wmt14"):
    """
    Stream-tokenize WMT14 tab file into sharded .pt files (concatenated 1D format).
    Memory: ~200 MB per shard buffer regardless of total dataset size.
    """
    cache_dir = os.path.join(data_dir, "tokenized", split)
    os.makedirs(cache_dir, exist_ok=True)

    tab_path = os.path.join(data_dir, f"wmt14.{split}.de-en")
    if not os.path.exists(tab_path):
        load_wmt14_de_en(split, data_dir)

    n_total = int(subprocess.run(["wc", "-l", tab_path], capture_output=True, text=True).stdout.split()[0])

    t0 = time.time()
    shard_id = 0
    processed = 0
    src_buf, tgt_buf = [], []
    src_n, tgt_n = [], []

    def flush():
        nonlocal shard_id, src_buf, tgt_buf, src_n, tgt_n
        if not src_buf:
            return
        src = torch.cat(src_buf)
        tgt = torch.cat(tgt_buf)
        src_ptr = torch.zeros(len(src_buf) + 1, dtype=torch.int64)
        tgt_ptr = torch.zeros(len(tgt_buf) + 1, dtype=torch.int64)
        for i, n in enumerate(src_n):
            src_ptr[i+1] = src_ptr[i] + n
        for i, n in enumerate(tgt_n):
            tgt_ptr[i+1] = tgt_ptr[i] + n
        path = os.path.join(cache_dir, f"shard_{shard_id:03d}.pt")
        torch.save({"src": src, "tgt": tgt, "src_ptr": src_ptr, "tgt_ptr": tgt_ptr}, path)
        src_buf, tgt_buf, src_n, tgt_n = [], [], [], []
        shard_id += 1

    print(f"  streaming tokenize {n_total} sentences -> shard_{shard_id:03d}.pt ...", flush=True)
    with open(tab_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) != 2:
                continue
            de, en = parts
            s = vocab_src.encode(de, max_len)
            t = vocab_tgt.encode(en, max_len)
            src_buf.append(torch.tensor(s, dtype=torch.int32))
            tgt_buf.append(torch.tensor(t, dtype=torch.int32))
            src_n.append(len(s))
            tgt_n.append(len(t))
            processed += 1
            if len(src_buf) >= shard_size:
                flush()
                r = processed / (time.time() - t0)
                print(f"  shard {shard_id} ({processed}/{n_total}, {r:.0f} s/s, ~{(n_total-processed)/r/60:.0f} min)", flush=True)
    flush()

    with open(os.path.join(cache_dir, "metadata.json"), "w") as f:
        json.dump({"total": n_total, "shards": shard_id, "shard_size": shard_size}, f)
    print(f"  tokenized {n_total} -> {shard_id} shards ({time.time()-t0:.0f}s)", flush=True)


class WMT14CachedDataset(Dataset):
    def __init__(self, src, src_ptr, tgt, tgt_ptr, n):
        self.src, self.src_ptr = src, src_ptr
        self.tgt, self.tgt_ptr = tgt, tgt_ptr
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        s = self.src[self.src_ptr[idx]:self.src_ptr[idx+1]].to(torch.long)
        t = self.tgt[self.tgt_ptr[idx]:self.tgt_ptr[idx+1]].to(torch.long)
        return s, t


def load_wmt14_tokenized(split, data_dir="/data/datasets/wmt14"):
    cache_dir = os.path.join(data_dir, "tokenized", split)
    with open(os.path.join(cache_dir, "metadata.json")) as f:
        meta = json.load(f)

    src_p, tgt_p, src_off, tgt_off = [], [], 0, 0
    src_ptr_all, tgt_ptr_all = [0], [0]

    for i in range(meta["shards"]):
        d = torch.load(os.path.join(cache_dir, f"shard_{i:03d}.pt"), weights_only=True)
        src_p.append(d["src"])
        tgt_p.append(d["tgt"])
        src_ptr_all.extend((d["src_ptr"][1:] + src_off).tolist())
        tgt_ptr_all.extend((d["tgt_ptr"][1:] + tgt_off).tolist())
        src_off += d["src"].size(0)
        tgt_off += d["tgt"].size(0)

    return WMT14CachedDataset(
        torch.cat(src_p), torch.tensor(src_ptr_all, dtype=torch.int64),
        torch.cat(tgt_p), torch.tensor(tgt_ptr_all, dtype=torch.int64),
        meta["total"]
    )


def build_dataloader_wmt14(split, vocab_src, vocab_tgt, batch_size=64, shuffle=True, max_len=128, data_dir="/data/datasets/wmt14"):
    cache_dir = os.path.join(data_dir, "tokenized", split)
    meta_path = os.path.join(cache_dir, "metadata.json")
    if not os.path.exists(meta_path):
        tokenize_and_cache_wmt14(split, vocab_src, vocab_tgt, max_len, data_dir=data_dir)
    ds = load_wmt14_tokenized(split, data_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, pin_memory=True)
