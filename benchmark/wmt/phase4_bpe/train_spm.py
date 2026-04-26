"""
训练 SentencePiece 模型（BPE），词表大小 8000。

输出: checkpoints/spm_de.model, checkpoints/spm_en.model
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sentencepiece as spm
from base.dataset import load_iwslt14


def train_spm(sentences, model_prefix, vocab_size=8000):
    """训练 SentencePiece 模型"""
    input_path = f"/tmp/{model_prefix}.txt"
    with open(input_path, "w") as f:
        for s in sentences:
            f.write(s + "\n")
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=os.path.join("checkpoints", model_prefix),
        vocab_size=vocab_size,
        character_coverage=1.0,   # 德语/英语都是拉丁字母
        model_type="bpe",
    )
    os.remove(input_path)


def main():
    os.makedirs("checkpoints", exist_ok=True)
    print("  loading iwslt14...")
    train_raw = load_iwslt14("train")
    de_sentences = [p["de"] for p in train_raw]
    en_sentences = [p["en"] for p in train_raw]

    print("  training SPM for DE (vocab_size=8000)...")
    train_spm(de_sentences, "spm_de", vocab_size=8000)

    print("  training SPM for EN (vocab_size=8000)...")
    train_spm(en_sentences, "spm_en", vocab_size=8000)

    print("  done: checkpoints/spm_{de,en}.model")


if __name__ == "__main__":
    main()
