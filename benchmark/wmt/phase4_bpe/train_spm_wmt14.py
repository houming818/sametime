"""
训练 SentencePiece 模型（BPE），WMT14 De-En，词表大小 32000。

输出: checkpoints/wmt14_spm_de.model, checkpoints/wmt14_spm_en.model
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sentencepiece as spm
from base.dataset import load_wmt14_de_en

def train_spm(sentences, model_prefix, vocab_size=32000):
    input_path = f"/tmp/{model_prefix}.txt"
    with open(input_path, "w") as f:
        for s in sentences:
            f.write(s + "\n")
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=os.path.join("checkpoints", model_prefix),
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="bpe",
    )
    os.remove(input_path)

def main():
    os.makedirs("checkpoints", exist_ok=True)
    print("  loading wmt14 de-en...")
    train_raw = load_wmt14_de_en("train")
    de_sentences = [p["de"] for p in train_raw]
    en_sentences = [p["en"] for p in train_raw]
    print(f"  {len(de_sentences)} sentences")

    print("  training SPM for DE (vocab_size=32000)...")
    train_spm(de_sentences, "wmt14_spm_de", vocab_size=32000)

    print("  training SPM for EN (vocab_size=32000)...")
    train_spm(en_sentences, "wmt14_spm_en", vocab_size=32000)

    print("  done: checkpoints/wmt14_spm_{de,en}.model")

if __name__ == "__main__":
    main()
