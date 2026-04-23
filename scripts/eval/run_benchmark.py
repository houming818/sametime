#!/usr/bin/env python3
"""
SaTi 模型评估基准脚本，支持 MMLU/GSM8K/ARC 等开源评分级。

Usage:
  python scripts/eval/run_benchmark.py --model_name llm-3-8b --benchmark mmlu_pro
"""

import os
import sys
import argparse
import json
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from evaluate import load

# 基准级注册表
BENCHMARK_REGISTRY = {
    "mmlu_pro": {
        "url": "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/master/datasets/mmlu_pro.json",
        "metrics": ["accuracy", "semantic_acc"],
    },
    "mmlu": {
        "url": "https://huggingface.co/datasets/cair/mmlu/raw/main/test/v1.json",
        "metrics": ["acc"],
    },
    "gsm8k": {
        "url": "https://huggingface.co/datasets/gsm8k/raw/main/test.json",
        "metrics": ["average", "exact"],
    },
    "human_eval": {
        "url": "https://huggingface.co/datasets/openai/human_eval/raw/main/test.json",
        "metrics": ["pass_rate"],
    },
    "bigbench_hard": {
        "url": "https://huggingface.co/datasets/google/big-bench/raw/main/bigbench-hard.json",
        "metrics": ["accuracy"],
    },
    "hellaswag": {
        "url": "https://huggingface.co/datasets/huggingface/hoonai/evaluate-hellaswag/raw/main/hellaswag.json",
        "metrics": ["acc"],
    },
    "arc": {
        "url": "https://huggingface.co/datasets/AIb/arc/raw/main/collection.json",
        "metrics": ["accuracy"],
    },
    "truthfulqa": {
        "url": "https://huggingface.co/datasets/truthfulqa/raw/main/test.json",
        "metrics": ["truthfulness", "faithfulness"],
    },
}

def load_model(model_name):
    """加载模型"""
    print(f"⚙️ Loading model: {model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

def generate_response(prompt, model, tokenizer, max_length=512):
    """生成模型回答"""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False
    )
    outputs = model.generate(
        inputs=tokenized_input,
        max_length=max_length,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0])

def load_dataset(name):
    """加载基准数据集"""
    try:
        return load_dataset(name, split="test")
    except:
        return pd.read_json(BENCHMARK_REGISTRY[name]["url"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama-3-8b")
    parser.add_argument("--benchmark", type=str, default="mmlu_pro")
    parser.add_argument("--output_path", type=str, default="eval/results")
    args = parser.parse_args()

    # 加载模型
    model_name = args.model_name
    model, tokenizer = load_model(model_name)

    # 加载数据
    output_path = "./eval/results"
    output_path = f"{output_path}/{args.benchmark}"
    os.makedirs(output_path, exist_ok=True)

    dataset = load_dataset(args.dataset)

    # 评估模型
    print(f"🔍 Running evaluation on {args.benchmark}")
    # ... 评估逻辑 ...

    print(f"✅ Evaluation complete")
    # 输出结果到文件
    save_path = f"{output_path}/result_{args.model_name}_{args.dataset}.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
