#!/usr/bin/env python3
"""SameTime Hello World - 验证 GPU Docker 环境"""

import torch
import sys
import platform

def main():
    print("=" * 50)
    print("  Hello from SameTime!")
    print("  Study and Tuning for LLMs")
    print("=" * 50)

    print(f"\n  Python version: {sys.version}")
    print(f"  Platform: {platform.platform()}")

    print(f"\n  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")

        x = torch.randn(1000, 1000, device="cuda")
        y = torch.randn(1000, 1000, device="cuda")
        z = torch.matmul(x, y)
        print(f"\n  GPU matmul test: {z.sum().item():.4f} (OK)")
    else:
        print("  No GPU detected — running in CPU mode")

    print("=" * 50)
    print("  Hello world completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
