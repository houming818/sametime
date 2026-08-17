#!/usr/bin/env python3
import argparse
from modelscope import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
path = snapshot_download(args.model, local_dir=args.output)
print(path, flush=True)
