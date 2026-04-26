FROM reg.grepcode.cn/sati/nvidia_cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 && \
    ln -sf /usr/bin/python3 /usr/bin/python

RUN pip3 install --no-cache-dir datasets sacrebleu sentencepiece
