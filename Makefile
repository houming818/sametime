# SaTi - 本地开发工作流
.PHONY: install eval fine test deploy clean deps hello_sati \
        wmt_phase0 wmt_phase1_0 wmt_phase1_1 wmt_phase2 wmt_phase3 wmt_phase4_bpe wmt_phase6

DOCKER_HOST ?= ssh://houming818@io.grepcode.cn
REGISTRY ?= reg.grepcode.cn/sati

# === sync nessary files to reg ===
build_base_images:
	docker pull nvidia/cuda:12.1.0-runtime-ubuntu22.04
	docker tag nvidia/cuda:12.1.0-runtime-ubuntu22.04 $(REGISTRY)/cuda:12.1.0-runtime-ubuntu22.04
	docker push $(REGISTRY)/cuda:12.1.0-runtime-ubuntu22.04
    docker build -t reg.grepcode.cn/sati/sametime-base:cu121-py310 benchmark/hello_sati/base
	docker push reg.grepcode.cn/sati/sametime-base:cu121-py310

# === Benchmark 快速启动 ===
hello_sati:
	DOCKER_HOST=$(DOCKER_HOST) docker build -t $(REGISTRY)/sametime-hello:latest benchmark/hello_sati/
	DOCKER_HOST=$(DOCKER_HOST) docker run --gpus all --rm $(REGISTRY)/sametime-hello:latest

# === WMT Phase 训练（Phase 0→6） ===
# rsync 代码到 io 持久卷，然后 docker compose 运行
WMT_CODE = /data/homecicd/sametime/code/wmt
WMT_DIR = benchmark/wmt

# wmt_base target: rsync + mkdir. Depended by all wmt phase targets.
wmt_base:
	ssh houming818@io.grepcode.cn 'sudo mkdir -p /data/homecicd/sametime/{code,datasets,checkpoints,logs,huggingface} && sudo chown houming818:houming818 /data/homecicd/sametime && sudo rm -rf $(WMT_CODE) && mkdir -p $(WMT_CODE)'
	rsync -az --delete $(WMT_DIR)/ houming818@io.grepcode.cn:$(WMT_CODE)/

# Each phase runs via docker compose
wmt_phase0: wmt_base
	DOCKER_HOST=$(DOCKER_HOST) docker compose -f $(WMT_DIR)/phase0_skeleton/docker-compose.yml run --rm phase
wmt_phase1_0: wmt_base
	DOCKER_HOST=$(DOCKER_HOST) docker compose -f $(WMT_DIR)/phase1_0_rnn/docker-compose.yml run --rm phase $(ARGS)
wmt_phase1_1: wmt_base
	DOCKER_HOST=$(DOCKER_HOST) docker compose -f $(WMT_DIR)/phase1_1_lstm/docker-compose.yml run --rm phase $(ARGS)
wmt_phase1: wmt_phase1_1
wmt_phase2: wmt_base
	DOCKER_HOST=$(DOCKER_HOST) docker compose -f $(WMT_DIR)/phase2_bahdanau/docker-compose.yml run --rm phase $(ARGS)
wmt_phase3: wmt_base
	DOCKER_HOST=$(DOCKER_HOST) docker compose -f $(WMT_DIR)/phase3_luong/docker-compose.yml run --rm phase $(ARGS)
wmt_phase4_bpe: wmt_base
	DOCKER_HOST=$(DOCKER_HOST) docker compose -f $(WMT_DIR)/phase4_bpe/docker-compose.yml run --rm phase
wmt_phase6: wmt_base
	DOCKER_HOST=$(DOCKER_HOST) docker compose -f $(WMT_DIR)/phase6_transformer/docker-compose.yml run --rm phase $(ARGS)

# 安装环节
install:
	pip install -r requirements.txt
	pre-commit install

# 拉取基础模型
deps:
	python scripts/data/download_models.py

# 评估任务
eval:
	python scripts/eval/run_benchmark.py

# 数据预处理
preprocess:
	python scripts/preprocess/clean_data.py

# LoRA 微调训练
fine:
	python scripts/finetune/train.py \
	  --model_name meta-llama/Llama-3-8B \
	  --output_path models/llama-3-8b-lora \
	  --epochs 3

# 训练与评估流程
train: preprocess fine eval

# 回滚操作
clean:
	rm -rf *.tmp/*.p* *.pt/*.p*

# 部署到 ZooDeploy
deploy:
	@\
	docker build -t sa_ti:latest . \
  &&\
	docker push \
  &&\
	echo "✅ Image deployed to reg.grepcode.cn/nio/SaTi:sa_ti:latest"

# SSH 同步到 ZooDeploy 集群
sync:
	@\
	rsync -avz --delete -e "ssh -p 22" \
	  SaTi/ sedcode@c12.sedcode.cn:/opt/ZooDeploy/SaTi/
