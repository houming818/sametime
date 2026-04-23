# SaTi - 本地开发工作流
.PHONY: install,eval,fine,test,deploy,clean,deps

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
