REMOTE_HOST ?= io.grepcode.cn
REMOTE_USER ?= houming818
REMOTE_DIR ?= /data/homecicd/sametime/code/wmt
REMOTE_WMT ?= $(REMOTE_DIR)/holds/SameTime/benchmark/wmt

RSYNC_FLAGS = -avz --delete --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='*.log' --exclude='metrics.jsonl' --exclude='queue.json'

SYNC_DIRS = base phase6_transformer phase1_1_lstm phase2_bahdanau \
  phase0_skeleton phase1_0_rnn phase3_luong phase4_bpe
SYNC_FILES = q.py Makefile run_*.sh

.DEFAULT_GOAL = help

help:
	@echo 'Usage:'
	@echo '  make sync      rsync code to git path on $(REMOTE_HOST)'
	@echo '  make deploy    sync + copy to Docker root level'
	@echo '  make status    q.py status'
	@echo '  make log N=T1  tail experiment log'
	@echo '  make queue     show queue.json'
	@echo '  make start     nohup q.py start'
	@echo '  make kill      docker kill current job'
	@echo '  make retry N=T1  reset to QUEUED'
	@echo '  make gpu       nvidia-smi'
	@echo '  make ps        docker ps'
	@echo '  make loki      query Loki logs (use Q= SINCE= LIMIT=)
	@echo '    ex: make loki Q="{node=\"io.grepcode.cn\"} |= \"epoch\"" SINCE="30 minutes" LIMIT=20'

sync:
	rsync $(RSYNC_FLAGS) $(SYNC_DIRS) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_WMT)/
	rsync -avz $(SYNC_FILES) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_WMT)/

deploy: sync
	ssh $(REMOTE_USER)@$(REMOTE_HOST) 'SRC=$(REMOTE_WMT); DST=$(REMOTE_DIR); \
	  for d in base phase6_transformer phase1_1_lstm phase2_bahdanau; do \
	    cp -r $$$$SRC/$$$$d/* $$$$DST/$$$$d/ 2>/dev/null; \
	  done; \
	  cp $$$$SRC/q.py $$$$DST/q.py; \
	  echo "deployed: q.py base dataset.py phase6_transformer/train_wmt14.py"; \
	'

status:
	ssh $(REMOTE_USER)@$(REMOTE_HOST) "cd $(REMOTE_DIR) && python3 q.py status"

queue:
	ssh $(REMOTE_USER)@$(REMOTE_HOST) "cat $(REMOTE_DIR)/queue.json"

log:
	ssh $(REMOTE_USER)@$(REMOTE_HOST) "tail -50 $(REMOTE_DIR)/q_logs/$(NAME).log"

logs:
	ssh $(REMOTE_USER)@$(REMOTE_HOST) "tail -30 $(REMOTE_DIR)/q_logs/T*.log"

ps:
	ssh $(REMOTE_USER)@$(REMOTE_HOST) "docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' | head -10"

gpu:
	ssh $(REMOTE_USER)@$(REMOTE_HOST) "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"

kill:
	ssh $(REMOTE_USER)@$(REMOTE_HOST) "cd $(REMOTE_DIR) && python3 q.py kill"

retry:
	ssh $(REMOTE_USER)@$(REMOTE_HOST) "cd $(REMOTE_DIR) && python3 q.py retry $(NAME)"

start:
	ssh $(REMOTE_USER)@$(REMOTE_HOST) "cd $(REMOTE_DIR) && nohup python3 q.py start > q_runner.log 2>&1 &"
	@sleep 2
	ssh $(REMOTE_USER)@$(REMOTE_HOST) "python3 $(REMOTE_DIR)/q.py status"

LOKI_SCRIPT = ../../../engines/scripts/query_loki.sh

loki:
	bash $(LOKI_SCRIPT) '$(Q)' '$(SINCE)' $(LIMIT)

fresh: kill
	ssh $(REMOTE_USER)@$(REMOTE_HOST) 'python3 -c "import json; q=json.load(open(\"$(REMOTE_DIR)/queue.json\")); [j.update({\"status\":\"QUEUED\",\"exit_code\":None,\"started\":None,\"finished\":None}) for j in q]; json.dump(q, open(\"$(REMOTE_DIR)/queue.json\",\"w\"),indent=2)"'
	$(MAKE) deploy
	$(MAKE) start

ci: deploy status

.PHONY: help sync deploy status queue log logs ps gpu kill retry start fresh ci