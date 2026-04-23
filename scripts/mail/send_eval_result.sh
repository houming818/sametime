#!/bin/bash
# SaTi 结果通知脚本
# 用法：./send_eval_result.sh <benchmark_name> <score>

BENCHMARK=${1:-mmlu_pro}
SCORE=${2:-0.0}
TIMESTAMP=$(date +%Y-%m-%dT%H:%M:%S)
SUBJECT="[SaTi] ${BENCHMARK} Evaluation Result - ${TIMESTAMP}"

# 编译邮件内容
echo "
========================================
SaTi 模型评估结果报告

评估基准：${BENCHMARK}
评估时间：${TIMESTAMP}
模型得分：${SCORE}

状态：✅ 评估完成

下一步:
1. 回滚分数低于预期的实验
2. 提交 PR 并触发 CI/CD 流水线
3. 更新 AGENTS.txt 并归档

----------------------------------------" | \
  mailx -s "${SUBJECT}" \
    -r 'evaluator' \
    -R 'niograpcode@gmail.com' \
    -C '/mail' /dev/stdin

echo "✅ 结果邮件已发送"
