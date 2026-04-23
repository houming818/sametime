#!/bin/bash
# 等待 SaTi 评估完成的脚本
# 用法：./wait_for_completion.sh

TIMESLEEP=5
MAXTRIES=10

echo "等待 SaTi 评估完成..."

i=0
while ((i < MAXTRIES)); do
  echo "正在等待结果..."
  ls -l eval/results/*.json 2>/dev/null >/dev/null
  if [[ -f "eval/results/*.json" ]]; then
    echo "✅ 结果已生成"
    break
  fi
  sleep $TIMESLEEP
  i=$((i+1))
done

if [[ $i -eq $MAXTRIES ]]; then
  echo "❌ 评估超时，请检查日志"
  exit 1
fi