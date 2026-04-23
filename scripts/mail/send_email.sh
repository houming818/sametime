#!/bin/bash
# 邮件通知脚本
# 用法：./send_email.sh <结果目录>

RESULTS_DIR=${1:-./eval/results}
EMAIL_TO="niograpcode@gmail.com"
SUBJECT="Sati 评估报告"

echo "========================================"
echo "Sati 模型评估报告"
echo "评估基准：$(ls $RESULTS_DIR/* 2>/dev/null | xargs basename | head -1)"
echo "评估时间：$(date +%Y-%m-%dT%H:%M:%S)"
echo "模型得分：$(cat $RESULTS_DIR/*.json 2>/dev/null | grep score | cut -d: -f2)"
echo "===-------------------------"

echo "Sati 评估报告已生成"
