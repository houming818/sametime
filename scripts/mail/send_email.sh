# ================================================
# = 文件信息
# ================================================
# 责任人：待补充
# 编写者：自动治理机器人（opencode）
# 创建时间：2026-04-23
# 更新时间：2026-04-23
# 文件功能：邮件通知，自动报警（事半功倍，妙手生花）
# 所属项目：SameTime
# 原则继承：本文件完全遵循 cabins/AGENTS.txt 及 log/AGENTS.txt 的 principles 原则，遇缺必补，继往开来。

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
