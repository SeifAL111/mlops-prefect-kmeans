#!/usr/bin/env bash
set -euo pipefail

cd /home/dell/ml_project
source /home/dell/ml_project/venv/bin/activate

mkdir -p /home/dell/ml_project/logs
echo "=== $(date '+%Y-%m-%d %H:%M:%S') ===" >> /home/dell/ml_project/logs/cron.log
prefect deployment run 'all/kmeans-all-deployment' >> /home/dell/ml_project/logs/cron.log 2>&1
