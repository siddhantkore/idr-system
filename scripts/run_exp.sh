#!/bin/bash
# Usage: ./scripts/run_exp.sh configs/baseline.yaml
CONFIG=${1:-configs/baseline.yaml}
NAME=$(grep '^name:' "$CONFIG" | awk '{print $2}')
STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="experiments/${STAMP}_${NAME}"
mkdir -p "$LOG_DIR"
nohup python3 scripts/train_velocity.py --config "$CONFIG" \
      > "$LOG_DIR/stdout.log" 2>&1 &
echo "Started. PID=$! Log: $LOG_DIR/stdout.log"