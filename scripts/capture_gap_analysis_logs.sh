#!/bin/bash
# Script to run gap analysis and capture both console and file logs
# Usage: ./scripts/capture_gap_analysis_logs.sh <config_file>

set -e  # Exit on error

if [ -z "$1" ]; then
    echo "Usage: $0 <config_file>"
    echo "Example: $0 wvs_values_gap_analysis/gap_analysis_test_config.yaml"
    exit 1
fi

CONFIG_FILE="$1"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Extract output file path from config to create console log file
OUTPUT_FILE=$(grep "^output_file_path:" "$CONFIG_FILE" | cut -d'"' -f2)
CONSOLE_LOG_FILE="${OUTPUT_FILE%.jsonl}_console.log"

echo "========================================"
echo "Gap Analysis with Log Capture"
echo "========================================"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_FILE"
echo "Console Log: $CONSOLE_LOG_FILE"
echo "File Log: ${OUTPUT_FILE%.jsonl}.log"
echo "========================================"
echo ""

# Run gap analysis with tee to capture console output
python llm_behavior_adaptation/value_measurement/wvs_gap_analysis.py \
    --config "$CONFIG_FILE" 2>&1 | tee "$CONSOLE_LOG_FILE"

echo ""
echo "========================================"
echo "Logs saved to:"
echo "  - Console log: $CONSOLE_LOG_FILE"
echo "  - File log: ${OUTPUT_FILE%.jsonl}.log"
echo "========================================"
