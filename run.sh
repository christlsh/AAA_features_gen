#!/bin/bash

# Advanced parallel processing script with resume support and monitoring

# Configuration
START_DATE="2020-01-01"
END_DATE="2025-08-15"
NUM_CORES=200
PYTHON_SCRIPT="/home/sihang/gyqk/AAA_features_gen/ob_l2_agg.py"  # 替换为你的Python脚本名称
OB_ORIG_PATH="/data/sihang/l2_ob_full_universe_with_info/"
SAVE_PATH="/data/sihang/l2_ob_agg_5min/"
LOG_DIR="./logs"
STATE_DIR="./state"
PARALLEL_JOBS=2  # 留一些核心给系统使用
BATCH_SIZE=50     # 每批处理的日期数，用于更好的进度跟踪

# Create directories
mkdir -p "$LOG_DIR" "$STATE_DIR"

# State files
ALL_DATES_FILE="${STATE_DIR}/all_dates.txt"
COMPLETED_DATES_FILE="${STATE_DIR}/completed_dates.txt"
FAILED_DATES_FILE="${STATE_DIR}/failed_dates.txt"
PROCESSING_STATUS_FILE="${STATE_DIR}/processing_status.txt"

# Function to get all trading dates
get_trading_dates() {
    python3 - <<EOF
from qdata.core.dt import get_td
import datetime as dt

start = "$1"
end = "$2"
tdates = get_td(start, end).to_numpy().flatten().tolist()
for date in tdates:
    print(date)
EOF
}

# Function to check if a date has already been processed
is_date_processed() {
    local date=$1
    # Check if the output file exists
    python3 - <<EOF
import os
import datetime as dt
from pathlib import Path

date_str = "$date"
save_path = "$SAVE_PATH"
date_obj = dt.datetime.strptime(date_str, "%Y-%m-%d").date()

# Construct the expected output path
output_path = Path(save_path) / f"date={date_obj}"
if output_path.exists() and any(output_path.iterdir()):
    print("1")
else:
    print("0")
EOF
}

# Function to process a single date with retry
process_date_with_retry() {
    local date=$1
    local max_retries=3
    local retry_count=0
    local log_file="${LOG_DIR}/process_${date}.log"
    
    # Check if already processed
    if [ "$(is_date_processed "$date")" = "1" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIPPED: $date (already processed)" >> "$log_file"
        echo "$date" >> "$COMPLETED_DATES_FILE"
        echo "SKIPPED: $date (already exists)"
        return 0
    fi
    
    while [ $retry_count -lt $max_retries ]; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing $date (attempt $((retry_count+1))/$max_retries)" >> "$log_file"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Python script output:" >> "$log_file"
        echo "----------------------------------------" >> "$log_file"
        
        # Use unbuffered Python output to ensure real-time logging
        timeout 1800 python3 -u "$PYTHON_SCRIPT" \
            -sdate "$date" \
            -edate "$date" \
            -ob_orig_path "$OB_ORIG_PATH" \
            -save_path "$SAVE_PATH" \
            >> "$log_file" 2>&1
        
        local exit_code=$?
        echo "----------------------------------------" >> "$log_file"
        
        if [ $exit_code -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $date" >> "$log_file"
            echo "$date" >> "$COMPLETED_DATES_FILE"
            echo "SUCCESS: $date"
            return 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] RETRY: $date (attempt $retry_count failed)" >> "$log_file"
                sleep 5
            fi
        fi
    done
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $date (after $max_retries attempts)" >> "$log_file"
    echo "$date" >> "$FAILED_DATES_FILE"
    echo "FAILED: $date"
    return 1
}

# Function to update progress
update_progress() {
    local completed=$(wc -l < "$COMPLETED_DATES_FILE" 2>/dev/null || echo 0)
    local total=$(wc -l < "$ALL_DATES_FILE" 2>/dev/null || echo 0)
    local failed=$(wc -l < "$FAILED_DATES_FILE" 2>/dev/null || echo 0)
    local percentage=$((completed * 100 / total))
    
    echo "Progress: $completed/$total ($percentage%) | Failed: $failed" > "$PROCESSING_STATUS_FILE"
    echo "Progress: $completed/$total ($percentage%) | Failed: $failed"
}

# Function to monitor progress
monitor_progress() {
    while true; do
        update_progress
        sleep 10
    done
}

# Export functions and variables
export -f process_date_with_retry is_date_processed update_progress
export PYTHON_SCRIPT OB_ORIG_PATH SAVE_PATH LOG_DIR STATE_DIR
export COMPLETED_DATES_FILE FAILED_DATES_FILE ALL_DATES_FILE

# Main execution
echo "==================== Order Book Parallel Processing ===================="
echo "Date Range: $START_DATE to $END_DATE"
echo "Parallel Jobs: $PARALLEL_JOBS"
echo "Python Script: $PYTHON_SCRIPT"
echo "Source Path: $OB_ORIG_PATH"
echo "Save Path: $SAVE_PATH"
echo "======================================================================="

# Check if this is a resume operation
if [ -f "$ALL_DATES_FILE" ] && [ -f "$COMPLETED_DATES_FILE" ]; then
    echo ""
    echo "Found previous processing state. Resuming..."
    TOTAL_DATES=$(wc -l < "$ALL_DATES_FILE")
    COMPLETED_DATES=$(wc -l < "$COMPLETED_DATES_FILE" 2>/dev/null || echo 0)
    echo "Previously completed: $COMPLETED_DATES/$TOTAL_DATES"
    
    # Get remaining dates
    REMAINING_DATES=$(comm -23 <(sort "$ALL_DATES_FILE") <(sort "$COMPLETED_DATES_FILE" 2>/dev/null || echo ""))
    REMAINING_COUNT=$(echo "$REMAINING_DATES" | grep -v '^$' | wc -l)
    echo "Remaining dates to process: $REMAINING_COUNT"
else
    echo ""
    echo "Starting fresh processing..."
    
    # Get all trading dates
    echo "Fetching trading dates..."
    get_trading_dates "$START_DATE" "$END_DATE" > "$ALL_DATES_FILE"
    
    # Initialize state files
    > "$COMPLETED_DATES_FILE"
    > "$FAILED_DATES_FILE"
    
    TOTAL_DATES=$(wc -l < "$ALL_DATES_FILE")
    REMAINING_DATES=$(cat "$ALL_DATES_FILE")
    REMAINING_COUNT=$TOTAL_DATES
    
    echo "Total trading dates to process: $TOTAL_DATES"
fi

# Start progress monitor in background
monitor_progress &
MONITOR_PID=$!

# Process dates in parallel
echo ""
echo "Starting parallel processing..."
echo "$REMAINING_DATES" | parallel -j $PARALLEL_JOBS \
    --timeout 2000 \
    --retries 0 \
    --progress \
    --eta \
    --joblog "${LOG_DIR}/parallel_joblog.txt" \
    process_date_with_retry {}

# Stop progress monitor
kill $MONITOR_PID 2>/dev/null

# Final summary
echo ""
echo "==================== Processing Summary ===================="
COMPLETED_COUNT=$(wc -l < "$COMPLETED_DATES_FILE" 2>/dev/null || echo 0)
FAILED_COUNT=$(wc -l < "$FAILED_DATES_FILE" 2>/dev/null || echo 0)

echo "Total dates: $TOTAL_DATES"
echo "Completed: $COMPLETED_COUNT"
echo "Failed: $FAILED_COUNT"
echo "Success rate: $((COMPLETED_COUNT * 100 / TOTAL_DATES))%"

# Show failed dates if any
if [ $FAILED_COUNT -gt 0 ]; then
    echo ""
    echo "Failed dates (saved in $FAILED_DATES_FILE):"
    head -20 "$FAILED_DATES_FILE"
    if [ $FAILED_COUNT -gt 20 ]; then
        echo "... and $((FAILED_COUNT - 20)) more"
    fi
fi

# Generate detailed report
REPORT_FILE="${LOG_DIR}/report_$(date '+%Y%m%d_%H%M%S').txt"
cat > "$REPORT_FILE" <<EOF
Order Book Processing Report
============================
Generated: $(date)

Configuration:
- Date Range: $START_DATE to $END_DATE
- Total Trading Dates: $TOTAL_DATES
- Parallel Jobs: $PARALLEL_JOBS
- Python Script: $PYTHON_SCRIPT

Results:
- Completed: $COMPLETED_COUNT
- Failed: $FAILED_COUNT
- Success Rate: $((COMPLETED_COUNT * 100 / TOTAL_DATES))%

Processing Time:
- Start: $(stat -c %y "$ALL_DATES_FILE" 2>/dev/null || echo "Unknown")
- End: $(date)

Failed Dates:
$(cat "$FAILED_DATES_FILE" 2>/dev/null || echo "None")
EOF

echo ""
echo "Detailed report saved to: $REPORT_FILE"
echo "============================================================"

# Cleanup option
echo ""
read -p "Do you want to retry failed dates? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "$FAILED_DATES_FILE" ] && [ -s "$FAILED_DATES_FILE" ]; then
        echo "Retrying failed dates..."
        cp "$FAILED_DATES_FILE" "${FAILED_DATES_FILE}.bak"
        > "$FAILED_DATES_FILE"
        
        cat "${FAILED_DATES_FILE}.bak" | parallel -j $((PARALLEL_JOBS / 2)) \
            --timeout 3000 \
            --progress \
            process_date_with_retry {}
        
        echo "Retry completed."
    fi
fi