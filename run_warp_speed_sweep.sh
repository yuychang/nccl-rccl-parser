#!/bin/bash

# Script to run ddp_unique.sh with various RCCL_WARP_SPEED settings
# - First run: only RCCL_WARP_SPEED_ENABLE
# - Subsequent runs: RCCL_WARP_SPEED_CU_COUNT from 1 to 304 (powers of 2)

set -e

# Default values
OUTPUT_DIR=""
DEBUG_MODE=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DDP_SCRIPT="${SCRIPT_DIR}/ddp_unique.sh"

# CU counts: powers of 2 from 1 to 256, plus max (304 for gfx942)
CU_COUNTS=(1 2 4 8 16 32 64 128 256 304)

usage() {
    echo "Usage: $0 [-d|--output-dir <directory>] [--debug]"
    echo ""
    echo "Options:"
    echo "  -d, --output-dir <dir>  Directory to save log files (default: current directory)"
    echo "  --debug                 Enable NCCL_DEBUG=INFO for verbose output"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -d ./logs --debug"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Create output directory if specified and doesn't exist
if [[ -n "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_DIR="${OUTPUT_DIR%/}"  # Remove trailing slash if present
else
    OUTPUT_DIR="."
fi

# Check if ddp_unique.sh exists and is executable
if [[ ! -f "$DDP_SCRIPT" ]]; then
    echo "Error: ddp_unique.sh not found at $DDP_SCRIPT"
    exit 1
fi

if [[ ! -x "$DDP_SCRIPT" ]]; then
    chmod +x "$DDP_SCRIPT"
fi

# Common environment variables for mpirun
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export RCCL_UNROLL_FACTOR=0

echo "=============================================="
echo "RCCL Warp Speed Sweep"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo "CU counts to test: ${CU_COUNTS[*]}"
if [[ "$DEBUG_MODE" == true ]]; then
    echo "Debug mode: enabled (NCCL_DEBUG=INFO)"
    export NCCL_DEBUG=INFO
else
    echo "Debug mode: disabled"
fi
echo "=============================================="
echo ""

# Run 1: Only RCCL_WARP_SPEED_ENABLE (no CU count specified)
echo "[Run 1] RCCL_WARP_SPEED_ENABLE=1 (no CU count specified)"
LOG_FILE="${OUTPUT_DIR}/rccl_warp_speed_enable_only.txt"
echo "Logging to: $LOG_FILE"
RCCL_WARP_SPEED_ENABLE=1 "$DDP_SCRIPT" 2>&1 | tee "$LOG_FILE"
echo ""

# Subsequent runs: RCCL_WARP_SPEED_CU_COUNT from 1 to 304
RUN_NUM=2
for CU_COUNT in "${CU_COUNTS[@]}"; do
    echo "[Run $RUN_NUM] RCCL_WARP_SPEED_ENABLE=1 RCCL_WARP_SPEED_CU_COUNT=$CU_COUNT"
    LOG_FILE="${OUTPUT_DIR}/rccl_warp_speed_cu_${CU_COUNT}.txt"
    echo "Logging to: $LOG_FILE"
    RCCL_WARP_SPEED_ENABLE=1 RCCL_WARP_SPEED_CU_COUNT=$CU_COUNT "$DDP_SCRIPT" 2>&1 | tee "$LOG_FILE"
    echo ""
    ((RUN_NUM++))
done

echo "=============================================="
echo "Sweep complete! Logs saved to: $OUTPUT_DIR"
echo "=============================================="
echo ""
echo "Log files:"
ls -la "${OUTPUT_DIR}"/rccl_warp_speed*.txt
