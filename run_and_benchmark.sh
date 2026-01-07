#!/bin/bash

# Argument parsing
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --run-command) RUN_COMMAND="$2"; shift ;;
        --use-rocm) USE_ROCM=1 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure RUN_COMMAND is set
if [ -z "$RUN_COMMAND" ]; then
    echo "Please provide --run-command argument."
    exit 1
fi

# Build test repository
if [ "$USE_ROCM" == "1" ]; then
    TEST_DIR="rccl-tests"
else
    TEST_DIR="nccl-tests"
fi
make -C ${TEST_DIR}

# Run code and capture debug log
if [ "$USE_ROCM" == "1" ]; then
    PYTHONBUFFERED=x HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL stdbuf -i0 -o0 -e0 $RUN_COMMAND 2>&1 | tee nccl_debug_log.txt
else
    PYTHONBUFFERED=x NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL stdbuf -i0 -o0 -e0 $RUN_COMMAND 2>&1 | tee nccl_debug_log.txt
fi

# Dump test commands
python rccl_nccl_parser.py --nccl-debug-log nccl_debug_log.txt --output-script-name test_commands

# Remove duplicates and count each collective occurence
awk '!a[$0]++' test_commands.sh > test_commands_unique.sh
awk '{a[$0]++} END {for (i in a) if (a[i] > i) print i", "a[i]}' test_commands.sh > test_commands_unique_counts.csv

# Copy test commands to tests folder
cp test_commands_unique.sh ${TEST_DIR}
cd ${TEST_DIR} && sh test_commands_unique.sh |& tee nccl_perf_data.txt && cd ..

# Generate summary
python generate_summary.py --log-file ${TEST_DIR}/nccl_perf_data.txt --output-file-name nccl_summary_data --script-file ${TEST_DIR}/test_commands_unique.sh 
echo "Performance data dumped to nccl-rccl-parser/nccl_summary_data"

sed -i 's/|/,/g' nccl_summary_data.csv
echo "Performance data converted to csv at nccl-rccl-parser/nccl_summary_data.csv"
echo "NOTE: counts for each kernel stored at test_commands_unique_counts.csv"
