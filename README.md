# nccl-rccl-parser
This tool parses RCCL/NCCL debug logs from distributed applications and generates corresponding rccl-tests/nccl-tests commands. It helps identify potential communication bottlenecks when scaling distributed applications using RCCL/NCCL.

## Features
- Automatic platform detection (ROCm/CUDA)
- Parses NCCL/RCCL debug logs and extracts collective operations
- Generates executable test scripts for performance benchmarking
- Supports unique command extraction with occurrence counts
- Generates CSV summary reports with performance metrics

## Getting Started

Clone the repository with submodules:
```
git clone --recursive https://github.com/ROCm/nccl-rccl-parser
```

The following test repositories are included as submodules:
* ROCm: [rccl-tests](https://github.com/ROCmSoftwarePlatform/rccl-tests)
* CUDA: [nccl-tests](https://github.com/NVIDIA/nccl-tests)

## Pre-requisites
* Python 3.x
* RCCL (ROCm) or NCCL (CUDA) installed
* rccl-tests or nccl-tests (included as submodules)

## Usage

### Step 1: Collect RCCL/NCCL Debug Log

Run your distributed application with NCCL debug logging enabled. The application needs to run for at least 1 iteration to capture the collective operations.

**On CUDA:**
```
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL <application/executable> |& tee nccl_debug_log.txt
```
**On ROCm:** (needed for PCIe P2P but not needed for GPUs connected by XGMI, [ref](https://github.com/ROCmSoftwarePlatform/rccl/issues/92#issuecomment-540696989))
```
HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL <application/executable> |& tee nccl_debug_log.txt
```
**NOTE:**
For some workloads buffered output can impact the RCCL/NCCL log format which may break the parser. The following env variables can help with this:
```
PYTHONBUFFERED=x stdbuf -i0 -o0 -e0
```


### Step 2: Parse and Run Tests

#### Option A: Automated Script (Recommended)

The automated script parses the log, runs the tests, and generates a summary report:

**On ROCm:**
```bash
python run_parser_and_generate_summary.py --nccl-debug-log nccl_debug_log.txt --rocm
```

**On CUDA:**
```bash
python run_parser_and_generate_summary.py --nccl-debug-log nccl_debug_log.txt --cuda
```

#### Option B: One-Command Mode

Run collection, parsing, and benchmarking in a single command using `run_and_benchmark.sh`:

**On ROCm:**
```bash
bash run_and_benchmark.sh --run-command "<your_application>" --use-rocm
```

**On CUDA:**
```bash
bash run_and_benchmark.sh --run-command "<your_application>"
```

This captures the logs, parses them, runs the tests, and generates the final CSV report.

#### Option C: Manual Step-by-Step

**1. Parse the debug log:**

```bash
# Generate all commands in execution order
python rccl_nccl_parser.py --nccl-debug-log nccl_debug_log.txt --output-script-name net

# Or generate unique commands with counts
python rccl_nccl_parser.py --nccl-debug-log nccl_debug_log.txt --output-script-name net --unique
```

The parser auto-detects the platform (ROCm/CUDA). To specify explicitly:
```bash
python rccl_nccl_parser.py --nccl-debug-log nccl_debug_log.txt --output-script-name net --platform rocm
python rccl_nccl_parser.py --nccl-debug-log nccl_debug_log.txt --output-script-name net --platform cuda
```

**2. Run the generated test script:**

Copy the script to the tests directory and execute:
```bash
cd rccl-tests  # or nccl-tests
sh net_unique.sh |& tee rccl_perf_data.txt
```

**3. Generate the summary report:**

```bash
python generate_summary.py --log-file rccl_perf_data.txt --output-file-name test_app_data --script-file net_unique.sh --count-file net_counts.csv
```

This generates a CSV file with performance metrics including Time(us), algBw, and busBw for both out-of-place and in-place operations.

## Supported Collectives

The parser supports the following collective operations:

| Collective | Test Command |
|------------|--------------|
| AllReduce | `all_reduce_perf` |
| AllGather | `all_gather_perf` |
| Broadcast | `broadcast_perf` |
| Reduce | `reduce_perf` |
| ReduceScatter | `reduce_scatter_perf` |
| Gather | `gather_perf` |
| Scatter | `scatter_perf` |
| AllToAll | `alltoall_perf` |
| AllToAllv | `alltoallv_perf` |
| Send/Recv | `sendrecv_perf` |
| Hypercube | `hypercube_perf` |

## Supported Data Types

| Type | Description |
|------|-------------|
| int8, uint8 | 8-bit integers |
| int32, uint32 | 32-bit integers |
| int64, uint64 | 64-bit integers |
| half | 16-bit floating point |
| float | 32-bit floating point |
| double | 64-bit floating point |
| bfloat16 | Brain floating point |
| fp8_e4m3 / f8e4m3 | FP8 (ROCm/CUDA) |
| fp8_e5m2 / f8e5m2 | FP8 (ROCm/CUDA) |

## Supported Reduction Operations

- sum
- prod
- max
- min
- avg
- mulsum

## License

MIT License - see [LICENSE](LICENSE) file for details.
