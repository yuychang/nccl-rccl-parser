import os
import sys
import argparse

def detect_platform():
    """Detect whether running on ROCm or CUDA platform."""
    # Check for ROCm
    rocm_paths = ["/opt/rocm", os.environ.get("ROCM_PATH", "")]
    for path in rocm_paths:
        if path and os.path.exists(path):
            return "rocm"

    # Check for CUDA
    cuda_paths = ["/usr/local/cuda", os.environ.get("CUDA_HOME", ""), os.environ.get("CUDA_PATH", "")]
    for path in cuda_paths:
        if path and os.path.exists(path):
            return "cuda"

    raise RuntimeError(
        "Could not detect platform. Neither ROCm nor CUDA found. "
        "Please specify --platform explicitly or set ROCM_PATH/CUDA_HOME environment variable."
    )

def get_data_types_map(platform):
    """Get data types map with platform-specific FP8 naming."""
    base_types = {
        "0" : "int8",
        "1" : "uint8",
        "2" : "int32",
        "3" : "uint32",
        "4" : "int64",
        "5" : "uint64",
        "6" : "half",
        "7" : "float",
        "8" : "double",
        "9" : "bfloat16",
    }

    if platform == "cuda":
        base_types["10"] = "f8e4m3"
        base_types["11"] = "f8e5m2"
    else:  # rocm
        base_types["10"] = "fp8_e4m3"
        base_types["11"] = "fp8_e5m2"

    return base_types

coll_op_map = {
            "Broadcast": "broadcast_perf",
            "Reduce": "reduce_perf",
            "AllGather": "all_gather_perf",
            "ReduceScatter": "reduce_scatter_perf",
            "AllReduce": "all_reduce_perf",
            "Gather": "gather_perf",
            "Scatter": "scatter_perf",
            "AllToAll": "alltoall_perf",
            "AllToAllv": "alltoallv_perf",
            "Send": "sendrecv_perf",
            "Recv": "sendrecv_perf",
            "Hypercube": "hypercube_perf",
            "AllReduceBias": "all_reduce_bias_perf",
          }

reduction_op_map = {
                "0" : "sum",
                "1" : "prod",
                "2" : "max",
                "3" : "min",
                "4" : "avg",
                "5" : "mulsum",
               }

# Platform-specific data types map is initialized in main()

data_type_bytes_map = {
                    "0" : 1,
                    "1" : 1,
                    "2" : 4,
                    "3" : 4,
                    "4" : 8,
                    "5" : 8,
                    "6" : 2,
                    "7" : 4,
                    "8" : 8,
                    "9" : 2,
                    "10" : 1,
                    "11" : 1,
                  }
                
def get_useful_info(log_file):
    fs = open(log_file, 'r')
    lines = fs.readlines()
    fs.close()

    useful_lines = []
    for j in range(len(lines)):
        line = lines[j].rstrip()
        if ("opCount" in line and "sendbuff" in line):
            useful_lines.append(line)

    return useful_lines

def get_test_cmd_prefix(platform):
    """Get the test command prefix based on platform."""
    if platform == "cuda":
        return "./nccl-tests/build/"
    else:  # rocm
        return "./rccl-tests/build/"

def parse_nccl_log(nccl_lines, data_types_map, platform):

    commands = []
    test_cmd_prefix = get_test_cmd_prefix(platform)
    for j in range(len(nccl_lines)):
        line = nccl_lines[j]
        split_list = line.split(" ")
        comm = split_list[split_list.index("INFO") + 1].replace(":", "")
        count = split_list[split_list.index("count") + 1]
        datatype = split_list[split_list.index("datatype") + 1]
        op_type = split_list[split_list.index("op") + 1]
        root = split_list[split_list.index("root") + 1]
        nnranks = next(item for item in split_list if 'nranks' in item).split("=")[1].replace("]", "")

        total_bytes = int(count) * data_type_bytes_map[datatype]

        test_cmd = test_cmd_prefix + coll_op_map[comm.replace("mscclFunc", "")] + " -d " + data_types_map[datatype] + \
                       " -b " + str(total_bytes) + " -e " + str(total_bytes) + \
                       " -o " + reduction_op_map[op_type] + " -g " + str(nnranks)
        commands.append((test_cmd, int(nnranks)))

    return commands

def generate_script(commands, output_script):
    filename = output_script + ".sh"
    fs = open(filename, "w")
    for j in range(len(commands)):
        fs.write(commands[j])
        fs.write("\n")
    fs.close()
    print("INFO: Dumped out the commands in a script named: {}".format(filename))

def dump_counts_map(counts_map, output_file):
    filename = output_file + ".csv"
    fs = open(filename, 'w')
    fs.write("sep=|")
    fs.write("\n")
    keys = counts_map.keys()
    for key in keys:
        fs.write(key + "|" + str(counts_map[key]))
        fs.write("\n")
    fs.close()
    print ("INFO: Dumped out the count of each command in a file named: {}".format(filename))

def get_unique_commands(commands_and_nranks):
    unique_values = []
    counts_map = {}
    nranks_map = {}
    for c_and_nr in commands_and_nranks:
        cmd = c_and_nr[0]
        nranks = c_and_nr[1]
        if (cmd not in unique_values):
            counts_map[cmd] = 1
            nranks_map[cmd] = nranks
            unique_values.append(cmd)
        else:
            counts_map[cmd] = counts_map[cmd] + 1
    assert len(counts_map) == len(nranks_map)
    for cmd in counts_map.keys():
        #assert counts_map[cmd] % nranks_map[cmd] == 0
        counts_map[cmd] = int(counts_map[cmd] / nranks_map[cmd])
    return unique_values, counts_map

def main():
    log_file = os.path.abspath(args.nccl_debug_log)

    # Detect or use specified platform
    if args.platform:
        platform = args.platform
    else:
        platform = detect_platform()
    print("INFO: Using platform: {} (FP8 types: {})".format(
        platform,
        "f8e4m3/f8e5m2" if platform == "cuda" else "fp8_e4m3/fp8_e5m2"
    ))

    data_types_map = get_data_types_map(platform)

    nccl_lines = get_useful_info(log_file)
    commands_and_nranks = parse_nccl_log(nccl_lines, data_types_map, platform)

    if (args.unique):
        new_commands, counts_map = get_unique_commands(commands_and_nranks)
        generate_script(new_commands, args.output_script_name + "_unique")
        dump_counts_map(counts_map, args.output_script_name + "_counts")
    else:
        commands = list(zip(*commands_and_nranks))[0]
        generate_script(commands, args.output_script_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nccl-debug-log", type=str, required=True, help="Log from app with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL")
    parser.add_argument("--output-script-name", type=str, required=False, default="net_nccl_rccl", help="Output command script")
    parser.add_argument("--unique", action="store_true", default=False, help="Get only the unique commands.")
    parser.add_argument("--platform", type=str, required=False, choices=["cuda", "rocm"], help="Platform to use (auto-detected if not specified)")

    args = parser.parse_args()
    main()
