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

def main():
    debug_log = os.path.abspath(args.nccl_debug_log)

    # Determine platform
    if args.platform:
        platform = args.platform
    else:
        platform = detect_platform()
        print(f"INFO: Auto-detected platform: {platform}")

    ##### Firstly call rccl_nccl_parser.py to parse the ......log.txt file
    ## Generate a script to run nccl/rccl tests.
    gen_cmd = f"python rccl_nccl_parser.py --nccl-debug-log {debug_log} --output-script-name net --unique --platform {platform}"
    if os.system(gen_cmd):
        print ("ERROR: Failed to parse the log.")
        sys.exit(1)

    ## change directory to rccl-tests/nccl-tests
    if platform == "rocm":
        rccl_tests_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "rccl-tests")
        os.system("cp net_unique.sh " + rccl_tests_path)
        os.chdir(rccl_tests_path)
        if os.system("./install.sh --rccl_home=/opt/rocm  2>&1"):
            print("ERROR: Failed to install rccl-tests.")
            sys.exit(1)

        os.system("cat net_unique.sh")
        run_script_cmd = "HSA_FORCE_FINE_GRAIN_PCIE=1 sh net_unique.sh | tee rccl_perf_log.txt"
        if os.system(run_script_cmd):
            print ("ERROR: Unable to run rccl-tests properly.")
            sys.exit(1)
        os.system("mv rccl_perf_log.txt ../")
        os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__))))

        print (os.getcwd())
        summary_cmd = "python generate_summary.py --log-file rccl_perf_log.txt --script-file net_unique.sh --count-file net_counts.csv"
        os.system(summary_cmd)
        print ("INFO: Finished dumping all data.")

    elif platform == "cuda":
        nccl_tests_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nccl-tests")
        os.system("cp net_unique.sh " + nccl_tests_path)
        os.chdir(nccl_tests_path)
        if os.system("make > /dev/null 2>&1"):
            print ("ERROR: Failed to install nccl-unit tests")
            sys.exit(1)

        os.system("cat net_unique.sh")
        run_script_cmd = "sh net_unique.sh | tee nccl_perf_log.txt"
        if os.system(run_script_cmd):
            print ("ERROR: unable to run nccl-tests")
            sys.exit(1)
        os.system("mv nccl_perf_log.txt ../")
        os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__))))

        summary_cmd = "python generate_summary.py --log-file nccl_perf_log.txt --script-file net_unique.sh --output-file-name nv_net_summary --count-file net_counts.csv"
        os.system(summary_cmd)
        print ("INFO: Finished dumping all data.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parse NCCL/RCCL debug log, replay collective operations, and generate performance summary."
    )
    parser.add_argument("--nccl-debug-log", type=str, required=True,
                        help="NCCL/RCCL log after running app with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL")
    parser.add_argument("--platform", type=str, required=False, choices=["cuda", "rocm"],
                        help="Platform to use (auto-detected if not specified)")

    args = parser.parse_args()
    main()
