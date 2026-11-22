#!/usr/bin/python3
import os
import re
import signal
import sh
import subprocess
import sys
import time

import config

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

#TODO add warmup
def load_module(module, module_path, args={}):
    print("Unloading nvidia-uvm (if loaded)")
    try:
        sh.sudo("rmmod",  "nvidia-uvm")
    except sh.ErrorReturnCode as e:
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)

        if "is not currently loaded" in err:
            print("nvidia-uvm not loaded, continuing...")
        else:
            print("Unexpected error while removing nvidia-uvm:")
            print(err)
            raise

    arg_list = [f"{k}={v}" for k, v in args.items()]
    print("Loading nvidia-uvm with args:", arg_list)
    try:
        sh.sudo("insmod", module_path, *arg_list)
    except sh.ErrorReturnCode as e:
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
        print(err)
        raise

    return

def install_module(path):
    print(f"Installing kernel module at {path}")
    oldpwd = os.getcwd()
    os.chdir(path)
    #sh.make("modules", "-j")
    #sh.sudo("make", "modules_install", "-j")
    os.chdir(oldpwd)

kernel_arg_dict={"uvm_perf_prefetch_enable": "nopf","uvm_perf_access_counter_batch_count": "", "uvm_perf_access_counter_granularity": "gran", "uvm_perf_access_counter_threshold": "thold", "uvm_perf_access_counter_mimc_migration_enable": "mimc", "uvm_perf_access_counter_momc_migration_enable": "momc", "uvm_perf_prefetch_threshold":"thold", "hpcs_online_mim": "omim", "hpcs_mim_window": "win", "hpcs_mim_min_mb": "minmb", "hpcs_mim_stable": "stable", "hpcs_log_short": "", "hpcs_log_prefetching": "", "hpcs_log_evictions": ""}
def kernel_args_string(kernel_args):
    if len(kernel_args.keys()) == 0:
        return ""
    return "_".join([kernel_arg_dict[key] for key in kernel_args.keys()]) + "_"

class Experiment:

    def __init__(self, benchmark, benchmark_exe, benchmark_dir, psizes, policies,\
                 benchmark_arg_desc, kernel_version, kernel_variant, kernel_args):
        self.benchmark = benchmark
        self.benchmark_exe = benchmark_exe
        self.benchmark_dir = benchmark_dir
        self.psizes = psizes
        self.policies = policies
        self.kernel_version = kernel_version
        self.kernel_variant = kernel_variant
        self.kernel_args = kernel_args
          
    def get_benchmark_arg_desc(self, policy, psize):
        return f"{policy}_{psize}_{self.benchmark}"

    def __str__(self):
        return (f"Benchmark: {self.benchmark}\n"
                f"Benchmark Executable: {self.benchmark_exe}\n"
                f"Benchmark Directory: {self.benchmark_dir}\n"
                f"Psizes: {self.psizes}\n"
                f"Policies: {self.policies}\n"
                f"Kernel Version: {self.kernel_version}\n"
                f"Kernel Variant: {self.kernel_variant}\n"
                f"Kernel Arguments: {self.kernel_args}\n")


    def _run(self, clear, psize, policy, timeout=float(os.environ['TIMEOUT'])): 
            logdir_base = f"{self.kernel_version}_{self.kernel_variant}_{kernel_args_string(self.kernel_args)}{self.get_benchmark_arg_desc(psize, policy)}"
            logdir = f"{self.benchmark_dir}/log_{logdir_base}"
            print("Running experiment:", logdir_base)

            install_module(f"{config.DRIVER_DIR}/{self.kernel_version}/{self.kernel_variant}/kernel/")
            module_path = f"{config.DRIVER_DIR}/{self.kernel_version}/{self.kernel_variant}/kernel/nvidia-uvm.ko"
            print(module_path)
            
            attempts = 0
            while attempts < 3:
                try:
                    load_module("nvidia_uvm", module_path, self.kernel_args)
                    break
                except Exception as e:
                    print(f"Attempt {attempts+1} failed:", e)
                    time.sleep((attempts + 1) *30)
                    attempts += 1
            else:
                sys.exit(1)


            oldpwd = os.getcwd()
            os.chdir(self.benchmark_dir)

            #TODO fix this for benchmarks that don't use make explicitly
            print("Build benchmark")
            sh.make("-j")
            print("Starting execution")
            psize_gb = int(int(psize) * config.VRAM_SIZE / 100)
            print(f"taskset 0xFFFFFFFF ./{self.benchmark_exe} {psize_gb} -p {policy}")

            print(f"Setting environment variable TIMEOUT={timeout}")
            env = os.environ.copy()
            env['TIMEOUT'] = repr(timeout)
            start_time = time.monotonic()
            p = subprocess.Popen(["taskset", "0xFFFFFFFF", f'./{self.benchmark_exe}', str(psize_gb), '-p', policy], stdout=subprocess.PIPE, preexec_fn=os.setsid, env=env)
            output, err = p.communicate()
            print("Waiting on process to die")
            p.wait()
            execution_time = time.monotonic() - start_time
            print(f"Execution time: {execution_time} seconds")



            print("out:", output)
            eprint("err:", err)

            os.chdir(oldpwd)
            print("Finished", logdir_base)
            return execution_time

    def run(self, clear=False, timeout=1000000000):
        for psize in self.psizes:
            print(f"Starting Experiment Set {self.benchmark} {psize}")
            local_policies = self.policies.copy()
            # find baseline policy
            target = 'm' * len(local_policies[0])
            for policy in local_policies:
                if policy == target:
                    mm_policy = policy
                    local_policies.remove(policy)
                    break
            # run and time baseline first; set timeout for following policies based on this.
            baseline = self._run(clear, psize, policy)
            print(f"Experiment set {self.benchmark} {psize}: Measured baseline execution {target} at {baseline} seconds. Using {2 * baseline} as timeout for remaining experiments.")
            for policy in local_policies:
                self._run(clear, psize, policy, min(2 * baseline, int(timeout)))


#print("building warmup app")
#os.chdir(config.WARMUP_DIR)
#sh.make("-j")
#os.chdir(oldpwd)

if __name__ == "__main__":
    main()
