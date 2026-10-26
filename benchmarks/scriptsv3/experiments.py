#!/usr/bin/python3
import os
import re
import sh
import subprocess
import sys
import time

import config

from copy import deepcopy

slurm_output_directory = f'{config.BENCHDIR}/{config.SCRIPTSDIR}/fault_plot_out/'
# Ensure the output directory exists
os.makedirs(slurm_output_directory, exist_ok=True)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def load_module(module, args={}):
    # load and unload module first to make sure its unloaded w/o error
    for i in range(3):
        try:
            sh.sudo("rmmod", "-f", "nvidia-uvm")
            break
        except Exception as e:
            print(e)
    print(f"insmod {module}", args)
    arg_strs = [f"{key}={value}" for key, value in args.items()] 
    sh.sudo("insmod", module, *arg_strs)
    if config.WARMUP_ENABLED:
        p = subprocess.Popen([f'{config.WARMUP_DIR}/{config.WARMUP_EXE}'], stdout=subprocess.PIPE)
        # insert warmup error logging?
        p.wait()

def install_module(path):
    print(f"Installing kernel module at {path}")
    oldpwd = os.getcwd()
    os.chdir(path)
    sh.make("modules", "-j")
    #sh.sudo("make", "modules_install", "-j")
    os.chdir(oldpwd)

def get_device_number(device_name):
    with open("/proc/devices") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2 and parts[1] == device_name:
                return int(parts[0])
    return None

def init_syslogger(logfile):
    # this is required for voltron because it doesn't support kernel-open and udev has to create this file, 
    # which has a slight asynchronous delay
    dev_num = get_device_number("hpcs_logger")
    counter = 0
    while not os.path.exists("/dev/hpcs_logger"):
        counter = counter + 1
        sh.sudo("mknod", "-m", "0666", "/dev/hpcs_logger", "c", str(dev_num), "0")
        if counter > 10:
            print("/dev/hpcs_logger still does not exist after 10 seconds; check dmesg for errors")
            sys.exit(1)
        time.sleep(1)
    oldpwd = os.getcwd()
    os.chdir(config.SYSLOG_PATH)
    sh.make()
    os.chdir(oldpwd)
    process = subprocess.Popen([f"{config.SYSLOG_PATH}/{config.SYSLOG_EXE}", logfile])#, creationflags=subprocess.DETACHED_PROCESS)
    return process

def close_syslogger(process):
    process.kill()
    
kernel_arg_dict={"uvm_perf_prefetch_enable": "nopf","uvm_perf_access_counter_batch_count": "", "uvm_perf_access_counter_granularity": "gran", "uvm_perf_access_counter_threshold": "thold", "uvm_perf_access_counter_mimc_migration_enable": "mimc", "uvm_perf_access_counter_momc_migration_enable": "momc", "uvm_perf_prefetch_threshold":"thold", "hpcs_log_short" : "", "hpcs_log_prefetching":  "", "hpcs_log_evictions": ""}
def kernel_args_string(kernel_args):
    parts = [kernel_arg_dict[key] for key in kernel_args.keys() if kernel_arg_dict[key]]
    if len(parts) == 0:
        return ""
    return "_".join(parts) + "_"

class Experiment:

    def __init__(self, benchmark, benchmark_exe, benchmark_dir, benchmark_args, \
                 kernel_version, kernel_variant, kernel_args):
        self.benchmark = deepcopy(benchmark)
        self.benchmark_exe = deepcopy(benchmark_exe)
        self.benchmark_dir = deepcopy(benchmark_dir)
        self.benchmark_args = deepcopy(benchmark_args)
        self.kernel_version = deepcopy(kernel_version)
        self.kernel_variant = deepcopy(kernel_variant)
        self.kernel_args = deepcopy(kernel_args)

    def get_arg_desc(self, psize):
        return f"{psize}_{self.benchmark}"
    
    def __str__(self):
        return f"{self.kernel_version}_{self.kernel_variant}_{kernel_args_string(self.kernel_args)}{self.benchmark}"

    def run(self, clear=False, txt_only=False):
        klogs = []
        for arg in self.benchmark_args:
            logdir_base = f"{self.kernel_version}_{self.kernel_variant}_{kernel_args_string(self.kernel_args)}{self.get_arg_desc(arg)}"
            logdir = f"{self.benchmark_dir}/log_{logdir_base}"
            klog = f"{logdir}/{self.benchmark}_klog"
            plog = f"{logdir}/{self.benchmark}_perf.txt"

            klogs.append(klog)
            print("Running experiment:", logdir_base)
            if not clear and os.path.exists(klog) and os.path.exists(plog):
                print("Existing experiment log found")
                return
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            if os.path.exists(klog):
                os.remove(klog)
            if os.path.exists(plog):
                os.remove(plog)
            
            module_str=f"{config.DRIVER_DIR}/{self.kernel_version}/{self.kernel_variant}/{config.KERNEL_LICENSE}/"
            install_module(module_str)
            
            #Load module
            attempts = 0
            success = False
            while not success:
                try:
                    load_module(f"{module_str}/nvidia-uvm.ko", self.kernel_args)
                except Exception as e:
                    print(f"Caught an exception: {e}")
                    time.sleep((attempts + 1))
                    attempts += 1

                else:
                   success = True 
                if attempts > 5:
                    print("Failed to load module after 5 tries, exiting")
                    sys.exit(1)


            slog_proc = init_syslogger(klog)
            oldpwd = os.getcwd()
            os.chdir(self.benchmark_dir)

            #TODO fix this for benchmarks that don't use make explicitly
            print("Build benchmark")
            sh.make("-j")
            print("Starting execution")
            #TODO fix this for benchmarks that need more than just their problem size
            psize_gb = int(int(arg) * config.VRAM_SIZE / 100)
            command = ["taskset", "0xFFFFFFFF", f'./{self.benchmark_exe}'] + [str(psize_gb)]
            #command = ["taskset", "-c", "4,5", f'./{self.benchmark_exe}'] + self.benchmark_args
            env = os.environ.copy()
            if config.KERNEL_MARKERS:
                assert os.path.exists(config.KERNEL_MARKER_LIB), f"{config.KERNEL_MARKER_LIB} not built"
                env["CUDA_INJECTION64_PATH"] = config.KERNEL_MARKER_LIB
                env["HPCS_KERNEL_MAP"] = f"{logdir}/{self.benchmark}_kernels.csv"
            p = subprocess.Popen(command, stdout=subprocess.PIPE, env=env)
            print(f"Running command {command}")
            output, err = p.communicate()
            exit_code = p.wait()

            close_syslogger(slog_proc)

            print("out:", output)
            eprint("err:", err)
            eprint("exit code:", exit_code)
            
            os.chdir(oldpwd)
            #self.launch_fault_plotting_job(klog, arg)
        #self.launch_fault_scaling_plotting_job(klogs)

    def launch_fault_plotting_job(self, klog, arg):
        experiment_str = f"{self.kernel_version}_{self.kernel_variant}_{kernel_args_string(self.kernel_args)}{self.get_arg_desc(arg)}"        
        print("Finished, launching plotting job", experiment_str)
        try:
            # Using the sh.sbatch function to submit the job script
            #print(sh.sbatch(script_file, klog))
            sh.sbatch(f"--job-name=fault_plot_{experiment_str}",\
                     f"--output={slurm_output_directory}/{experiment_str}.out",\
                     "--partition=hsw", "--nodes=1", "--exclusive", "--time=08:00:00",\
                     f"--chdir={config.ROOT}/tools/fault_plotsv2/",\
                     f"--wrap=python {config.ROOT}/tools/fault_plotsv2/fault_plot.py {klog}")
        except sh.ErrorReturnCode as e:
            # Handle errors returned from sbatch
            print("Error submitting job:", e)
    
    def launch_fault_scaling_plotting_job(self, klogs):
        experiment_str = f"{self.kernel_version}_{self.kernel_variant}_{'-'.join(self.benchmark_args)}_{self.benchmark}"
        print("Finished, launching scaling plotting job", experiment_str)
        try:
            # Using the sh.sbatch function to submit the job script
            #print(sh.sbatch(script_file, klog))
            sh.sbatch(f"--job-name=fault_scaling_plot_{experiment_str}",\
                     f"--output={slurm_output_directory}/{experiment_str}.out",\
                     "--partition=hsw", "--nodes=1", "--exclusive", "--time=08:00:00",\
                     f"--chdir={config.ROOT}/tools/fault_plotsv2/",\
                     f"--wrap=python {config.ROOT}/tools/fault_plotsv2/fault_scaling_plot.py {' '.join(klogs)}")
        except sh.ErrorReturnCode as e:
            # Handle errors returned from sbatch
            print("Error submitting job:", e)


print("Building logger exe")
oldpwd = os.getcwd()
os.chdir(config.SYSLOG_PATH)
sh.make("-j")
os.chdir(oldpwd)
#print("building warmup app")
#os.chdir(config.WARMUP_DIR)
#sh.make("-j")
#os.chdir(oldpwd)

if __name__ == "__main__":
    main()
