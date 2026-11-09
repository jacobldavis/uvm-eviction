ROOT = "/uvm-eviction"

VRAM_SIZE = 12

BENCHDIR = f"{ROOT}/benchmarks"
SCRIPTSDIR = f"{BENCHDIR}/scriptsv3"
DRIVER_DIR=f"{ROOT}/drivers"
SYSLOG_PATH=f"{ROOT}/tools/sysloggerv2"
SYSLOG_EXE=f"log"
WARMUP_DIR=f"{ROOT}/benchmarks/scriptsv3"
WARMUP_EXE="warmup"
WARMUP_ENABLED=False
# log kernel boundaries with tools/kernel_marker (build it first; serializes launches)
KERNEL_MARKERS = False
KERNEL_MARKER_LIB = f"{ROOT}/tools/kernel_marker/libkernel_marker.so"

KERNEL_VERSION = "x86_64-555.42.02"
KERNEL_VARIANT = "faults-new"
KERNEL_LICENSE = "kernel" # or kernel-open
KERNEL_ARGS = {"hpcs_log_short": 0, "hpcs_log_prefetching" : 0, "hpcs_log_evictions" : 1}#,"uvm_perf_access_counter_momc_migration_enable":1,"uvm_perf_access_counter_granularity":"2m","uvm_perf_prefetch_enable":0, "uvm_perf_access_counter_threshold":"1"}
DO_NOPF = False

PSIZES = ["75", "100", "125", "150", "175"] #Percentage of VRAM_SIZE
BENCHMARKS = ['bfs-worst', 'sgemm', 'FDTD-2D', 'stream', 'GRAMSCHM',  'cg', 'tealeaf']

PSIZES_SUBSET = ["100", "125"] #Percentage of VRAM_SIZE
BENCHMARKS_SUBSET = ['bfs-worst', 'sgemm', 'FDTD-2D', 'stream']
