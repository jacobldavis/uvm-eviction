import argparse
import os
import struct
import numpy as np
import pandas as pd

MAX_FILE_SIZE = 15 * (1024**3)

class Experiment:
    
    def __init__(self, file_name, df_faults, df_prefetch, df_evictions, df_address_ranges, output_state):
        self.file_name = file_name
        self.df_faults = df_faults
        self.df_prefetch = df_prefetch
        self.df_evictions = df_evictions
        self.df_address_ranges = df_address_ranges
        self.output_state = output_state
        self.labels = get_legend_labels(output_state.benchmark)

    def __str__(self):
        return f"{self.output_state.strategy}_{self.output_state.dir_name.replace('log_', '')}"

class OutputState:
    def __init__(self, strategy, benchmark, dir_name, hostname):
        self.strategy = strategy
        self.benchmark = benchmark
        self.dir_name = dir_name
        self.hostname = hostname
        # TODO this is very fragile to naming convention changes
        self.psize = int(dir_name.split("_")[-2])

    def get_output_path(self, plot_type):
        extension='.png'
        if "stats" in plot_type:
            extension='.txt'

        file_name = f"{self.strategy}_{self.dir_name.replace('log_', '')}{extension}"
        output_dir = f"../../figs/{self.hostname}/{plot_type}/{self.benchmark}/"
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, file_name)

def get_output_state_from_path_hostname(path, hostname):
    path_parts = path.split('/')
    benchmark_index = path_parts.index('benchmarks') + 1
    strategy = path_parts[benchmark_index]
    benchmark = path_parts[benchmark_index + 1]
    dir_name = path_parts[benchmark_index + 2]
    output_state = OutputState(strategy, benchmark, dir_name, hostname)
    return output_state


def get_output_state_from_path(path):
    hostname = extract_hostname(path)
    return get_output_state_from_path_hostname(path, hostname)

def get_legend_labels(name):
    name = name.upper()
    with open("legend_labels.conf") as file:
        legend_label_lines = file.readlines()
    
    for i, line in enumerate(legend_label_lines):
        if name in line:
            labels = legend_label_lines[i+1].split(',')
            labels = [label.strip() for label in labels]
            return labels
    print(f"Name: {name} not found!")


def extract_hostname(path):
    with open(path, 'rb') as file:
        mode = ord(file.read(1))
        hostname_len = ord(file.read(1))
        hostname = file.read(hostname_len).decode('ascii')
    return hostname


TWO_MB = 2 * 1024 * 1024
# record types from drivers/include/hpcs_types.h
HPCS_KERNEL_BEGIN, HPCS_KERNEL_END = 4, 5
KERNEL_BETWEEN = "between"

def read_kernel_names(file_path):
    # map written by the kernel marker shim next to the log: <bench>_klog -> <bench>_kernels.csv
    if not file_path.endswith("_klog"):
        return {}
    map_path = file_path[:-len("_klog")] + "_kernels.csv"
    if not os.path.exists(map_path):
        print(f"No kernel name map at {map_path}; using kernel ids as names")
        return {}
    names = {}
    with open(map_path) as file:
        next(file)
        for line in file:
            # split once; names are mangled symbols but don't trust them to be comma free
            kernel_id, name = line.rstrip("\n").split(",", 1)
            names[int(kernel_id)] = name
    return names

def get_kernel_spans(df_data):
    begins = df_data[df_data['record_type'] == HPCS_KERNEL_BEGIN]
    ends = df_data[df_data['record_type'] == HPCS_KERNEL_END]
    df_kernels = pd.DataFrame({'kernel_id': begins['faddr'].values, 'begin': begins.index.values})
    df_ends = pd.DataFrame({'kernel_id': ends['faddr'].values, 'end': ends.index.values})
    df_kernels = df_kernels.merge(df_ends, on='kernel_id', how='left')
    # a kernel without an end marker (e.g. benchmark killed by timeout) runs to the end of the log
    # (merge leaves 'end' as float, so use a sentinel that survives the round trip)
    df_kernels['end'] = df_kernels['end'].fillna(df_data.index.max() + 1).astype(np.int64)
    return df_kernels.sort_values('begin').reset_index(drop=True)

def add_kernel_columns(df, df_kernels, kernel_names):
    # records are in log order, so a record belongs to the last kernel that began before it,
    # provided that kernel has not ended yet
    positions = df.index.values
    k = np.searchsorted(df_kernels['begin'].values, positions, side='right') - 1
    k_clipped = np.clip(k, 0, None)
    inside = (k >= 0) & (positions < df_kernels['end'].values[k_clipped])
    kernel_ids = np.where(inside, df_kernels['kernel_id'].values[k_clipped], -1).astype(np.int32)
    df['kernel_id'] = kernel_ids
    # categorical name column: launches of the same kernel share a category, so grouping by name
    # aggregates them; slot 0 is records that fall between kernels
    names = [KERNEL_BETWEEN] + [kernel_names.get(kid, f"kernel_{kid}") for kid in df_kernels['kernel_id']]
    categories = list(dict.fromkeys(names))
    name_codes = np.array([categories.index(name) for name in names])
    df['kernel'] = pd.Categorical.from_codes(name_codes[np.where(inside, k_clipped + 1, 0)], categories=categories)


def parse_fault_data(file_path):
    with open(file_path, 'rb') as file:
        mode = ord(file.read(1))
        hostname_len = ord(file.read(1))
        hostname = file.read(hostname_len).decode('ascii')
    
    print(f"Loading data from host {hostname}, using mode {mode} set by magic byte")
    # Define the dtype based on the mode
    if mode == 0:
        dtype = np.dtype([
            ('faddr', 'u8'), ('timestamp', 'u8'), ('num_instances', 'u2'),
            ('fault_type', 'u1'), ('access_type', 'u1'), ('access_type_mask', 'u1'),
            ('client_type', 'u1'), ('mmu_engine_type', 'u1'), ('client_id', 'u1'),
            ('mmu_engine_id', 'u1'), ('utlb_id', 'u1'), ('gpc_id', 'u1'),
            ('channel_id', 'u1'), ('ve_id', 'u1'), ('record_type', 'u1'), ('_padding', 'u2')
        ])
    elif mode == 1:
        dtype = np.dtype([('faddr', 'u8')])

    record_size = dtype.itemsize
    file_size = min(os.path.getsize(file_path), MAX_FILE_SIZE)
    num_records = (file_size - 2 - hostname_len) // record_size
    
    df_data = pd.DataFrame(np.fromfile(file_path, dtype=dtype, count=num_records, offset=2 + hostname_len))
    print("Finished parsing file; preparing data through post-processing")
    
    if mode == 0:
        print("extracting address ranges")
        df_address_ranges = df_data[df_data['record_type'] == 3]#.copy()
        df_address_ranges['base_address'] = df_address_ranges['faddr']
        df_address_ranges['length'] = df_address_ranges['timestamp']
        df_address_ranges = df_address_ranges[['base_address', 'length']]
        print("address ranges:")
        print(df_address_ranges)
        df_data = df_data[df_data['record_type'] != 3]
    
    elif mode == 1:
        print("for mode 1, computing record types from records")
        # Adjust 'faddr' to remove the record type
        df_data['record_type'] = (df_data.faddr.values >> 56).astype('u1')
        df_data['faddr'] &= 0x00FFFFFFFFFFFFFF

        print("extracting address_range indices")

        condition = df_data['record_type'] == 3
        indices = df_data.index[condition]
        print("type(indices):", type(indices))
        base_addresses = df_data.loc[indices, 'faddr']
        lengths = df_data.loc[indices + 1, 'faddr']
        df_data.loc[indices + 1, 'record_type'] = 3

        df_address_ranges = pd.DataFrame({
            'base_address': base_addresses.values,
            'length': lengths.values
        })
        print("Base Addresses:")
        print(base_addresses)
        print("\nLengths:")
        print(lengths)

    print("extracting kernel markers")
    df_kernels = get_kernel_spans(df_data)
    print(f"found {len(df_kernels)} kernel launches")

    print("filtering dataframes")
    # Filter and delete entries for both modes
    df_faults = df_data[df_data['record_type'] == 0]
    df_prefetch = df_data[df_data['record_type'] == 1]
    df_evictions = df_data[df_data['record_type'] == 2]

    # logs without markers keep the old per-application columns only
    if not df_kernels.empty:
        print("assigning records to kernels")
        kernel_names = read_kernel_names(file_path)
        for df in [df_faults, df_prefetch, df_evictions]:
            add_kernel_columns(df, df_kernels, kernel_names)

 
    print("dropping unneeded metadata columns")
    del df_data
    df_faults.drop(columns='record_type', inplace=True)
    df_evictions.drop(columns='record_type', inplace=True)
    df_prefetch.drop(columns='record_type', inplace=True)

    print("computing minimum fault addresses and valid address ranges")
    # Compute min_faddr and adjusted_faddr
    min_faddr = (df_faults['faddr'].min() // TWO_MB) * TWO_MB
    print("min_faddr:", df_faults['faddr'].min() )
    print("min_faddr:", min_faddr)
    print("ranges:", df_address_ranges)
    valid_ranges = [range for idx, range in df_address_ranges.iterrows() if range['base_address'] >= min_faddr]
    print("valid_ranges:", valid_ranges)

    df_valid_ranges = pd.DataFrame(valid_ranges)
    global_min_address = df_valid_ranges['base_address'].min() if not df_valid_ranges.empty else 0

    print("computing adjusted address ranges")
    # Adjusting addresses
    for df in [df_faults, df_prefetch, df_evictions]:
        if not df.empty:
            df['faddr'] = df['faddr'] - global_min_address
        df.rename(columns={'faddr': 'adjusted_faddr'}, inplace=True)

    df_valid_ranges['adjusted_base'] = df_valid_ranges['base_address'] - global_min_address
    df_valid_ranges['adjusted_end'] = df_valid_ranges['adjusted_base'] + df_valid_ranges['length']

    print("data loading and pre-processing complete")
    return hostname, df_faults, df_prefetch, df_evictions, df_valid_ranges


def main():
    parser = argparse.ArgumentParser(description="Process binary data into DataFrame and plot.")
    parser.add_argument("file_path", type=str, help="Path to the binary file")
    parser.add_argument("output_dir", type=str, nargs='?', help="Extra directory to store output", default="faults_relative")
    args = parser.parse_args()
    output_state = get_output_state_from_path(args.file_path)
    print(output_state.get_output_path(args.output_dir)) 
    

if __name__ == "__main__":
    main()
