#!/usr/bin/env python3
# Per-kernel optimum vs whole-run optimum vs HELM+MIM labels, from perkernel.py's runs
# usage: python3 perkernel_compare.py kernel_labels_perkernel.csv [--apps ...] [--psizes ...] [--host cuda-host]
import argparse
import glob
import os
import re
import socket

import numpy as np
import pandas as pd

import config
from kernel_labels import metric_file

# kernel name prefix -> kernel index in launch order
KERNELS = {
    '2MM': {'mm2_kernel1': 0, 'mm2_kernel2': 1},
    'GRAMSCHM': {'gramschmidt_kernel1': 0, 'gramschmidt_kernel2': 1, 'gramschmidt_kernel3': 2},
    'sgemm': {'': 0},   # single cuBLAS kernel (warmup is skipped)
    # phase 0: setup, phase 1: timestep loop; zero_buffer runs before any policy is set
    'tealeaf': {'zero_buffer': None, 'set_chunk': 0, '': 1},
}
RECORD = np.dtype([('faddr', 'u8'), ('timestamp', 'u8'), ('num_instances', 'u2'), ('pad', 'u1', (11,)),
                   ('record_type', 'u1'), ('_padding', 'u2')])


def allocation_order(app, psize):
    # managed allocation base addresses, in allocation order
    path, = glob.glob(f'../../benchmarks/default/{app}/log_*_{psize}_{app}/{app}_klog')
    with open(path, 'rb') as file:
        mode, hostname_len = file.read(2)
        file.read(hostname_len)
        if mode != 0:
            raise ValueError(f"{path}: short-mode logs are not supported here")
        records = np.fromfile(file, dtype=RECORD)
    return list(records['faddr'][records['record_type'] == 3])


def to_policy(app, psize, label, n_allocs, host):
    # label order (by address) -> allocation order, untouched '.' -> 'm'
    metrics = pd.read_csv(metric_file(f'../../figs/{host}', 'metrics_stats_relative', app, psize))
    metrics = metrics.sort_values('label').reset_index(drop=True)
    # drops the CUDA runtime's own 2 MiB managed range
    order = [base for base in allocation_order(app, psize) if base in set(metrics['base_address'])]
    if len(order) != n_allocs:
        raise ValueError(f"{app} {psize}%: {len(order)} labeled allocations in the log, expected {n_allocs}")
    policy = ['m'] * n_allocs
    for letter, base in zip(label, metrics['base_address']):
        policy[order.index(base)] = 'm' if letter == '.' else letter
    return ''.join(policy)


def kernel_index(app, name):
    if app == 'sgemm':
        return None if name.startswith('warmup') else 0
    for prefix, index in KERNELS[app].items():
        if name.startswith(prefix):
            return index
    return None


def mim_policies(labels, app, psize, n_allocs, n_kernels, host):
    rows = labels[(labels['app'] == app) & (labels['psize'] == int(psize))]
    if rows.empty:
        return None, None
    app_policy = to_policy(app, psize, rows['app_label'].iloc[0], n_allocs, host)
    # kernels sharing a phase: the kernel with the most faults wins each allocation
    merged = [None] * n_kernels
    for _, row in rows.sort_values('kernel_faults').iterrows():
        index = kernel_index(app, row['kernel_name'].replace('void ', ''))
        if index is None or row['kernel'] == 'between':
            continue
        base = merged[index] or '.' * len(row['kernel_label'])
        merged[index] = ''.join(new if new != '.' else old for old, new in zip(base, row['kernel_label']))
    per_kernel = [to_policy(app, psize, label, n_allocs, host) if label else 'm' * n_allocs for label in merged]
    return app_policy, ':'.join(per_kernel) + (':' if n_kernels == 1 else '')


def load_times(app):
    # reads the .data directly: perf_data2csv.py records a timeout as the limit time
    times = {}
    # per-kernel runs from the old file predate the 'd' fix
    for tag, keep_per_kernel in [('perkernel', False), ('perkernel2', True)]:
        path = f'../../benchmarks/strategied/{app}/{app.lower()}_{tag}_numa_pref.data'
        if os.path.exists(path):
            load_blocks(open(path).read(), keep_per_kernel, times)
    return times


def load_blocks(text, keep_per_kernel, times):
    for block in re.split(r'^(?=\./)', text, flags=re.M)[1:]:
        words = block.split('\n', 1)[0].split()
        if '-p' not in words or not re.search(r'timed out|did not time out', block):
            continue
        if not keep_per_kernel and ':' in words[words.index('-p') + 1]:
            continue
        runtime = re.search(r'GPU Runtime: ([\d.]+)s', block)
        limit = re.search(r'within ([\d.]+) seconds', block)
        finished = runtime and not (limit and float(runtime.group(1)) >= 0.95 * float(limit.group(1)))
        psize = round(100 * int(words[1]) / config.VRAM_SIZE)
        times.setdefault((psize, words[words.index('-p') + 1]), []).append(float(runtime.group(1)) if finished else None)


def summarize(runs):
    # median of finished runs, None if all timed out
    finished = [t for t in runs if t is not None]
    return float(np.median(finished)) if finished else None


def main():
    parser = argparse.ArgumentParser(description="Compare per-kernel optimum, whole-run optimum and HELM+MIM")
    parser.add_argument('labels_csv', help="kernel_labels.py output for these apps")
    parser.add_argument('--apps', nargs='+', default=['2MM', 'GRAMSCHM', 'sgemm'])
    parser.add_argument('--psizes', nargs='+', default=['100', '125'])
    parser.add_argument('--host', default=socket.gethostname())
    parser.add_argument('-o', '--output', default='perkernel_compare')
    args = parser.parse_args()

    labels = pd.read_csv(args.labels_csv)
    rows, missing = [], []
    for app in args.apps:
        all_times = load_times(app)
        if not all_times:
            print(f"skipping {app}: no perkernel .data files (run perkernel.py)")
            continue
        n_kernels = len({i for i in KERNELS[app].values() if i is not None})
        control = app == 'sgemm'
        for psize in args.psizes:
            runs = {p: r for (ps, p), r in all_times.items() if ps == int(psize)}
            if not runs:
                print(f"skipping {app} {psize}%: no runs")
                continue
            t = {p: summarize(r) for p, r in runs.items()}
            n_allocs = len(next(p for p in t if ':' not in p))
            whole = {p: s for p, s in t.items() if ':' not in p and s is not None}
            per_kernel = {p: s for p, s in t.items() if ':' in p and s is not None}
            mim_app, mim_kernel = mim_policies(labels, app, psize, n_allocs, n_kernels, args.host)
            app_opt = min(whole, key=whole.get)

            def timing(policy):
                if policy not in t:
                    missing.append((app, psize, policy))
                    return 'not run'
                return round(t[policy], 1) if t[policy] is not None else 'timeout'

            row = {'app': app, 'psize': psize, 'runs': sum(len(r) for r in runs.values()),
                   'baseline_s': round(t['m' * n_allocs], 1), 'app_opt': app_opt, 'app_opt_s': round(whole[app_opt], 1),
                   'mim_app': mim_app, 'mim_app_s': timing(mim_app) if mim_app else None}
            if control:
                # control: "xyz:" should time the same as "xyz"
                diffs = [abs(t[p] - t[p + ':']) / t[p] for p in whole if t.get(p + ':') is not None]
                row['control_pairs'] = len(diffs)
                row['control_max_rel_diff'] = round(max(diffs), 3) if diffs else None
            elif per_kernel:
                kernel_opt = min(per_kernel, key=per_kernel.get)
                row.update({'kernel_opt': kernel_opt, 'kernel_opt_s': round(per_kernel[kernel_opt], 1),
                            'kernel_opt_speedup': round(whole[app_opt] / per_kernel[kernel_opt], 3)})
            else:
                row['kernel_opt'] = 'no per-kernel runs'
            if not control and mim_kernel:
                row.update({'mim_kernel': mim_kernel, 'mim_kernel_s': timing(mim_kernel)})
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f'{args.output}.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 50)
    print(df.to_string(index=False))
    for app, psize, policy in missing:
        print(f"untimed MIM policy {app} {psize}%: cd benchmarks/scriptsv3 && "
              f"python3 -u perkernel.py --apps {app} --psizes {psize} --policies {policy}")
    print(f"\nwrote {args.output}.csv")


if __name__ == "__main__":
    main()
