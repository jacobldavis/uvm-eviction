#!/usr/bin/env python3
"""
perkernel_compare.py - per-kernel brute-force optimum vs whole-run optimum vs HELM+MIM, per app and psize.

Inputs:
  - timings: benchmarks/strategied/<app>/<app>_perkernel2_numa_pref.data (and whole-run runs from the older
    <app>_perkernel_numa_pref.data), written by benchmarks/scriptsv3/perkernel.py; a policy with ':' is per
    kernel, without is whole-run
  - MIM labels: kernel_labels.py output for the same apps (marked fault logs + both metric passes)

MIM labels list allocations by address (A = highest); policies follow allocation order. The fault log's
address-range records are written in allocation order, so they give the mapping. A per-kernel MIM label
becomes a runnable policy with untouched allocations ('.') unpinned ('m'), as in perkernel.py's search.
Repeated runs are combined by their median; a policy with no finished run shows as 'timeout'. sgemm (one
kernel) is reported as the control for the per-kernel path. MIM policies that were never run are listed with
the perkernel.py command that runs them.

Usage (from tools/fault_plotsv3):
    python3 perkernel_compare.py kernel_labels_perkernel.csv --apps 2MM GRAMSCHM sgemm [--host cuda-host]
"""
import argparse
import glob
import os
import re
import socket

import numpy as np
import pandas as pd

import config
from kernel_labels import metric_file

# demangled kernel name prefix -> kernel index in launch order; anything else (e.g. warmup) is ignored
KERNELS = {
    '2MM': {'mm2_kernel1': 0, 'mm2_kernel2': 1},
    'GRAMSCHM': {'gramschmidt_kernel1': 0, 'gramschmidt_kernel2': 1, 'gramschmidt_kernel3': 2},
    'sgemm': {'': 0},   # one cuBLAS GEMM; every labeled kernel but warmup is it
}
RECORD = np.dtype([('faddr', 'u8'), ('timestamp', 'u8'), ('num_instances', 'u2'), ('pad', 'u1', (11,)),
                   ('record_type', 'u1'), ('_padding', 'u2')])


def allocation_order(app, psize):
    """Base addresses of the app's managed allocations, in allocation order, from its fault log."""
    path, = glob.glob(f'../../benchmarks/default/{app}/log_*_{psize}_{app}/{app}_klog')
    with open(path, 'rb') as file:
        mode, hostname_len = file.read(2)
        file.read(hostname_len)
        if mode != 0:
            raise ValueError(f"{path}: short-mode logs are not supported here")
        records = np.fromfile(file, dtype=RECORD)
    return list(records['faddr'][records['record_type'] == 3])


def to_policy(app, psize, label, n_allocs, host):
    """Reorder a label (one letter per allocation, by label) into allocation order; '.' -> 'm'."""
    metrics = pd.read_csv(metric_file(f'../../figs/{host}', 'metrics_stats_relative', app, psize))
    metrics = metrics.sort_values('label').reset_index(drop=True)
    # keep the labeled allocations; this drops the CUDA runtime's own 2 MiB managed range (logged first)
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
    per_kernel = ['m' * n_allocs] * n_kernels
    for _, row in rows.iterrows():
        index = kernel_index(app, row['kernel_name'].replace('void ', ''))
        if index is not None:
            per_kernel[index] = to_policy(app, psize, row['kernel_label'], n_allocs, host)
    return app_policy, ':'.join(per_kernel) + (':' if n_kernels == 1 else '')


def load_times(app):
    """{(psize %, policy): [seconds, or None for a run that did not finish]} from the app's raw .data file.

    Same rules as perkernel.py's load_results: an interrupted run (no outcome line) is ignored, and a
    GRAMSCHM run that stopped itself at its limit counts as not finished. Reads the .data directly because
    perf_data2csv.py writes a timed-out run's limit as its time.
    """
    times = {}
    # per-kernel runs only from 'perkernel2' (after the fix to per-kernel 'd'); whole-run runs from both
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
    """Median of the finished runs; None if none finished (shown as a timeout)."""
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
        n_kernels = len(KERNELS[app])
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
                # one kernel: "xyz:" (per-kernel path) must time like "xyz" (whole-run path)
                diffs = [abs(t[p] - t[p + ':']) / t[p] for p in whole if t.get(p + ':') is not None]
                row['control_pairs'] = len(diffs)
                row['control_max_rel_diff'] = round(max(diffs), 3) if diffs else None
            elif per_kernel:
                kernel_opt = min(per_kernel, key=per_kernel.get)
                row.update({'kernel_opt': kernel_opt, 'kernel_opt_s': round(per_kernel[kernel_opt], 1),
                            'kernel_opt_speedup': round(whole[app_opt] / per_kernel[kernel_opt], 3),
                            'mim_kernel': mim_kernel, 'mim_kernel_s': timing(mim_kernel) if mim_kernel else None})
            else:
                row['kernel_opt'] = 'no per-kernel runs'
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
