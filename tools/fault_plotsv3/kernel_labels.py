#!/usr/bin/env python3
import argparse
import os
import shutil
import socket
import subprocess

import glob
import heapq

import numpy as np
import pandas as pd

import config
from parse_metrics_relative import app_is_streaming, knapsack

CAPACITY = 0.99
STREAM_D_MIGRATE = 0.13
NONSTREAM_D_MIGRATE = 0.02
NONSTREAM_DR_MIGRATE = 1
UNTOUCHED = '.'
KERNEL_BETWEEN = "between"
RECORD = np.dtype([('faddr', 'u8'), ('timestamp', 'u8'), ('num_instances', 'u2'), ('pad', 'u1', (11,)),
                   ('record_type', 'u1'), ('_padding', 'u2')])


def knapsack_fixed(df, cap, cost_str, val_str, threshold=0.02):
    # same as parse_metrics_relative.knapsack, but the last item can be picked
    costs = df[cost_str].values
    values = df[val_str].values
    low = [i for i in range(len(costs)) if costs[i] < threshold]
    remaining = cap - sum(costs[i] for i in low)
    items = sorted([(i, costs[i], values[i]) for i in range(len(costs)) if i not in low], key=lambda x: -x[2] / x[1])
    if not items or remaining <= 0:
        return low
    best_value, best = sum(values[i] for i in low), low
    heap = [(-best_value, 0, 0, best_value, low)]
    while heap:
        neg, level, cost, value, solution = heapq.heappop(heap)
        if value > best_value:
            best_value, best = value, solution
        if level == len(items) or cost > remaining:
            continue
        i, c, v = items[level]
        if cost + c <= remaining:
            heapq.heappush(heap, (neg - v, level + 1, cost + c, value + v, solution + [i]))
        heapq.heappush(heap, (neg, level + 1, cost, value, solution))
    return best


def knapsack_dp(df, cap, cost_str, val_str, threshold=0.02, unit=0.0001):
    # exact 0/1 knapsack over capacity in `unit` steps (the metric files' size precision);
    # sizes round up so a pick never overfills
    costs = df[cost_str].values
    values = df[val_str].values
    low = [i for i in range(len(costs)) if costs[i] < threshold]
    rest = [i for i in range(len(costs)) if i not in low]
    slots = int((cap - sum(costs[i] for i in low)) / unit + 1e-9)
    if not rest or slots <= 0:
        return low
    weight = [int(np.ceil(costs[i] / unit - 1e-9)) for i in rest]
    best = [0.0] * (slots + 1)
    keep = [[False] * (slots + 1) for _ in rest]
    for k, i in enumerate(rest):
        for c in range(slots, weight[k] - 1, -1):
            if best[c - weight[k]] + values[i] > best[c]:
                best[c] = best[c - weight[k]] + values[i]
                keep[k][c] = True
    chosen, c = [], slots
    for k in range(len(rest) - 1, -1, -1):
        if keep[k][c]:
            chosen.append(rest[k])
            c -= weight[k]
    return low + chosen


KNAPSACK = knapsack
EXCLUDE_HOST_READ = False


def mim_label(allocs, pinnable=None):
    tmp = allocs.reset_index(drop=True).copy()
    tmp['size'] = tmp['size'] / 100.0
    streaming = app_is_streaming(tmp)
    if streaming:
        m_prediction = tmp['d_mean_1000'] >= STREAM_D_MIGRATE
        value = -1 * tmp['dc_intra_rel_mean_1000']
    else:
        m_prediction = (tmp['d_mean_1000'] > NONSTREAM_D_MIGRATE) & (tmp['dr_intra_mean_1000'] > NONSTREAM_DR_MIGRATE)
        value = tmp[['tr_median_1000', 'ts_rel_median_1000']].max(axis=1)
    min_item = value.min()
    if min_item < 0:
        value = value + 1.001 * abs(float(min_item))
    if min_item <= 0:
        value = value + 0.001
    tmp['new'] = value
    # allocations that can't be 'd' are left out of the knapsack
    candidates = tmp if pinnable is None else tmp[np.asarray(pinnable)]
    on_device = {candidates.index[i] for i in KNAPSACK(candidates.reset_index(drop=True), CAPACITY, 'size', 'new')}
    strategy = ['d' if i in on_device else ('m' if m_prediction[i] else 'h') for i in range(len(tmp))]
    return streaming, ''.join(strategy)


def metric_file(root, kind, app, psize):
    directory = os.path.join(root, kind, app)
    if not os.path.isdir(directory):
        return None
    matches = [f for f in os.listdir(directory)
               if f.startswith('default_') and f.endswith(f'_{psize}_{app}.txt') and 'nopf' not in f]
    return os.path.join(directory, sorted(matches)[0]) if matches else None


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


def host_read_bases(app, psize, app_df):
    # base addresses of the allocations in config.HOST_READ
    if not EXCLUDE_HOST_READ or app not in config.HOST_READ:
        return set()
    order = [base for base in allocation_order(app, psize) if base in set(app_df['base_address'])]
    return {order[i] for i in config.HOST_READ[app] if i < len(order)}


def demangle(names):
    if not shutil.which('c++filt'):
        return names
    out = subprocess.run(['c++filt'], input='\n'.join(names), capture_output=True, text=True).stdout.splitlines()
    return out if len(out) == len(names) else names


def compare(app, psize, app_df, kernel_df, min_kernel_faults):
    app_df = app_df.sort_values('label').reset_index(drop=True)
    labels = list(app_df['label'])
    blocked = host_read_bases(app, psize, app_df)
    app_streaming, app_label = mim_label(app_df, ~app_df['base_address'].isin(blocked))
    app_letter = dict(zip(labels, app_label))

    rows = []
    for kernel, group in kernel_df.groupby('kernel', sort=False):
        group = group.set_index('label').reindex(labels)
        touched = group['f_rel_mean_1'].notna()
        if not touched.any():
            continue
        sub = group[touched].reset_index()
        streaming, sub_label = mim_label(sub, ~sub['base_address'].isin(blocked))
        letters = dict(zip(group.index[touched], sub_label))
        kernel_label = ''.join(letters.get(label, UNTOUCHED) for label in labels)
        differ = sum(letters[label] != app_letter[label] for label in letters)
        rows.append({'app': app, 'psize': psize, 'kernel': kernel,
                     'launches': int(group['launches'].dropna().iloc[0]),
                     'kernel_faults': int(group['kernel_faults'].dropna().iloc[0]),
                     'app_label': app_label, 'kernel_label': kernel_label,
                     'app_streaming': app_streaming, 'kernel_streaming': streaming,
                     'allocs_compared': int(touched.sum()), 'allocs_differ': differ,
                     'label_differs': differ > 0,
                     'included': kernel != KERNEL_BETWEEN and group['kernel_faults'].dropna().iloc[0] >= min_kernel_faults})
    return rows


def summarize(df):
    inc = df[df['included']]
    summary = []
    for (app, psize), g in inc.groupby(['app', 'psize']):
        differs = g[g['label_differs']]
        summary.append({'app': app, 'psize': psize, 'app_label': g['app_label'].iloc[0],
                        'kernels': len(g), 'kernels_differ': len(differs),
                        'pct_kernels_differ': round(100 * len(differs) / len(g), 1),
                        'pct_faults_in_differing_kernels': round(100 * differs['kernel_faults'].sum() / g['kernel_faults'].sum(), 1),
                        'pct_alloc_letters_differ': round(100 * g['allocs_differ'].sum() / g['allocs_compared'].sum(), 1)})
    return pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description="Compare per-kernel MIM labels against application-level labels")
    parser.add_argument('--host', default=socket.gethostname(), help="figs/<host> to read (default: this hostname)")
    parser.add_argument('--apps', nargs='+', required=True)
    parser.add_argument('--psizes', nargs='+', default=['100', '125', '150', '175'])
    parser.add_argument('--min-kernel-faults', type=int, default=0,
                        help="leave kernels with fewer faults out of the summary (they are still listed)")
    parser.add_argument('-o', '--output', default='kernel_labels', help="output prefix for <prefix>.csv and <prefix>_summary.csv")
    parser.add_argument('--fixed-knapsack', action='store_true', help="use the corrected knapsack instead of the paper's")
    parser.add_argument('--dp-knapsack', action='store_true', help="use the DP knapsack instead of the paper's")
    parser.add_argument('--exclude-host-read', action='store_true', help="never pin the allocations in config.HOST_READ")
    args = parser.parse_args()
    global KNAPSACK, EXCLUDE_HOST_READ
    if args.fixed_knapsack:
        KNAPSACK = knapsack_fixed
    if args.dp_knapsack:
        KNAPSACK = knapsack_dp
    EXCLUDE_HOST_READ = args.exclude_host_read

    root = f'../../figs/{args.host}'
    rows = []
    for app in args.apps:
        for psize in args.psizes:
            app_path = metric_file(root, 'metrics_stats_relative', app, psize)
            kernel_path = metric_file(root, 'metrics_stats_relative_kernel', app, psize)
            if not app_path or not kernel_path:
                print(f"skipping {app} {psize}%: missing {'app-level' if not app_path else 'per-kernel'} metrics under {root}")
                continue
            rows += compare(app, psize, pd.read_csv(app_path), pd.read_csv(kernel_path), args.min_kernel_faults)

    if not rows:
        print("no metric pairs found")
        return
    df = pd.DataFrame(rows)
    df.insert(3, 'kernel_name', demangle(list(df['kernel'])))
    df.to_csv(f'{args.output}.csv', index=False)
    summary = summarize(df)
    summary.to_csv(f'{args.output}_summary.csv', index=False)

    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 60)
    print(df[['app', 'psize', 'kernel_name', 'launches', 'kernel_faults', 'app_label', 'kernel_label', 'label_differs', 'included']].to_string(index=False))
    print()
    print(summary.to_string(index=False))
    print(f"\nwrote {args.output}.csv and {args.output}_summary.csv")


if __name__ == "__main__":
    main()
