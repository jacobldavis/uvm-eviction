#!/usr/bin/env python3
import argparse
import os
import shutil
import socket
import subprocess

import numpy as np
import pandas as pd

from parse_metrics_relative import app_is_streaming, knapsack

CAPACITY = 0.99
STREAM_D_MIGRATE = 0.13
NONSTREAM_D_MIGRATE = 0.02
NONSTREAM_DR_MIGRATE = 1
UNTOUCHED = '.'
KERNEL_BETWEEN = "between"


def mim_label(allocs):
    """allocs: one row per allocation, sorted by label. Returns (streaming, label string)."""
    tmp = allocs.reset_index(drop=True).copy()
    tmp['size'] = tmp['size'] / 100.0
    streaming = app_is_streaming(tmp)
    if streaming:
        m_prediction = tmp['d_mean_1000'] >= STREAM_D_MIGRATE
        value = -1 * tmp['dc_intra_rel_mean_1000']
    else:
        m_prediction = (tmp['d_mean_1000'] > NONSTREAM_D_MIGRATE) & (tmp['dr_intra_mean_1000'] > NONSTREAM_DR_MIGRATE)
        value = tmp[['tr_median_1000', 'ts_rel_median_1000']].max(axis=1)
    # knapsack needs positive values; same shift rule as classify_d_with()
    min_item = value.min()
    if min_item < 0:
        value = value + 1.001 * abs(float(min_item))
    if min_item <= 0:
        value = value + 0.001
    tmp['new'] = value
    on_device = set(knapsack(tmp, CAPACITY, 'size', 'new'))
    strategy = ['d' if i in on_device else ('m' if m_prediction[i] else 'h') for i in range(len(tmp))]
    return streaming, ''.join(strategy)


def metric_file(root, kind, app, psize):
    directory = os.path.join(root, kind, app)
    if not os.path.isdir(directory):
        return None
    matches = [f for f in os.listdir(directory)
               if f.startswith('default_') and f.endswith(f'_{psize}_{app}.txt') and 'nopf' not in f]
    return os.path.join(directory, sorted(matches)[0]) if matches else None


def demangle(names):
    if not shutil.which('c++filt'):
        return names
    out = subprocess.run(['c++filt'], input='\n'.join(names), capture_output=True, text=True).stdout.splitlines()
    return out if len(out) == len(names) else names


def compare(app, psize, app_df, kernel_df, min_kernel_faults):
    app_df = app_df.sort_values('label').reset_index(drop=True)
    labels = list(app_df['label'])
    app_streaming, app_label = mim_label(app_df)
    app_letter = dict(zip(labels, app_label))

    rows = []
    for kernel, group in kernel_df.groupby('kernel', sort=False):
        group = group.set_index('label').reindex(labels)
        touched = group['f_rel_mean_1'].notna()
        if not touched.any():
            continue
        streaming, sub_label = mim_label(group[touched].reset_index())
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
    args = parser.parse_args()

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
