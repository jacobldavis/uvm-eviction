#!/usr/bin/env python3
"""
kernel_transitions.py - how often the per-kernel MIM label changes from one kernel launch to the next.

Walks each run's launch order (<app>_kernels.csv, written by tools/kernel_marker next to the _klog) and looks
up each launch's per-kernel label in kernel_labels.py's output. Launches of kernels without a label (no
faults on a significant allocation, or left out by --min-kernel-faults) are skipped, since they would not
change placement. A boundary is two consecutive labeled launches of different kernels; it is a label change
if the two kernels disagree on any allocation both of them touched. Boundaries with no allocation in common
are counted separately as disjoint: their labels cannot disagree, but a per-kernel policy would still move
different allocations there.

Usage (from tools/fault_plotsv3, after kernel_labels.py):
    python3 kernel_transitions.py kernel_labels_tealeaf_cg.csv [-o kernel_transitions]
"""
import argparse
import glob
from collections import Counter

import pandas as pd

UNTOUCHED = '.'


def launch_order(app, psize):
    paths = glob.glob(f'../../benchmarks/default/{app}/log_*_{psize}_{app}/{app}_kernels.csv')
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one kernel map for {app} {psize}%, found {paths}")
    with open(paths[0]) as file:
        next(file)
        # kernel ids are the launch sequence; names may contain commas, so split once
        return [line.rstrip('\n').split(',', 1)[1] for line in file]


def compare_labels(a, b):
    common = [(x, y) for x, y in zip(a, b) if x != UNTOUCHED and y != UNTOUCHED]
    if not common:
        return 'disjoint'
    return 'change' if any(x != y for x, y in common) else 'same'


def main():
    parser = argparse.ArgumentParser(description="Count per-kernel label changes between consecutive launches")
    parser.add_argument('labels_csv', help="per-kernel CSV written by kernel_labels.py")
    parser.add_argument('-o', '--output', default='kernel_transitions', help="output prefix")
    args = parser.parse_args()

    df = pd.read_csv(args.labels_csv)
    df = df[df['included']]
    summary, pairs = [], []
    for (app, psize), g in df.groupby(['app', 'psize']):
        label = dict(zip(g['kernel'], g['kernel_label']))
        short = dict(zip(g['kernel'], g['kernel_name'].str.split('(').str[0].str.replace('void ', '')))
        order = launch_order(app, psize)
        labeled = [name for name in order if name in label]
        outcomes = Counter()
        pair_counts = Counter()
        for prev, cur in zip(labeled, labeled[1:]):
            if prev == cur:
                continue
            outcome = compare_labels(label[prev], label[cur])
            outcomes[outcome] += 1
            pair_counts[(prev, cur, outcome)] += 1
        boundaries = sum(outcomes.values())
        summary.append({'app': app, 'psize': psize, 'launches': len(order), 'labeled_launches': len(labeled),
                        'boundaries': boundaries, 'label_changes': outcomes['change'],
                        'same': outcomes['same'], 'disjoint': outcomes['disjoint'],
                        'pct_boundaries_changing': round(100 * outcomes['change'] / boundaries, 1) if boundaries else 0.0})
        for (prev, cur, outcome), count in pair_counts.items():
            pairs.append({'app': app, 'psize': psize, 'from': short[prev], 'to': short[cur],
                          'from_label': label[prev], 'to_label': label[cur], 'outcome': outcome, 'count': count})

    summary = pd.DataFrame(summary)
    pairs = pd.DataFrame(pairs).sort_values(['app', 'psize', 'count'], ascending=[True, True, False])
    summary.to_csv(f'{args.output}.csv', index=False)
    pairs.to_csv(f'{args.output}_pairs.csv', index=False)
    pd.set_option('display.width', 220)
    print(summary.to_string(index=False))
    print()
    print(pairs.to_string(index=False))
    print(f"\nwrote {args.output}.csv and {args.output}_pairs.csv")


if __name__ == "__main__":
    main()
