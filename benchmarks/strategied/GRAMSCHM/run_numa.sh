#!/bin/bash
logbase=gramschm${PERF_TAG:+_$PERF_TAG}
exe=./gramschmidt.exe
cmd="$exe $1 0 $TIMEOUT ${@:2}"
out=$cmd

../common/numa_exp.sh $logbase "$cmd" "$out"
