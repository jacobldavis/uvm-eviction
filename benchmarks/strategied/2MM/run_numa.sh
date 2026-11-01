#!/bin/bash
logbase=2mm${PERF_TAG:+_$PERF_TAG}
exe=./2mm.exe
cmd="$exe $1 0 ${@:2}"
out=$cmd

../common/numa_exp.sh $logbase "$cmd" "$out"
