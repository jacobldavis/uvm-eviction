#!/bin/bash
logbase=bicg${PERF_TAG:+_$PERF_TAG}
exe=./bicg.exe
cmd="$exe $1 0 ${@:2}"
out=$cmd

../common/numa_exp.sh $logbase "$cmd" "$out"
