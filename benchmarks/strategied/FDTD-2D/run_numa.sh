#!/bin/bash

logbase=fdtd-2d${PERF_TAG:+_$PERF_TAG}
exe=./fdtd2d.exe
cmd="$exe $1 0 ${@:2}"
out=$cmd
../common/numa_exp.sh $logbase "$cmd" "$out"

