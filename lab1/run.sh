#!/bin/bash
echo "running 16384 16384"
out/main_v$1 16384 16384
echo "running 32768 8192"
out/main_v$1 32768 8192
echo "running 30 8947850"
out/main_v$1 30 8947850
echo "running 1 333333333"
out/main_v$1 1 333333333
echo "running 3 111111111"
out/main_v$1 3 111111111
