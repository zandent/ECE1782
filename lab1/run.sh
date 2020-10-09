#!/bin/bash
echo "running 16384 16384"
out/$1 16384 16384
echo "running 32768 8192"
out/$1 32768 8192
echo "running 30 8947850"
out/$1 30 8947850
