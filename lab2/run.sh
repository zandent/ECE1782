#!/bin/bash
echo "-----lab2 tester----"
for i in {1..1000..1}
  do
     echo "run size $i"
     out/main $i
 done
for i in {3..700..167}
  do
     echo "run size $i"
     out/main $i
 done
