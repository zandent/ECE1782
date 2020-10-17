#!/bin/bash
echo "-----lab2 repeater tester----"
for i in {1..5}
  do
     echo "run $i"
     out/main $1
 done
