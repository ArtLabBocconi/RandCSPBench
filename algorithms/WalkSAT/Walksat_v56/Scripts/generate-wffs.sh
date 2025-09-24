#!/bin/bash

if (( $# != 4 )); then
    echo "use: generate-wffs.sh number_vars ratio start_wff_numer end_wff_number"
    echo "  Enter ratio as a percentage, e.g. 425 == 4.25 clause/variable"
    exit
fi

v=$1
r=$2
s=$3
e=$4

(( c = ( v * r ) / 100 ))
for x in $(seq -f "%03g" $s $e)
do
   ../makewff -cnf 3 $v $c > f$v-r$r-x$x.cnf
done


