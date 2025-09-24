#!/bin/bash -x
# use: 

if (( $# != 3 )); then
    echo "generate-hard-wffs.sh number_vars start_wff_numer end_wff_number"
    exit
fi

v=$1
s=$2
e=$3

(( r = 425 ))
(( c = ( v * r ) / 100 ))
for x in $(seq -f "%03g" $s $e)
do
   makewff -cnf 3 $v $c > f$v-r$r-x$x.cnf
done


