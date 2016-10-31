#!/bin/bash

if [ -f sorting_time_results.csv ]
then
    now=$(date +"%H%M%S_%d_%M_%Y")
    mv sorting_time_results.csv sorting_time_results_${now}.csv
fi

i=0
for j in {1..20..1}
do
    let "i=2**j"
    #sleep 15
    printf "Running simulation using 2**%d = %d\n"  "$j" "$i"
    time ./cdpSimpleQuicksort num_items=${i}
done

