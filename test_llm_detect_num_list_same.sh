#!/bin/bash

rm -f /tmp/llm_test_*

for i in $(seq 1 100);
do
    i=$(printf "%03d" $i)
    echo $i
    ./generate_detect_num_list.py /tmp/llm_test_diff_$i 100 1
    time ( ls -1 /tmp/llm_test_diff_$i*.json | parallel -j 50 ./send_local_llm_query.py ) 
done
