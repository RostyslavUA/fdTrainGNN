#!/bin/bash

act='sigmoid'
feat_type='mix'
w_multiplier=1.0
architecture='centralized'
batch_size=100
# for opt in 'sgd' 'adam' ; do
#     echo "Launch strategy ${architecture} ${feat_type} ${w_multiplier} ${act} ${batch_size} ${opt}"
#     python regression.py ${architecture} ${feat_type} ${w_multiplier} ${act} ${batch_size} ${opt} &
#     running_jobs=$(jobs -p | wc -l)
#     if [ $running_jobs -ge 12 ]; then
#         wait -n
#     fi
# done

architecture='distributed'
for opt in 'dsgd' 'dadam' 'dams' ; do
    echo "Launch strategy ${architecture} ${feat_type} ${w_multiplier} ${act} ${batch_size} ${opt}"
    python regression.py ${architecture} ${feat_type} ${w_multiplier} ${act} ${batch_size} ${opt} &
    running_jobs=$(jobs -p | wc -l)
    if [ $running_jobs -ge 12 ]; then
        wait -n
    fi
done
wait
