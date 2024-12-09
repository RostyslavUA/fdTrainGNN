#!/bin/bash

dataset="ER_Graph_Uniform_GEN21_test1"
output_dir="./output/${dataset}"
num_cons=1
mkdir -p "$output_dir"

for lr in 0.00005 ; do 
    for opt in 'Adam' ; do
        for arch in 'centralized' 'decentralized' ; do
            for dropout_op in 0.0 ; do
                lr_sci=$(awk "BEGIN {printf \"%.0e\",$lr}")
                if [ "$arch" == 'decentralized' ]; then
                    outname=${arch:0:1}_${opt}_lr${lr_sci}_drop${dropout_op}_cons${num_cons}_relu.out
                else
                    outname=${arch:0:1}_${opt}_lr${lr_sci}_drop${dropout_op}_relu.out
                fi
                echo "${output_dir}/${outname}"
                python -u mwis_gcn_train_twin.py --training_set=${arch:0:1}_ER\
                            --architecture=${arch}\
                            --optimizer=${opt}\
                            --num_cons=${num_cons}\
                            --dropout_op=${dropout_op}\
                            --epsilon=1.0\
                            --epsilon_min=0.001\
                            --gamma=0.99\
                            --feature_size=1\
                            --diver_num=1\
                            --max_degree=1\
                            --predict=mwis\
                            --learning_rate=${lr}\
                            --learning_decay=1.0\
                            --hidden1=32\
                            --num_layer=3\
                            --epochs=5\
                            --ntrain=1 | tee >( grep "Epoch: " > ${output_dir}/${outname}) &

                running_jobs=$(jobs -p | wc -l)
                if [ $running_jobs -ge 12 ]; then
                    wait
                fi
            done
        done
    done
done
# Wait for all jobs to finish
wait