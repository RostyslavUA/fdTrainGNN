#!/bin/bash
# gc=5.0
reg_const=0.0
thr=1
for file in 'main.py' 'main_d.py' ; do
    for opt in 'adam' ; do
        for s in "set1_UMa_Optional_square_n25_lim500_thr${thr}_fading1" ; do
#             for lr in 0.005 0.001 0.0005 ; do  # circle
            for lr in 0.005 0.01 ; do  # square
                    lr_sci=$(awk "BEGIN {printf \"%.0e\",$lr}")
                    if [[ "${file}" == "main.py" ]] ; then
                        echo "Launch ${file} ${s} uwmmse train ${opt} lr ${lr_sci} ${reg_const}"
                        python -u ${file} ${s} 'uwmmse' 'train' ${opt} ${lr} ${reg_const} | tee "output/uwmmse_${opt}_${s}_lr${lr_sci}_reg${reg_const}_mu0.txt" &
                    elif [[ "${file}" == "main_d.py" ]] ; then
                        echo "Launch ${file} ${s} duwmmse train ${opt} lr ${lr_sci} ${reg_const}"
                        python -u ${file} ${s} 'duwmmse' 'train' ${opt} ${lr} ${reg_const} | tee "output/duwmmse_${opt}_${s}_lr${lr_sci}_reg${reg_const}_mu0.txt" &
                    fi
                    running_jobs=$(jobs -p | wc -l)
                    if [ $running_jobs -ge 12 ]; then
                        wait -n
                    fi
            done
        done
    done
done
wait
