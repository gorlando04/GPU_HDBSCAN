#!/bin/bash

function clean_job() {
	  echo "Limpando ambiente..."
	    rm -rf "${local_job}"
    }
repeat(){

	sleep 1
	N=1000000
	shards=3
	ID=10M
        for i in 1 2 3 4 5
        do 
        
 

		## Cria o dataset em .txt
		python3 /nndescent/GPU_HDBSCAN/data/artificial/create.py $N
		sleep 1
		

		#./hdbscan_ > /nndescent/GPU_HDBSCAN/experiments/HDBSCAN/results/$ID/HDBSCAN_$i.txt
		./hdbscan_ 1000000 3
		python3 eval.py -name standart -D 12 -N $N -iter $i -mpts 32

		rm /nndescent/GPU_HDBSCAN/results/NNDescent-KNNG.*
		rm /nndescent/GPU_HDBSCAN/HDBSCAN/groundtruth/approximate_result.txt
		rm /nndescent/GPU_HDBSCAN/data/vectors.*

        sleep 2; 


         done
	exit
}

    trap clean_job EXIT HUP INT TERM ERR

    set -eE

    umask 077

    repeat

    echo exit
