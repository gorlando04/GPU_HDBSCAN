#!/bin/bash

function clean_job() {
	  echo "Limpando ambiente..."
	    rm -rf "${local_job}"
    }
repeat(){

	sleep 1

	N=1000000
	shards=3
	ID=1M

        ## Cria o dataset em .txt
        python3 /nndescent/GPU_HDBSCAN/experiments/data/artificial/create.py $N
        sleep 1
        
        ## Transforma o dataset para um arquivo binário
        ./gknng 
        sleep 5

        ## Roda o NNDescent
        ./gknng false $shards > /nndescent/GPU_HDBSCAN/experiments/HDBSCAN/results/$ID/kNNG_$.txt


        # Remover os resultados
        rm /nndescent/GPU_KNNG/data/vectors.*



	exit
}

    trap clean_job EXIT HUP INT TERM ERR

    set -eE

    umask 077

    repeat

    echo exit
