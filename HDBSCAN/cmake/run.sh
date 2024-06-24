
#!/bin/bash

function clean_job() {
	  echo "Limpando ambiente..."
	    rm -rf "${local_job}"
    }
repeat(){

	sleep 1
	shards=9
        fodasse=0
	M='M'
        for n in 100000000 
	do
	ID=$((n /1000000))
	#mkdir ../results_large/$ID$M
        #mkdir ../results_eclgraph/$ID$M-MST-1
 

		## Cria o dataset em .txt
	#	python3 /nndescent/GPU_HDBSCAN/data/artificial/create.py $n
		sleep 1
                echo "$n"
		file="../results_large/$ID$M/graphConstruction_${ID}"
		file+=".txt"
		./hdbscan_ $n 32 $shards 0 #> ${file}

		#rm ../../results/NNDescent-KNNG.*
		#rm /nndescent/GPU_HDBSCAN/HDBSCAN/groundtruth/approximate_result.txt
		#rm /nndescent/GPU_HDBSCAN/data/vectors.*

	if [ $n -eq 60000000 ]
		then
		shards=`expr $shards + 3`

	fi
        if [ $n -eq 80000000 ]
                then
		shards=`expr $shards + 3`
        fi

	done
	exit
}

    trap clean_job EXIT HUP INT TERM ERR

    set -eE

    umask 077

    repeat

    echo exit
