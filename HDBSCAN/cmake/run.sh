
#!/bin/bash

function clean_job() {
	  echo "Limpando ambiente..."
	    rm -rf "${local_job}"
    }
repeat(){

	sleep 1
	shards=16
        fodasse=0
	M='M'
        for n in 65000000 70000000 75000000 80000000 85000000 90000000 95000000 100000000
	do
	ID=$((n /1000000))
	mkdir ../results_large/$ID$M
         if [ $n -eq 25000000 ]
                then
                shards=`expr $shards + 4`

        fi

        if [ $n -eq 45000000 ]
                then
                shards=`expr $shards + 4`

        fi

        if [ $n -eq 70000000 ]
                then
                shards=`expr $shards + 4`

        fi
        if [ $n -eq 85000000 ]
                then
                shards=`expr $shards + 4`
        fi


		for iter in 1 2 3 4 5 6 7 8 9 10
		do
			## Cria o dataset em .txt
			python3 /nndescent/GPU_HDBSCAN/data/artificial/create.py $n
			sleep 1
                	echo "$n"
			file="../results_large/$ID$M/graphConstruction_${ID}_${iter}"
			file+=".txt"
			./hdbscan_ $n 32 $shards 0 > ${file}
			echo "$file"
			rm ../../results/NNDescent-KNNG.*
			rm ../../results/dict.*
			rm ../../results/euclidean.*
			rm /nndescent/GPU_HDBSCAN/data/vectors.*
		done

	done
	exit
}

    trap clean_job EXIT HUP INT TERM ERR

    set -eE

    umask 077

    repeat

    echo exit
