
#!/bin/bash

function clean_job() {
	  echo "Limpando ambiente..."
	    rm -rf "${local_job}"
    }
repeat(){

	sleep 1
	shards=3
	M='M'
       buckets='0'
	threads='0'
        for n in 1000000 
	do
	ID=$((n /1000000))
	mkdir ../results/$ID$M-MST-0
        #mkdir ../results_eclgraph/$ID$M-MST-1
 

		## Cria o dataset em .txt
#		python3 /nndescent/GPU_HDBSCAN/data/artificial/create.py $n
		sleep 1

		for buckets in 128 #256 512 1024
		do
			for threads in 32 #64 128 256 512
			do
        			for i in 1 2 3 4 5
        			do 
					echo "$n 32  $i"
					file="../results/$ID$M-MST-0/graphConstruction_${i}"
					#file+="_${buckets}"
					#file+="_${threads}"
					file+=".txt"
					./hdbscan_ $n 32 $shards 0 $buckets $threads > ${file}
				done
			done
		#python3 eval.py -name standart -D 12 -N $N -iter $i -mpts 32

		#rm /nndescent/GPU_HDBSCAN/results/NNDescent-KNNG.*
		#rm /nndescent/GPU_HDBSCAN/HDBSCAN/groundtruth/approximate_result.txt
		#rm /nndescent/GPU_HDBSCAN/data/vectors.*
        	sleep 2; 

		done

	shards=`expr $shards + 3`
	done
	exit
}

    trap clean_job EXIT HUP INT TERM ERR

    set -eE

    umask 077

    repeat

    echo exit
