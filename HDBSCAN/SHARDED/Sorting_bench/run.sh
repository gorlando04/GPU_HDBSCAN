#!/bin/bash

function clean_job() {
	  echo "Limpando ambiente..."
	    rm -rf "${local_job}"
    }
repeat(){


        for i in 1000000 2000000 5000000 10000000 20000000 40000000 80000000 160000000 320000000
        do 

	mkdir results-$i        
        for iters in 0 1 2 3 4 5 6 7 8 9
        do

        ./main $i $iters

        done

         done 
	exit
}

    trap clean_job EXIT HUP INT TERM ERR

    set -eE

    umask 077

    repeat

    echo exit
