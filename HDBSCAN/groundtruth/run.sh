#!/bin/bash

function clean_job() {
	  echo "Limpando ambiente..."
	    rm -rf "${local_job}"
    }
repeat(){

	sleep 1


        for i in 1000000 5000000 10000000
        do 
	python3 groundtruth.py -name standart -D 12 -N $i -iter 0 -mpts 32


        sleep 2; 


         done
	exit
}

    trap clean_job EXIT HUP INT TERM ERR

    set -eE

    umask 077

    repeat

    echo exit
