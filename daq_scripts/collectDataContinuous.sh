#!/bin/bash

while [ 1 ]
do

  #DATASETNAME=${1}

  DATE=`date +%Y%m%d`
  TIME=`date +%H%M`
  
  DATASETNAME="screener_${DATE}T${TIME}"

  
  NFILES=10

  if [ ! -d "./screener/new/$DATASETNAME" ]; then
    mkdir ./screener/new/$DATASETNAME
  fi
  
  cp datasetConfig.txt ./screener/new/$DATASETNAME/
  echo "Taking dataset $DATASETNAME"
  for i in `seq 1 $NFILES`
  do
  
    echo "Collecting file $i of $NFILES..."
    #FILENAME="$DATASETNAME_$i.bin"
    ./fbinary_7binary_btime3_2 5000 10000 \
      ./screener/new/$DATASETNAME/screener10k_$i.bin > ./screener/new/$DATASETNAME/log_$i.txt
    sleep 1
  
  done

  touch ./screener/new/$DATASETNAME/acquisition_done.flag

  echo "Done."
  echo " "
  sleep 3
done
