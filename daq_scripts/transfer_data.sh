#!/bin/bash

#script to transfer data from DAQ computer to computer for processing/analysis
#once the acquisition has stopped

# Variables

#these two exist on the DDC10 computer:
#DATADIR=/net/cms26/cms26r0/sjh/new
#TRANSFERLOGDIR=/net/cms26/cms26r0/sjh/log
DATADIR=~/DDC10/miniscreener/new
TRANSFERLOGDIR=~/DDC10/miniscreener/log


NEWDIR=/net/cms26/cms26r0/msolmaz/HighBay_data/new
PROCESSINGDIR=/net/cms26/cms26r0/msolmaz/HighBay_data/processing
PROCESSEDDIR=/net/cms26/cms26r0/msolmaz/HighBay_data/processed
PROCESSINGFAILEDDIR=/net/cms26/cms26r0/msolmaz/HighBay_data/processing_failed

CODEDIR=/net/cms26/cms26r0/msolmaz/Scripts/data_processing_scripts

HOST=cms26.physics.ucsb.edu
USER=msolmaz

ACQUISITION_FLAG='acquisition_done.flag'
TRANSFER_FLAG='transfer_done.flag'
TO_PROCESS_FLAG='process_this.flag'

# Step 1: Check directory for new data ---------------------

while [ 1 ]
do
data_dirs=`ls -d $DATADIR/*`

for fulldir in $data_dirs
do
   
   dir=$(basename $fulldir)
   #echo $dir
   
   #if flag is present, transfer data to cms24
   if [ -f "$fulldir/$ACQUISITION_FLAG" ] && [ ! -f "$fulldir/$TRANSFER_FLAG" ]; then
      
      now=`date`
      
      echo "Starting transfer of $dir at $now" >> $TRANSFERLOGDIR/transfers.log
      
      rsync -rvt --timeout=600 $fulldir $USER@$HOST:$NEWDIR >> $TRANSFERLOGDIR/transfers.log
      sleep 1
      
      touch $fulldir/$TRANSFER_FLAG
      sleep 1
      
      rsync -rvt --timeout=600 --exclude="*.bin" --ignore-existing $fulldir/$TRANSFER_FLAG $USER@$HOST:$NEWDIR/$dir >> $TRANSFERLOGDIR/transfers.log
      sleep 1
      
      echo "Transfer of $dir completed at $now" >> $TRANSFERLOGDIR/transfers.log
      echo " " >> $TRANSFERLOGDIR/transfers.log
   fi

done
sleep 60
done
