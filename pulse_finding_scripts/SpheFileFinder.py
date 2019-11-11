import csv
from datetime import datetime

# Function to figure out what sphe file to use based on input dataset date
# Reads master list of sphe files.  
# input data should get processed with sphe file from date *before* its own date

def GetSpheFile( inputdate_str ):
    
    # loop through sphe file list and find when the dataset in the file
    # is earlier than the input dataset.  that is the sphe file that should be
    # used.
    
    inputdate = datetime.strptime( inputdate_str, "%Y%m%dT%H%M")
    
    try:
        #f_list = csv.reader(open("/net/cms24/cms24r0/sjh/ScreenerCode_git/data_processing_scripts/pmt_sphe_files_run2_sally_corrected/processing_list.txt"))
        f_list = csv.reader(open("/net/cms24/cms24r0/sjh/SamplerCode_git/data_processing_scripts/pmt_sphe_files/processing_list.txt"))
        
        for line in reversed( list(f_list) ):
            
            testdate = datetime.strptime( line[0], "%Y%m%dT%H%M")
            
            if testdate <= inputdate:
                sphe_file_str = line[1]
                break
            
    except IOError:
        print "Couldn't find or open the sphe processing file! Exiting..."
        exit(1)
    
    return sphe_file_str
