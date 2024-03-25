#!/usr/bin/bash
echo "CONDOR BATCH SUBMIT"
echo " "
echo "Data 2018"
echo "starting..."
#export COFFEA_IMAGE=coffeateam/coffea-dask-cc7:latest-py3.10
#export EXTERNAL_BIND=../
CATEGORY="boosted"
LEPTON="e"
WORKERS=16
REDIRECTOR="wisc"
OUTPUT_DIRECTORY="debug_Output"
mkdir -p ./logs

#python prepare_for_run.py
#echo "created helper_files.zip"

#DataA
nohup python runner_Top.py -k DataA -e condor -c 500000 -lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 1 --end 39 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataA_MET_Run2018_1_to_39.txt 2>&1 &
echo "DataA_MET_Run2018_1_to_39 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

#DataB
nohup python runner_Top.py -k DataB -e condor -c 500000 -lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 40 --end 61 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataB_MET_Run2018_40_to_61.txt 2>&1 &
echo "DataB_MET_Run2018_40_to_61 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

#DataC
nohup python runner_Top.py -k DataC -e condor -c 500000 -lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 62 --end 85 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataC_MET_Run2018_62_to_85.txt 2>&1 &
echo "DataC_MET_Run2018_62_to_85 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

#DataD
nohup python runner_Top.py -k DataD -e condor -c 500000 -lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 86 --end 206 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataD_MET_Run2018_86_to_206.txt 2>&1 &
echo "DataD_MET_Run2018_86_to_206 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

echo "The End"
