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
nohup python runner_Top.py -k DataA -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 1 --end 40 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataA_MET_Run2018_1_to_40.txt 2>&1 &
echo "DataA_MET_Run2018_1_to_40 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

#DataB
nohup python runner_Top.py -k DataB -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 1 --end 22 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataB_MET_Run2018_1_to_22.txt 2>&1 &
echo "DataB_MET_Run2018_1_to_22 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

#DataC
nohup python runner_Top.py -k DataC -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 1 --end 24 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataC_MET_Run2018_1_to_24.txt 2>&1 &
echo "DataC_MET_Run2018_1_to_24 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

#DataD
nohup python runner_Top.py -k DataD -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 1 --end 50 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataD_MET_Run2018_1_to_50.txt 2>&1 &
echo "DataD_MET_Run2018_1_to_50 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

nohup python runner_Top.py -k DataD -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 51 --end 100 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataD_MET_Run2018_51_to_100.txt 2>&1 &
echo "DataD_MET_Run2018_51_to_100 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

nohup python runner_Top.py -k DataD -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 101 --end 121 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataD_MET_Run2018_101_to_121.txt 2>&1 &
echo "DataD_MET_Run2018_101_to_121 submitted with job id $!" | tee -a ./logs/jobids.txt

echo "The End"
