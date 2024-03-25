#!/usr/bin/bash
echo "CONDOR BATCH SUBMIT"
echo " "
echo "MET_Run2018"
echo "starting..."
#export COFFEA_IMAGE=coffeateam/coffea-dask-cc7:latest-py3.10
#export EXTERNAL_BIND=../
CATEGORY="resolved"
LEPTON="e"
WORKERS=16
REDIRECTOR="wisc"
mkdir -p ./logs

python prepare_for_run.py
echo "created helper_files.zip"


nohup python runner_Top.py -k MET_Run2018 -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 1 --end 39 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_MET_Run2018_1_to_39.txt 2>&1 &
echo "MET_Run2018_1_to_39 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30


nohup python runner_Top.py -k MET_Run2018 -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 40 --end 61 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_MET_Run2018_40_to_61.txt 2>&1 &
echo "MET_Run2018_40_to_61 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30


nohup python runner_Top.py -k MET_Run2018 -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 62 --end 85 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_MET_Run2018_62_to_85.txt 2>&1 &
echo "MET_Run2018_62_to_85 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30


nohup python runner_Top.py -k MET_Run2018 -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 86 --end 206 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_MET_Run2018_86_to_206.txt 2>&1 &
echo "MET_Run2018_86_to_206 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

echo "The End"
