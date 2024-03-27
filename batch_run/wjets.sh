#!/usr/bin/bash
echo "CONDOR BATCH SUBMIT"
echo " "
echo "MCWlvJets"
echo "starting..."
#export COFFEA_IMAGE=coffeateam/coffea-dask-cc7:latest-py3.10
#export EXTERNAL_BIND=../
CATEGORY="boosted"
LEPTON="e"
WORKERS=16
REDIRECTOR="fnal"
OUTPUT_DIRECTORY="debug_Output"
mkdir -p ./logs

#nohup python runner_Top.py -k MCWlvJets -e condor -c 500000 --outdir ${OUTPUT_DIRECTORY} --lepton ${LEPTON} --begin 1 --end 50 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_MCWlvJets_1_to_50.txt 2>&1 &
#echo "MCWlvJets_1_to_50 submitted with job id $!" | tee -a ./logs/jobids.txt
#sleep 30

nohup python runner_Top.py -k MCWlvJets -e condor -c 500000 --outdir ${OUTPUT_DIRECTORY} --lepton ${LEPTON} --begin 51 --end 100 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_MCWlvJets_51_to_100.txt 2>&1 &
echo "MCWlvJets_51_to_100 submitted with job id $!" | tee -a ./logs/jobids.txt
#sleep 30

#nohup python runner_Top.py -k MCWlvJets -e condor -c 500000 --outdir ${OUTPUT_DIRECTORY} --lepton ${LEPTON} --begin 101 --end 164 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_MCWlvJets_101_to_164.txt 2>&1 &
#echo "MCWlvJets_101_to_164 submitted with job id $!" | tee -a ./logs/jobids.txt
#sleep 30

echo "The End"
