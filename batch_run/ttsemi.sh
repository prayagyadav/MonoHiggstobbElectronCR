#!/usr/bin/bash
echo "CONDOR BATCH SUBMIT"
echo " "
echo "MCTTbar1l1v"
echo "starting..."
#export COFFEA_IMAGE=coffeateam/coffea-dask-cc7:latest-py3.10
#export EXTERNAL_BIND=../
CATEGORY="boosted"
LEPTON="e"
WORKERS=8
REDIRECTOR="infn"
OUTPUT_DIRECTORY="debug_Output"
mkdir -p ./logs

nohup python runner_Top.py -k MCTTbar1l1v -e condor -c 500000 --outdir ${OUTPUT_DIRECTORY} --lepton ${LEPTON} --begin 1 --end 47 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_MCTTbar1l1v_1_to_47.txt 2>&1 &
echo "MCTTbar1l1v_1_to_47 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 10

echo "The End"
