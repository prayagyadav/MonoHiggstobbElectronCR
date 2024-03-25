#!/usr/bin/bash
echo "CONDOR BATCH SUBMIT"
echo " "
echo "ST"
echo "starting..."
#export COFFEA_IMAGE=coffeateam/coffea-dask-cc7:latest-py3.10
#export EXTERNAL_BIND=../
CATEGORY="resolved"
LEPTON="e"
WORKERS=16
REDIRECTOR="fnal"
mkdir -p ./logs

nohup python runner_Top.py -k ST -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 1 --end 50 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_ST_1_to_50.txt 2>&1 &
echo "ST_1_to_50 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

nohup python runner_Top.py -k ST -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 51 --end 100 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_ST_51_to_100.txt 2>&1 &
echo "ST_51_to_100 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

nohup python runner_Top.py -k ST -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 101 --end 150 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_ST_101_to_150.txt 2>&1 &
echo "ST_101_to_150 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

nohup python runner_Top.py -k ST -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 151 --end 200 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_ST_151_to_200.txt 2>&1 &
echo "ST_151_to_200 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

nohup python runner_Top.py -k ST -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 201 --end 250 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_ST_201_to_250.txt 2>&1 &
echo "ST_201_to_250 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

nohup python runner_Top.py -k ST -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 251 --end 300 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_ST_251_to_300.txt 2>&1 &
echo "ST_251_to_300 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

nohup python runner_Top.py -k ST -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 301 --end 354 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_ST_301_to_354.txt 2>&1 &
echo "ST_301_to_354 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

echo "The End"
