#!/usr/bin/bash
echo "CONDOR BATCH SUBMIT"
echo " "
echo "DYJets_LL"
echo "starting..."
#export COFFEA_IMAGE=coffeateam/coffea-dask-cc7:latest-py3.10
#export EXTERNAL_BIND=../
CATEGORY="resolved"
LEPTON="e"
WORKERS=16
REDIRECTOR="fnal"
mkdir -p ./logs

nohup python runner_Top.py -k DYJets_LL -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 1 --end 50 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DYJets_LL_1_to_50.txt 2>&1 &
echo "DYJets_LL_1_to_50 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

nohup python runner_Top.py -k DYJets_LL -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 51 --end 100 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DYJets_LL_51_to_100.txt 2>&1 &
echo "DYJets_LL_51_to_100 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

nohup python runner_Top.py -k DYJets_LL -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 101 --end 150 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DYJets_LL_101_to_150.txt 2>&1 &
echo "DYJets_LL_101_to_150 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

nohup python runner_Top.py -k DYJets_LL -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 151 --end 200 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DYJets_LL_151_to_200.txt 2>&1 &
echo "DYJets_LL_151_to_200 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

nohup python runner_Top.py -k DYJets_LL -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 201 --end 250 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DYJets_LL_201_to_250.txt 2>&1 &
echo "DYJets_LL_201_to_250 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

nohup python runner_Top.py -k DYJets_LL -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 251 --end 300 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DYJets_LL_251_to_300.txt 2>&1 &
echo "DYJets_LL_251_to_300 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

nohup python runner_Top.py -k DYJets_LL -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 301 --end 350 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DYJets_LL_301_to_350.txt 2>&1 &
echo "DYJets_LL_301_to_350 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

nohup python runner_Top.py -k DYJets_LL -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 351 --end 400 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DYJets_LL_351_to_400.txt 2>&1 &
echo "DYJets_LL_351_to_400 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

nohup python runner_Top.py -k DYJets_LL -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 401 --end 450 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DYJets_LL_401_to_450.txt 2>&1 &
echo "DYJets_LL_401_to_450 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

nohup python runner_Top.py -k DYJets_LL -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 451 --end 500 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DYJets_LL_451_to_500.txt 2>&1 &
echo "DYJets_LL_451_to_500 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

nohup python runner_Top.py -k DYJets_LL -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 501 --end 550 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DYJets_LL_501_to_550.txt 2>&1 &
echo "DYJets_LL_501_to_550 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

nohup python runner_Top.py -k DYJets_LL -e condor -c 500000 -cat ${CATEGORY} -lepton ${LEPTON} --begin 551 --end 592 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DYJets_LL_551_to_592.txt 2>&1 &
echo "DYJets_LL_551_to_592 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 30

echo "The End"
