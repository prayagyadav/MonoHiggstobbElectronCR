#!/usr/bin/bash
echo "CONDOR BATCH SUBMIT"
echo " "
echo "MCSingleTop1 and MCSingleTop2"
echo "starting..."
#export COFFEA_IMAGE=coffeateam/coffea-dask-cc7:latest-py3.10
#export EXTERNAL_BIND=../
CATEGORY="boosted"
LEPTON="e"
WORKERS=16
REDIRECTOR="fnal"
OUTPUT_DIRECTORY="debug_Output"
mkdir -p ./logs

#MCSingleTop1
nohup python runner_Top.py -k MCSingleTop1 -e condor -c 500000 --outdir ${OUTPUT_DIRECTORY} --lepton ${LEPTON} --begin 1 --end 50 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_MCSingleTop1_1_to_50.txt 2>&1 &
echo "MCSingleTop1_1_to_50 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

nohup python runner_Top.py -k MCSingleTop1 -e condor -c 500000 --outdir ${OUTPUT_DIRECTORY} --lepton ${LEPTON} --begin 51 --end 100 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_MCSingleTop1_51_to_100.txt 2>&1 &
echo "MCSingleTop1_51_to_100 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

nohup python runner_Top.py -k MCSingleTop1 -e condor -c 500000 --outdir ${OUTPUT_DIRECTORY} --lepton ${LEPTON} --begin 101 --end 150 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_MCSingleTop1_101_to_150.txt 2>&1 &
echo "MCSingleTop1_101_to_150 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

nohup python runner_Top.py -k MCSingleTop1 -e condor -c 500000 --outdir ${OUTPUT_DIRECTORY} --lepton ${LEPTON} --begin 151 --end 200 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_MCSingleTop1_151_to_200.txt 2>&1 &
echo "MCSingleTop1_151_to_200 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

nohup python runner_Top.py -k MCSingleTop1 -e condor -c 500000 --outdir ${OUTPUT_DIRECTORY} --lepton ${LEPTON} --begin 201 --end 250 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_MCSingleTop1_201_to_250.txt 2>&1 &
echo "MCSingleTop1_201_to_250 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

nohup python runner_Top.py -k MCSingleTop1 -e condor -c 500000 --outdir ${OUTPUT_DIRECTORY} --lepton ${LEPTON} --begin 251 --end 279 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_MCSingleTop1_251_to_279.txt 2>&1 &
echo "MCSingleTop1_251_to_279 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

#MCSingleTop2
nohup python runner_Top.py -k MCSingleTop2 -e condor -c 500000 --outdir ${OUTPUT_DIRECTORY} --lepton ${LEPTON} --begin 1 --end 52 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_MCSingleTop2_1_to_52.txt 2>&1 &
echo "MCSingleTop2_1_to_52 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

nohup python runner_Top.py -k MCSingleTop2 -e condor -c 500000 --outdir ${OUTPUT_DIRECTORY} --lepton ${LEPTON} --begin 53 --end 75 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_MCSingleTop2_53_to_75.txt 2>&1 &
echo "MCSingleTop2_53_to_75 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

echo "The End"
