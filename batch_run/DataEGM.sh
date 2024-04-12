#!/usr/bin/bash
echo "CONDOR BATCH SUBMIT"
echo " "
echo "Data 2018 EGM"
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
nohup python runner_Top.py -k DataA -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 1 --end 50 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataA_EGM_Run2018_1_to_50.txt 2>&1 &
echo "DataA_MET_Run2018_1_to_50 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

nohup python runner_Top.py -k DataA -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 51 --end 100 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataA_EGM_Run2018_51_to_100.txt 2>&1 &
echo "DataA_MET_Run2018_51_to_100 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

nohup python runner_Top.py -k DataA -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 101 --end 150 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataA_EGM_Run2018_101_to_150.txt 2>&1 &
echo "DataA_MET_Run2018_101_to_150 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

nohup python runner_Top.py -k DataA -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 151 --end 200 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataA_EGM_Run2018_151_to_200.txt 2>&1 &
echo "DataA_MET_Run2018_151_to_200 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

nohup python runner_Top.py -k DataA -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 201 --end 226 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataA_EGM_Run2018_201_to_226.txt 2>&1 &
echo "DataA_MET_Run2018_201_to_226 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

#DataB
nohup python runner_Top.py -k DataB -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 1 --end 50 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataB_EGM_Run2018_1_to_50.txt 2>&1 &
echo "DataB_MET_Run2018_1_to_50 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

nohup python runner_Top.py -k DataB -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 51 --end 74 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataB_EGM_Run2018_51_to_74.txt 2>&1 &
echo "DataB_MET_Run2018_51_to_74 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

#DataC
nohup python runner_Top.py -k DataC -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 1 --end 50 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataC_EGM_Run2018_1_to_50.txt 2>&1 &
echo "DataC_MET_Run2018_1_to_50 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

nohup python runner_Top.py -k DataC -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 51 --end 83 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataC_EGM_Run2018_51_to_83.txt 2>&1 &
echo "DataC_MET_Run2018_51_to_83 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

#DataD
nohup python runner_Top.py -k DataD -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 1 --end 50 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataD_EGM_Run2018_1_to_50.txt 2>&1 &
echo "DataD_MET_Run2018_1_to_50 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

nohup python runner_Top.py -k DataD -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 51 --end 100 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataD_EGM_Run2018_51_to_100.txt 2>&1 &
echo "DataD_MET_Run2018_51_to_100 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

nohup python runner_Top.py -k DataD -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 101 --end 150 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataD_EGM_Run2018_101_to_150.txt 2>&1 &
echo "DataD_MET_Run2018_101_to_150 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

nohup python runner_Top.py -k DataD -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 151 --end 200 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataD_EGM_Run2018_151_to_200.txt 2>&1 &
echo "DataD_MET_Run2018_151_to_200 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

nohup python runner_Top.py -k DataD -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 201 --end 250 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataD_EGM_Run2018_201_to_250.txt 2>&1 &
echo "DataD_MET_Run2018_201_to_250 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

nohup python runner_Top.py -k DataD -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 251 --end 300 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataD_EGM_Run2018_251_to_300.txt 2>&1 &
echo "DataD_MET_Run2018_251_to_300 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

nohup python runner_Top.py -k DataD -e condor -c 500000 --lepton ${LEPTON} --outdir ${OUTPUT_DIRECTORY} --begin 301 --end 355 -w ${WORKERS} --redirector ${REDIRECTOR}> ./logs/log_DataD_EGM_Run2018_301_to_355.txt 2>&1 &
echo "DataD_MET_Run2018_301_to_355 submitted with job id $!" | tee -a ./logs/jobids.txt
sleep 60

echo "The End"
