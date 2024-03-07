# monohiggsbb analysis


## To run processor locally: 

```
voms-proxy-init -voms cms -rfc -valid 192:00
./shell
python runFullDataset_MC18.py MCTTbar1l1v --era 2018 -e local --chunksize 100 --maxchunks 2 --workers 4 --outdir Outputs
```

## To run processor on condor: 

```
voms-proxy-init -voms cms -rfc -valid 192:00
./shell
python runFullDataset_MC18.py MCTTbar1l1v --era 2018 -e wiscjq --chunksize 100 --maxchunks 2 --workers 4 --outdir Outputs
```

- [ ] You can give which process to run in the second argument while running runFullDataset file. 
- [ ] You can also change the chunksize (10k or 100k is a good option).
- [ ] If want to run over full dataset, remove the --maxchunks argument since default is set to 'None', which runs over all root files provided in samples/files 
- [ ] For stable runs, number of workers is best kept below few 10s. 

 
***
