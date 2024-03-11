#!/usr/bin/env python3
import uproot
import datetime
import logging
import hist
from coffea import processor, nanoevents, util
from coffea.nanoevents import NanoAODSchema
from monoHbb.utils.crossSections import lumis, crossSections
import time
import sys
import os
import argparse

#from monoHbb.processor_BCat_Topmu import monoHbbProcessor
from samples.files_Data18 import fileList_Data_18 as filelist

group_mapping = {
    "DataA": ['Data_MET_Run2018A'],
    "DataB": ['Data_MET_Run2018B'],
    "DataC": ['Data_MET_Run2018C'],
    "DataD": ['Data_MET_Run2018D'],
}

def move_X509():
    try:
        _x509_localpath = (
            [
                line
                for line in os.popen("voms-proxy-info").read().split("\n")
                if line.startswith("path")
            ][0]
            .split(":")[-1]
            .strip()
        )
    except Exception as err:
        raise RuntimeError(
            "x509 proxy could not be parsed, try creating it with 'voms-proxy-init'"
        ) from err
    _x509_path = f'/scratch/{os.environ["USER"]}/{_x509_localpath.split("/")[-1]}'
    os.system(f"cp {_x509_localpath} {_x509_path}")
    return os.path.basename(_x509_localpath)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
        level=logging.WARNING,
    )

    parser = argparse.ArgumentParser(
        description="Batch processing script"
    )
    parser.add_argument(
        "mcGroup",
        choices=list(group_mapping),
        help="Name of process to run",
    )
    parser.add_argument(
        "--era",
        type=int,
        choices=[2018, 2017],
        default=2018,
        help="Era to run over",
    )
    parser.add_argument("--chunksize", type=int, default=3000, help="Chunk size")
    parser.add_argument("--maxchunks", type=int, default=None, help="Max chunks")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--outdir", type=str, default="Outputs", help="Where to put the output files")
    parser.add_argument(
        "--batch", action="store_true", help="Batch mode (no progress bar)"
    )
    parser.add_argument(
        "-e",
        "--executor",
        choices=["local", "wiscjq", "debug"],
        default="local",
        help="How to run the processing",
    )
    parser.add_argument(
        "-l",
        "--lepton",
        choices=["mu","e"],
        help="whether to run mu or e region",
    )

    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    if args.lepton == "e":
        from monoHbb.processor_BCat_Tope import monoHbbProcessor
        print("Processing Top Electron CR")
    elif args.lepton == "mu":
        from monoHbb.processor_BCat_Topmu import monoHbbProcessor
        print("Processing Top Muon CR")

    tstart = time.time()

    print("Running mcGroup {}".format(args.mcGroup))

    if args.executor == "local":
        executor = processor.FuturesExecutor(
            workers=args.workers, status=not args.batch
        )
    elif args.executor == "debug":
        executor = processor.IterativeExecutor(status=not args.batch)
    elif args.executor == "wiscjq":
        from distributed import Client
        from dask_jobqueue import HTCondorCluster

        if args.workers == 1:
            print("Are you sure you want to use only one worker?")

        os.environ["CONDOR_CONFIG"] = "/etc/condor/condor_config"
        _x509_path = move_X509()

        cluster = HTCondorCluster(
            cores=1,
            memory="4 GB",
            disk="20 GB",
            #death_timeout = '60',
            job_extra_directives={
                #"+JobFlavour": '"longlunch"',
                "+JobFlavour": '"workday"',
                "log": "dask_job_output.$(PROCESS).$(CLUSTER).log",
                "output": "dask_job_output.$(PROCESS).$(CLUSTER).out",
                "error": "dask_job_output.$(PROCESS).$(CLUSTER).err",
                "should_transfer_files": "yes",
                "when_to_transfer_output": "ON_EXIT_OR_EVICT",
                "+SingularityImage": '"/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-cc7:latest-py3.10"',
                "Requirements": "HasSingularityJobStart",
                #"request_GPUs" : "1",
                "InitialDir": f'/scratch/{os.environ["USER"]}',
                "transfer_input_files": f'{_x509_path},{os.environ["EXTERNAL_BIND"]}/monoHbb'
            },
            job_script_prologue=[
                "export XRD_RUNFORKHANDLER=1",
                f"export X509_USER_PROXY={_x509_path}",
            ]
        )
        cluster.adapt(minimum=1, maximum=args.workers)
        executor = processor.DaskExecutor(client=Client(cluster), status=not args.batch)

    runner = processor.Runner(
        executor=executor,
        schema=NanoAODSchema,
        chunksize=args.chunksize,
        maxchunks=args.maxchunks,
        #skipbadfiles=True,
        xrootdtimeout=1000,
    )


    if (args.mcGroup == "DataA") | (args.mcGroup == "DataB") | (args.mcGroup == "DataC") | (args.mcGroup == "DataD"):
        print('This is Data!')
        job_fileset = {key: filelist[key] for key in group_mapping[args.mcGroup]}
        output = runner(
            job_fileset,
            treename="Events",
            processor_instance=monoHbbProcessor(isMC=False,era=args.era),
        )
        for dataset_name,dataset_files in job_fileset.items():
            print(dataset_name,":",output[dataset_name]["EventCount"].value)
        
    else:
        print('This is MC!')
        job_fileset = {key: filelist[key] for key in group_mapping[args.mcGroup]}
        output = runner(
            job_fileset,
            treename="Events",
            processor_instance=monoHbbProcessor(isMC=True,era=args.era),
        )
        
        #scale with xsec and luminosity
        for dataset_name,dataset_files in job_fileset.items():
            # Calculate luminosity scale factor
            lumi_sf = (
                crossSections[dataset_name]
                * lumis[args.era]
                / output[dataset_name]["EventCount"].value
            )
            print(dataset_name,":",output[dataset_name]["EventCount"].value)
            for key, obj in output[dataset_name].items():
                if isinstance(obj, hist.Hist):
                    obj *= lumi_sf


    elapsed = time.time() - tstart
    print(f"Total time: {elapsed:.1f} seconds")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    outfile = os.path.join(args.outdir, f"output_{args.mcGroup}_BCatTop{args.lepton}2018_run{timestamp}.coffea")

    util.save(output, outfile)
    print(f"Saved output to {outfile}")
