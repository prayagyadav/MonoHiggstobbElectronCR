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

#from monoHbb.processor_BCat_Tope import monoHbbProcessor
from samples.files_MC18 import fileList_MC_18 as filelist

#define mapping for running MC on Condor
mc_group_mapping = {
    "MCTTbar1l1v": ['TTToSemiLeptonic_18'],
    "MCTTbar0l0v": ['TTToHadronic_18'],
    "MCTTbar2l2v": ['TTTo2L2Nu_18'],
    "MCZ1Jets": ['Z1Jets_NuNu_ZpT_50To150_18', 'Z1Jets_NuNu_ZpT_150To250_18', 'Z1Jets_NuNu_ZpT_250To400_18', 'Z1Jets_NuNu_ZpT_400Toinf_18'],
    "MCZ2Jets": ['Z2Jets_NuNu_ZpT_50To150_18', 'Z2Jets_NuNu_ZpT_150To250_18', 'Z2Jets_NuNu_ZpT_250To400_18', 'Z2Jets_NuNu_ZpT_400Toinf_18'],
    "MCWlvJets": ['WJets_LNu_WPt_100To250_18', 'WJets_LNu_WPt_250To400_18', 'WJets_LNu_WPt_400To600_18', 'WJets_LNu_WPt_600Toinf_18'],
    "MCDYJets": ['DYJets_LL_HT_70To100_18', 'DYJets_LL_HT_100To200_18', 'DYJets_LL_HT_200To400_18', 'DYJets_LL_HT_400To600_18', 'DYJets_LL_HT_600To800_18', 'DYJets_LL_HT_800To1200_18', 'DYJets_LL_HT_1200To2500_18', 'DYJets_LL_HT_2500ToInf_18'],
    "MCDYJets_M4to50": ['DYJets_LL_M4to50_HT_70To100_18', 'DYJets_LL_M4to50_HT_100To200_18', 'DYJets_LL_M4to50_HT_200To400_18', 'DYJets_LL_M4to50_HT_400To600_18', 'DYJets_LL_M4to50_HT_600ToInf_18'],
    "MCSingleTop1": ['ST_tchannel_top_18', 'ST_tchannel_antitop_18'],
    "MCSingleTop2": ['ST_tW_top_18', 'ST_tW_antitop_18'],
    "MCVV": ['WZ_1L1Nu2Q_18', 'WZ_2L2Q_18', 'WZ_3L1Nu_18', 'ZZ_2L2Nu_18', 'ZZ_2L2Q_18', 'ZZ_2Q2Nu_18', 'ZZ_4L_18', 'WW_2L2Nu_18', 'WW_1L1Nu2Q_18'],
    "MCHiggs": ['VBFHToBB_18', 'ttHTobb_18', 'WminusH_HToBB_WToLNu_18', 'WplusH_HToBB_WToLNu_18', 'ggZH_HToBB_ZToNuNu_18', 'ggZH_HToBB_ZToLL_18', 'ZH_HToBB_ZToLL_18', 'ZH_HToBB_ZToNuNu_18'],
    "MCQCD": ['QCD_HT100To200_18', 'QCD_HT200To300_18', 'QCD_HT300To500_18', 'QCD_HT500To700_18', 'QCD_HT700To1000_18', 'QCD_HT1000To1500_18', 'QCD_HT1500To2000_18', 'QCD_HT2000Toinf_18'],
    "MCDYincl": ['DYJetsToLL_inclu_v1_18'],
    "MCDYJetsZpT1": ['DYJets_LL_ZpT_0To50_18', 'DYJets_LL_ZpT_50To100_18'],
    "MCDYJetsZpT2": ['DYJets_LL_ZpT_100To250_18', 'DYJets_LL_ZpT_250To400_18'],
    "MCDYJetsZpT3": ['DYJets_LL_ZpT_400To650_18', 'DYJets_LL_ZpT_650Toinf_18'],
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
        choices=list(mc_group_mapping),
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
            disk="1 GB",
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
        skipbadfiles=True,
        xrootdtimeout=1000,
    )


    if (args.mcGroup == "DataA") | (args.mcGroup == "DataB") | (args.mcGroup == "DataC") | (args.mcGroup == "DataD"):
        print('This is Data!')
        job_fileset = {key: filelist[key] for key in mc_group_mapping[args.mcGroup]}
        output = runner(
            job_fileset,
            treename="Events",
            processor_instance=monoHbbProcessor(isMC=False,era=args.era),
        )
        for dataset_name,dataset_files in job_fileset.items():
            print(dataset_name,":",output[dataset_name]["EventCount"].value)
        
    else:
        print('This is MC!')
        job_fileset = {key: filelist[key] for key in mc_group_mapping[args.mcGroup]}
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
