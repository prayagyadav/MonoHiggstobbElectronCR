'''
Runner adapted for Shivani's codes.

'''
if __name__=="__main__":
    from coffea import processor
    import argparse
    from coffea import util
    from coffea.nanoevents import NanoAODSchema , NanoEventsFactory
    from coffea.lumi_tools import LumiMask
    import awkward as ak
    import numba
    from monoHbb.utils.crossSections import lumis, crossSections
    import json
    import rich
    import numpy as np
    import os
    import shutil
    import logging

    ##############################
    # Define the terminal inputs #
    ##############################

#     group_mapping = {
#     "DataA": ['Data_MET_Run2018A'],
#     "DataB": ['Data_MET_Run2018B'],
#     "DataC": ['Data_MET_Run2018C'],
#     "DataD": ['Data_MET_Run2018D'],
# }

    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--executor",
        choices=["futures","condor", "dask"],
        help="Enter where to run the file : futures(local) or dask(local) or condor",
        default="futures",
        type=str
    )

    #  parser.add_argument(
    #      "-k",
    #     "--mcGroup",
    #     choices=list(group_mapping),
    #     help="Name of process to run",
    #     type=str
    # )

    parser.add_argument(
        "-k",
        "--keymap",
        # choices=[
        #     "MET_Run2018",
        #     "ZJets_NuNu",
        #     "TTToSemiLeptonic",
        #     "TTTo2L2Nu",
        #     "TTToHadronic",
        #     "WJets_LNu",
        #     "DYJets_LL",
        #     "VV",
        #     "QCD",
        #     "ST"
        #     ],
        choices=[
            "DataA",
            "DataB",
            "DataC",
            "DataD",
            "MCTTbar1l1v",
            "MCTTbar0l0v",
            "MCTTbar2l2v",
            "MCZ1Jets",
            "MCZ2Jets",
            "MCWlvJets",
            "MCDYJets",
            "MCDYJets_M4to50",
            "MCSingleTop1",
            "MCSingleTop2",
            "MCVV",
            "MCHiggs",
            "MCQCD",
            "MCDYincl",
            "MCDYJetsZpT1",
            "MCDYJetsZpT2",
            "MCDYJetsZpT3"
        ],
        help="Enter which dataset to run: example MET_Run2018 , ZJets_Nu_Nu etc.",
        type=str
    )

    parser.add_argument(
        "-c",
        "--chunk_size",
        help="Enter the chunksize; by default 100k",
        type=int ,
        default=100000
        )
    parser.add_argument(
        "-m",
        "--max_chunks",
        help="Enter the number of chunks to be processed; by default None ie full dataset",
        type=int
        )
    parser.add_argument(
        "-w",
        "--workers",
        help="Enter the number of workers to be employed for processing in local; by default 4",
        type=int ,
        default=4
        )
    parser.add_argument(
        "-f",
        "--files",
        help="Enter the number of files to be processed",
        type=int
        )
    parser.add_argument(
        "--begin",
        help="Begin Sequential execution from file number (inclusive)",
        type=int
    )
    parser.add_argument(
        "--end",
        help="End Sequential execution from file number 'int'",
        type=int
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="Outputs",
        help="Where to put the output files"
    )
    parser.add_argument(
        "--lepton",
        help="mu or e : muon CR or electron CR to run",
        choices=["mu","e"],
        type=str
    )
    parser.add_argument(
        "--redirector",
        help="Choose a non-default redirector, use commonfs for local run in the cmslab-workstation server",
        choices=["fnal","infn","kisti","wisc","unl","commonfs"],
        type=str,
        default="fnal"
    )
    inputs = parser.parse_args()


    mc_group_mapping = {
    "DataA": ['Data_MET_Run2018A'],
    "DataB": ['Data_MET_Run2018B'],
    "DataC": ['Data_MET_Run2018C'],
    "DataD": ['Data_MET_Run2018D'],
    "MCTTbar1l1v": ['TTToSemiLeptonic_18'],
    "MCTTbar0l0v": ['TTToHadronic_18'],
    "MCTTbar2l2v": ['TTTo2L2Nu_18'],
    "MCZ1Jets": [
        'Z1Jets_NuNu_ZpT_50To150_18',
        'Z1Jets_NuNu_ZpT_150To250_18',
        'Z1Jets_NuNu_ZpT_250To400_18',
        'Z1Jets_NuNu_ZpT_400Toinf_18'],
    "MCZ2Jets": [
        'Z2Jets_NuNu_ZpT_50To150_18',
        'Z2Jets_NuNu_ZpT_150To250_18',
        'Z2Jets_NuNu_ZpT_250To400_18',
        'Z2Jets_NuNu_ZpT_400Toinf_18'],
    "MCWlvJets": [
        'WJets_LNu_WPt_100To250_18',
        'WJets_LNu_WPt_250To400_18',
        'WJets_LNu_WPt_400To600_18',
        'WJets_LNu_WPt_600Toinf_18'],
    "MCDYJets": [
        'DYJets_LL_HT_70To100_18',
        'DYJets_LL_HT_100To200_18', 
        'DYJets_LL_HT_200To400_18',
        'DYJets_LL_HT_400To600_18',
        'DYJets_LL_HT_600To800_18',
        'DYJets_LL_HT_800To1200_18',
        'DYJets_LL_HT_1200To2500_18',
        'DYJets_LL_HT_2500ToInf_18'],
    "MCDYJets_M4to50": [
        'DYJets_LL_M4to50_HT_70To100_18', 
        'DYJets_LL_M4to50_HT_100To200_18',
        'DYJets_LL_M4to50_HT_200To400_18',
        'DYJets_LL_M4to50_HT_400To600_18',
        'DYJets_LL_M4to50_HT_600ToInf_18'],
    "MCSingleTop1": [
        'ST_tchannel_top_18',
        'ST_tchannel_antitop_18'],
    "MCSingleTop2": [
        'ST_tW_top_18',
        'ST_tW_antitop_18'],
    "MCVV": [
        'WZ_1L1Nu2Q_18',
        'WZ_2L2Q_18',
        'WZ_3L1Nu_18',
        'ZZ_2L2Nu_18',
        'ZZ_2L2Q_18',
        'ZZ_2Q2Nu_18',
        'ZZ_4L_18',
        'WW_2L2Nu_18',
        'WW_1L1Nu2Q_18'],
    "MCHiggs": [
        'VBFHToBB_18',
        'ttHTobb_18',
        'WminusH_HToBB_WToLNu_18',
        'WplusH_HToBB_WToLNu_18',
        'ggZH_HToBB_ZToNuNu_18',
        'ggZH_HToBB_ZToLL_18',
        'ZH_HToBB_ZToLL_18',
        'ZH_HToBB_ZToNuNu_18'],
    "MCQCD": [
        'QCD_HT100To200_18',
        'QCD_HT200To300_18',
        'QCD_HT300To500_18',
        'QCD_HT500To700_18',
        'QCD_HT700To1000_18',
        'QCD_HT1000To1500_18',
        'QCD_HT1500To2000_18',
        'QCD_HT2000Toinf_18'],
    "MCDYincl": ['DYJetsToLL_inclu_v1_18'],
    "MCDYJetsZpT1": [
        'DYJets_LL_ZpT_0To50_18',
        'DYJets_LL_ZpT_50To100_18'],
    "MCDYJetsZpT2": [
        'DYJets_LL_ZpT_100To250_18',
        'DYJets_LL_ZpT_250To400_18'],
    "MCDYJetsZpT3": [
        'DYJets_LL_ZpT_400To650_18',
        'DYJets_LL_ZpT_650Toinf_18'],
}

    if not os.path.exists(inputs.outdir):
        os.makedirs(inputs.outdir)

    if inputs.lepton == "e":
        from monoHbb.processor_BCat_Tope import monoHbbProcessor
        print("Processing Top Electron CR")
    elif inputs.lepton == "mu":
        from monoHbb.processor_BCat_Topmu import monoHbbProcessor
        print("Processing Top Muon CR")
        
    ismc = True
    if inputs.keymap == "MET_Run2018":
        ismc = False

    Era = 2018
        
    print("Running dataset {}".format(inputs.keymap))

    class Loadfileset():
        def __init__(self, jsonfilename) :
            with open(jsonfilename) as f :
                self.handler = json.load(f)
    
        
        def Show(self , verbosity=1):
            if verbosity==1 :
                for key, value in self.handler.items() :
                    rich.print(key+" : ", list(value.keys()))
            elif verbosity==2 :
                for key, value in self.handler.items() :
                    rich.print(key+" : ", list(value.keys()), "\n")
                    for subkey , subvalue in value.items() :
                        rich.print("\t"+subkey+" : ")
                        for file in subvalue :
                            rich.print("\t", file)
            elif verbosity==3 :
                for key, value in self.handler.items() :
                    rich.print(key+" : ", list(value.keys()), "\n")
                    for subkey , subvalue in value.items() :
                        rich.print("\t"+subkey+" : ")
                        for subsubkey, subsubvalue in subvalue.items() :
                            try :
                                for file in subvalue :
                                    rich.print("\t", file)
                            except:
                                rich.print("\t"+subsubkey+" : ")
        
        def getFileset(self, mode ,superkey, key, redirector ) :
            if redirector=="fnal":
                redirector_string = "root://cmsxrootd.fnal.gov//"
            elif redirector=="infn":
                redirector_string = "root://xrootd-cms.infn.it//"
            elif redirector=="wisc":
                redirector_string = "root://pubxrootd.hep.wisc.edu//"
            elif redirector=="unl":
                redirector_string = "root://xrootd-local.unl.edu:1094//"
            elif redirector=="kisti":
                redirector_string = "root://cms-xrdr.sdfarm.kr:1094//xrd//"
            raw_fileset = self.handler[mode][superkey][key] 
            requested_fileset = {superkey : [redirector_string+filename for filename in raw_fileset]}
            return requested_fileset
        
        def getraw(self):
            #load the raw dictionary
            full_fileset = self.handler
            return full_fileset
    def getDataset(keymap, load=True, dict = None, files=None, begin=0, end=0, mode = "sequential"):
        #Warning : Never use 'files' with 'begin' and 'end'
        fileset = Loadfileset("samples/2018_samples.json")
        fileset_dict = fileset.getraw()
        MCmaps =[
            "DataA",
            "DataB",
            "DataC",
            "DataD",
            "MCTTbar1l1v",
            "MCTTbar0l0v",
            "MCTTbar2l2v",
            "MCZ1Jets",
            "MCZ2Jets",
            "MCWlvJets",
            "MCDYJets",
            "MCDYJets_M4to50",
            "MCSingleTop1",
            "MCSingleTop2",
            "MCVV",
            "MCHiggs",
            "MCQCD",
            "MCDYincl",
            "MCDYJetsZpT1",
            "MCDYJetsZpT2",
            "MCDYJetsZpT3"
        ]
        
        runnerfileset = buildFileset(fileset_dict[keymap],inputs.redirector)
        flat_list={}
        flat_list[keymap] = []
    
        if mode == "sequential":
            if end - begin < 0:
                print("Invalid begin and end values.\nFalling back to full dataset...")
                outputfileset = runnerfileset
            else:
                #indexer
                index={}
                i = 1
                for key in runnerfileset.keys() :
                    index[key] = []
                    for file in runnerfileset[key] :
                        index[key].append(i)
                        i += 1
    
                accept = np.arange(begin,end+1,1)
                print(accept)
                temp = {}
                for key in runnerfileset.keys() :
                    temp[key] = []
                    for i in range(len(runnerfileset[key])) :
                        if index[key][i] in accept :
                            temp[key].append(runnerfileset[key][i])

                outputfileset = temp
        elif mode == "divide" :
            if files == None:
                print("Invalid number of files.\nFalling back to full dataset...")
                outputfileset = runnerfileset
            else:
                # Divide the share of files from all the 8 categories of ZJets_NuNu
                file_number = 0
                while file_number < files :
                    for key in runnerfileset.keys():
                        if file_number >= files :
                            break
                        flat_list[keymap] += [runnerfileset[key][0]]
                        runnerfileset[key] = runnerfileset[key][1:]
                        file_number += 1
                outputfileset = {keymap : flat_list[keymap]}
        else:
            print("Invalid mode of operation", mode)
            raise KeyError
        
        print("Running ", np.array([len(value) for value in outputfileset.values()]).sum(), " files...")
        return outputfileset
    def buildFileset(dict , redirector):
        '''
        To return a run-able dict with the appropriate redirector.
        Please input a dictionary which is only singly-nested
        '''
        redirectors = {
            "fnal": "root://cmsxrootd.fnal.gov//",
            "infn": "root://xrootd-cms.infn.it//",
            "wisc": "root://pubxrootd.hep.wisc.edu//",
            "unl":  "root://xrootd-local.unl.edu:1094/",
            "kisti": "root://cms-xrdr.sdfarm.kr:1094//xrd/",
            "hdfs": "/hdfs",
            "commonfs": "/commonfs"
    
        }
    
        if (redirector=="fnal") | (redirector==1) :
            redirector_string = redirectors["fnal"]
        elif (redirector=="infn") | (redirector==2) :
            redirector_string = redirectors["infn"]
        elif (redirector=="wisc") | (redirector==3):
            redirector_string = redirectors["wisc"]
        elif (redirector=="unl") | (redirector==4):
            redirector_string = redirectors["unl"]
        elif (redirector=="kisti") | (redirector==5):
            redirector_string = redirectors["kisti"]
        elif (redirector=="hdfs") | (redirector==6):
            redirector_string = redirectors["hdfs"]
        elif (redirector=="commonfs") | (redirector==7):
            redirector_string = redirectors["commonfs"]
    
        temp = dict 
        output = {}
        for key in temp.keys() :
            try :
                g = temp[key]
                if isinstance(g,list):
                    templist = []
                    for filename in g :
                        filename = filename[filename.find("/store/") :]
                        templist.append(redirector_string+filename)
                    output[key] = templist
            except :
                raise KeyError
        return output

        
    #def zip_files(list_of_files):
    #    if not os.path.exists("temp_folder"):
    #        os.makedirs("temp_folder")
    #    for file in list_of_files :
    #        shutil.copy(file,"temp_folder")
    #    archive_name = "helper_files"
    #    shutil.make_archive(archive_name,"zip","temp_folder")
    #    shutil.rmtree("temp_folder")
    #    return archive_name+".zip"

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
    
    def runCondor(cores=1, memory="2 GB", disk="1 GB", death_timeout = '60', workers=4):
        from distributed import Client
        from dask_jobqueue import HTCondorCluster
    
        os.environ["CONDOR_CONFIG"] = "/etc/condor/condor_config"
        _x509_path = move_X509()
    
        cluster = HTCondorCluster(
            cores=cores,
            memory=memory,
            disk=disk,
            death_timeout = death_timeout,
            job_extra_directives={
                #"+JobFlavour": '"espresso"', # 20 minutes
                #"+JobFlavour": '"microcentury"' , # 1 hour
                "+JobFlavour": '"longlunch"' , # 2 hours
                #"+JobFlavour": '"workday"' , # 8 hours
                #"+JobFlavour": '"tomorrow"' , # 1 day
                #"+JobFlavour": '"testmatch"' , # 3 days
                #"+JobFlavour": '"nextweek"' , # 1 week
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
        cluster.adapt(minimum=1, maximum=workers)
        executor = processor.DaskExecutor(client=Client(cluster))
        return executor, Client(cluster)

    
    #For futures execution
    if inputs.executor == "futures" :
        files = getDataset(keymap=inputs.keymap,load=True, mode="sequential", begin=inputs.begin, end=inputs.end)
        futures_run = processor.Runner(
            executor = processor.FuturesExecutor(workers=inputs.workers),
            schema=NanoAODSchema,
            chunksize= inputs.chunk_size ,
            maxchunks= inputs.max_chunks,
            xrootdtimeout=120
        )
        Output = futures_run(
            files,
            "Events",
            processor_instance=monoHbbProcessor(isMC=ismc,era=Era)
        )
    #For dask execution
    elif inputs.executor == "dask" :
        print("WARNING: This feature is still in development!\nAttemping to run nevertheless ...")
        from dask.distributed import Client , LocalCluster
        cluster = LocalCluster()
        client = Client(cluster)
        cluster.scale(inputs.workers)

        #client.upload_file(zip_files(
        #    [
        #        #"Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt",
        #        "snippets.py",
        #        "processor_Top.py"
        #        ]
        #    )
        #    )
        #client.upload_file('helper_files.zip')

        with open("samples/2018_samples.json") as f: #load the fileset
            filedict = json.load(f)
        files = getDataset(
            keymap=inputs.keymap,
            load=False ,
            dict=filedict,
            mode="sequential",
            begin=inputs.begin,
            end=inputs.end
            )
        print(files)
        #files 
        dask_run = processor.Runner(
            executor = processor.DaskExecutor(client=client),
            schema=NanoAODSchema,
            chunksize= inputs.chunk_size ,
            maxchunks= inputs.max_chunks
        )
        Output = dask_run(
            files,
            "Events",
            processor_instance=monoHbbProcessor(isMC=ismc,era=Era)
        )
    
    #For condor execution
    elif inputs.executor == "condor" :
        #Create a console log for easy debugging 
        logging.basicConfig(
            format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
            level=logging.WARNING,
        )
        print("Preparing to run at condor...\n")
        executor , client = runCondor(cores=1,memory="4 GB",disk="4 GB",workers=inputs.workers)
        print("Executor and Client Obtained")
        #client.upload_file(
        #    zip_files(
        #        [
        #            #"Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt",
        #            "snippets.py",
        #            "processor_Top.py"
        #        ]
        #    )
        #)
        #client.upload_file('helper_files.zip')

        with open("samples/2018_samples.json") as f: #load the fileset
            filedict = json.load(f)
    
        files = getDataset(
            keymap=inputs.keymap,
            load=False ,
            dict=filedict,
            mode="sequential",
            begin=inputs.begin,
            end=inputs.end
            )
    
        runner = processor.Runner(
            executor=executor,
            schema=NanoAODSchema,
            chunksize=inputs.chunk_size,
            maxchunks=inputs.max_chunks,
            xrootdtimeout=300,
        )
        print("Starting the workers...\n")
        Output = runner(
            files,
            treename="Events",
            processor_instance=monoHbbProcessor(isMC=ismc,era=Era)
        )

    #################################
    # Create the output file #
    #################################

    outfile = os.path.join(inputs.outdir, f"output_{inputs.keymap}_BCatTop{inputs.lepton}2018.coffea")

    util.save(output, outfile)
    print(f"Saved output to {outfile}")

    # print("Output produced")
    # try :
    #     output_file = f"CR_{inputs.cat}_Top_{inputs.lepton}_{inputs.keymap}_from_{inputs.begin}_to_{inputs.end}.coffea"
    #     pass
    # except :
    #     output_file = f"CR_{inputs.cat}_Top_{inputs.lepton}_{inputs.keymap}.coffea"
    # print("Saving the output to : " , output_file)
    # util.save(output= Output, filename="coffea_files/electron/v2/"+output_file)
    # print(f"File {output_file} saved.")
    # print("Execution completed.")

