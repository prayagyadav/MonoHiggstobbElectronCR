#!/usr/bin/env python3
import sys
import os
import argparse
from coffea import util
import coffea.processor as processor
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.nanoevents.methods import nanoaod, vector
from coffea.lookup_tools import extractor, dense_lookup

from monoHbb.utils.crossSections import lumis, crossSections
from monoHbb.scalefactors import jerjesCorrection

from collections import defaultdict
import awkward as ak
import numpy as np
import pickle
import hist


#--------------------------------------------------------------------------------------------------
# Object Selection functions

def selectMuons(events):
    # Twiki link: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonSelection
    # select tight and loose muons
    muonSelectTight = (
        (events.Muon.pt > 30.) & (abs(events.Muon.eta) < 2.4)
        & (events.Muon.isPFcand) 
        & (events.Muon.isTracker | events.Muon.isGlobal)
        & (events.Muon.tightId)
        & (events.Muon.pfRelIso04_all < 0.15)
    )
    muonSelectLoose = (
        (events.Muon.pt > 15) & (abs(events.Muon.eta) < 2.4)
        & (events.Muon.isPFcand)
        & (events.Muon.isTracker | events.Muon.isGlobal)
        & (events.Muon.looseId)
        & (events.Muon.pfRelIso04_all < 0.25)
        #& np.invert(muonSelectTight)
    )
    return events.Muon[muonSelectTight], events.Muon[muonSelectLoose]

def selectElectrons(events):
    # Twiki link: https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2#Offline_selection_criteria_for_V  
    # CutBased Electron ID
    # Impact parameter cuts are explicitly made on dxy and dz
    eleEtaGap = (abs(events.Electron.eta) < 1.4442) | (abs(events.Electron.eta) > 1.566)
    elePassDXY = (abs(events.Electron.eta) < 1.479) & (abs(events.Electron.dxy) < 0.05) | (abs(events.Electron.eta) > 1.479) & (abs(events.Electron.dxy) < 0.1)
    elePassDZ = (abs(events.Electron.eta) < 1.479) & (abs(events.Electron.dz) < 0.1) | (abs(events.Electron.eta) > 1.479) & (abs(events.Electron.dz) < 0.2)
    # select tight and loose electrons
    electronSelectTight = (
        (events.Electron.pt > 40)
        & (abs(events.Electron.eta) < 2.5)
        & (abs(events.Electron.cutBased) >= 4)
        & (eleEtaGap)
        & elePassDXY
        & elePassDZ
    )
    electronSelectLoose = (
        (events.Electron.pt > 10)
        & (abs(events.Electron.eta) < 2.5)
        & (events.Electron.cutBased >= 2)
        & (eleEtaGap)
        & elePassDXY
        & elePassDZ
        #& np.invert(electronSelectTight)
    )
    return events.Electron[electronSelectTight], events.Electron[electronSelectLoose]

def selectTaus(events):
    #Twiki link: https://twiki.cern.ch/twiki/bin/view/CMS/TauIDRecommendationForRun2 
    tauSelection = (
        (events.Tau.pt > 20.) & (abs(events.Tau.eta) < 2.3)
        & (abs(events.Tau.dz) < 0.2)
        & (events.Tau.idDecayModeOldDMs) & (events.Tau.decayMode != 5) & (events.Tau.decayMode != 6) #veto "experimental 2-prong" DMs
        & (events.Tau.idDeepTau2017v2p1VSe >= 8)
        & (events.Tau.idDeepTau2017v2p1VSmu >= 2)
        & (events.Tau.idDeepTau2017v2p1VSjet >= 8)
    )
    return events.Tau[tauSelection]

def selectPhotons(events):
    #Twiki link: https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedPhotonIdentificationRun2
    PhotonSelection = (
        (events.Photon.pt>20) 
        & (abs(events.Photon.eta)<2.5)
        & (events.Photon.cutBased >= 1) #looseID (efficiency of 90%) (which has rho corrected PF photon isolation)
    )    
    return events.Photon[PhotonSelection]

# -----------------------------------------------------------------------

class TriggerEfficiencies(processor.ProcessorABC):
    def __init__(self, isMC=True, era=2018):
        ak.behavior.update(nanoaod.behavior)
        self.isMC = isMC
        self.era = era

        #Axes
        dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary dataset")
        met_axis = hist.axis.Regular(100, 0, 1000, name="met", label=r"MET [GeV]")
        recoil_axis = hist.axis.Regular(100, 0, 1000, name="recoil", label=r"Recoil [GeV]")
        ht_axis = hist.axis.Regular(300, 0, 3000, name="ht", label=r"HT [GeV]")
        bool_EvtSel_axis = hist.axis.Integer(0, 2, name="bool_evtsel", label="Boolean")
        bool_EvtSelnTrigger_axis = hist.axis.Integer(0, 2, name="bool_evtseltrig", label="Boolean")


        # Accumulator for holding histograms
        self.make_output = lambda: {
            "EventCount": processor.value_accumulator(int),            
            "hMET_pT": hist.Hist(dataset_axis, met_axis, label="Events"),
            "hRecoil": hist.Hist(dataset_axis, recoil_axis, label="Events"),  
            "hHT": hist.Hist(dataset_axis, ht_axis, label="Events"),  
            "hBool_EvtSelnTrigger": hist.Hist(dataset_axis, bool_EvtSel_axis, bool_EvtSelnTrigger_axis, recoil_axis, label="Events"),
        }

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):

        output = self.make_output()
        dataset = events.metadata['dataset']
        cutflow = defaultdict(int)
        cutflow['TotalEvents'] += len(events)

        output['EventCount'] += len(events)

        #era = 2018 or 2017
        era = self.era

        #corr_MET_pt, corr_MET_phi = events.MET.pt, events.MET.phi

        # For Data and MC, apply MET Phi modulation correction
        if self.isMC==False:
            corr_MET_pt, corr_MET_phi = jerjesCorrection.get_polar_corrected_MET(runera=dataset, npv=events.PV.npvsGood, met_pt=events.MET.pt, met_phi=events.MET.phi)
        if self.isMC==True:
            corr_MET_pt, corr_MET_phi = jerjesCorrection.get_polar_corrected_MET(runera=era, npv=events.PV.npvsGood, met_pt=events.MET.pt, met_phi=events.MET.phi)

        # For HEM cleaning, separate the MC events into affected and non-affected
        # fraction of lumi from affected 2018 Data runs is 0.647724485 (from brilcalc)
        HEM_MCbool_1 = np.ones(round(len(events.MET.pt)*0.647724485), dtype=bool)
        HEM_MCbool_0 = ~np.ones(round(len(events.MET.pt)*(1.0-0.647724485)), dtype=bool)
        HEM_MCbool = np.concatenate((HEM_MCbool_1, HEM_MCbool_0))
        HEM_MCbool = ak.singletons(HEM_MCbool)


        ###############################
        # Object selections 
        tightMuons, looseMuons = selectMuons(events)
        tightElectrons, looseElectrons = selectElectrons(events)
        leptons = ak.concatenate([looseMuons, looseElectrons], axis=1)
        # tau and photon selections
        taus = selectTaus(events)
        photons = selectPhotons(events)

        # AK4 Jet
        jets = events.Jet

        #look for jets isolated from electron and muon passing loose selection 
        jetMuMask = ak.all(jets.metric_table(looseMuons) > 0.4, axis=-1)
        jetEleMask = ak.all(jets.metric_table(looseElectrons) > 0.4, axis=-1)

        jetSelect = (
            (jets.pt > 30) & (abs(jets.eta) < 2.5) & (jets.jetId >= 2)
            & (jetMuMask) & (jetEleMask)
        )
        Jets = jets[jetSelect]
        leadingJet = jets[(jetSelect) & (jets.pt > 50)] 

        #Photon candidate away from Elec, Muon, AK4jet in dR 0.4
        PhoElecMask = ak.all(photons.metric_table(looseElectrons) > 0.4, axis=-1)
        PhoMuonMask = ak.all(photons.metric_table(looseMuons) > 0.4, axis=-1)
        PhoAK4jetMask = ak.all(photons.metric_table(Jets) > 0.4, axis=-1)
        photons = photons[PhoElecMask & PhoMuonMask & PhoAK4jetMask]      


        # HEM cleaning
        # veto events if any jet present in HEM affected region
        if era == 2017:
            HEM_cut = np.ones(len(events.MET.pt), dtype=bool)
        elif era == 2018:
            if self.isMC==False:
                HEM_cut = jerjesCorrection.HEM_veto(isMC=self.isMC, nrun=events.run, HEMaffected=False, obj=Jets)
            elif self.isMC==True:
                HEM_cut = jerjesCorrection.HEM_veto(isMC=self.isMC, nrun=np.ones(len(events)), HEMaffected=HEM_MCbool, obj=Jets)

        
        ###############################
        # Event Variables 
        # Top_mu CR Recoil U
        metpt_muonTopCR = ak.mask(corr_MET_pt, ak.num(tightMuons)==1)
        metphi_muonTopCR = ak.mask(corr_MET_phi, ak.num(tightMuons)==1)
        muon_TopCR = ak.mask(tightMuons, ak.num(tightMuons)==1)
        vec1 = ak.zip( {"x": muon_TopCR.pt*np.cos(muon_TopCR.phi), "y": muon_TopCR.pt*np.sin(muon_TopCR.phi), }, with_name="TwoVector", behavior=vector.behavior,)
        vec2 = ak.zip( {"x": metpt_muonTopCR*np.cos(metphi_muonTopCR), "y": metpt_muonTopCR*np.sin(metphi_muonTopCR), }, with_name="TwoVector", behavior=vector.behavior,)
        Recoil_muTopCR = ak.firsts(vec1.add(vec2))


        # Jets HT (scalar sum pT of jets)
        event_HT = ak.sum(Jets.pt, axis=-1)

        # Event Selections
        # One tightMuon, atleast one central AK4jet with pT>50, and >=1 additional AK4jet with pT>30, Muon_Trigger

        selection = PackedSelection()
        selection.add("HEM_veto", HEM_cut) 
        selection.add("TightMuon", ak.num(tightMuons)==1)
        selection.add("AK4jet_50GeV", ak.num(leadingJet)>0)
        selection.add("NaddAK4jets>=2", ak.num(Jets)>1)
        if era==2017:
            selection.add("MuonTrigger", (events.HLT.IsoMu27) )
            selection.add("met_Trigger", events.HLT.PFMETNoMu120_PFMHTNoMu120_IDTight)
        if era==2018:
            selection.add("MuonTrigger", (events.HLT.IsoMu24))
            selection.add("met_Trigger", (events.HLT.PFMETNoMu120_PFMHTNoMu120_IDTight) )

        selection.add("metFilters", ( (events.Flag.goodVertices) 
                                      & (events.Flag.globalSuperTightHalo2016Filter)
                                      & (events.Flag.HBHENoiseFilter)
                                      & (events.Flag.HBHENoiseIsoFilter)
                                      & (events.Flag.EcalDeadCellTriggerPrimitiveFilter)
                                      & (events.Flag.BadPFMuonFilter)
                                      & (events.Flag.BadPFMuonDzFilter)
                                      & (events.Flag.eeBadScFilter)
                                      & (events.Flag.ecalBadCalibFilter) )
        ) 

        evtSels = {
            "HEM_veto",
            "metFilters",
            "TightMuon",
            "AK4jet_50GeV",
            "NaddAK4jets>=2",
            "MuonTrigger",
        }
        selection.add("EvtSelection", selection.all(*evtSels))

        Recoil = ak.fill_none(abs(Recoil_muTopCR.pt), -10)
        
        # Fill Histograms
        output["hMET_pT"].fill(
            dataset=dataset,
            met=corr_MET_pt[selection.all("EvtSelection")],
        )
        output["hRecoil"].fill(
            dataset=dataset,
            recoil=Recoil[selection.all("EvtSelection")],
        )
        output["hHT"].fill(
            dataset=dataset,
            ht=event_HT[selection.all("EvtSelection")],
        )
        output["hBool_EvtSelnTrigger"].fill(
            dataset=dataset,
            recoil=Recoil,
            bool_evtsel=selection.all("EvtSelection"),
            bool_evtseltrig=(selection.all("EvtSelection") & selection.all("met_Trigger")),
        )
        
        return {dataset:output}

    def postprocess(self, accumulator):
        return accumulator


#------------------------------------------------------------------------------------

mc_group_mapping = {

    "MCTTbar1l1v_18": ['TTToSemiLeptonic_18'],
    "MCTTbar0l0v_18": ['TTToHadronic_18'],
    "MCTTbar2l2v_18": ['TTTo2L2Nu_18'],
    "MCSingleTop1_18": ['ST_tchannel_top_18', 'ST_tchannel_antitop_18'],
    "MCSingleTop2_18": ['ST_tW_top_18', 'ST_tW_antitop_18'],
    "MCWlvJets_18": ['WJets_LNu_WPt_100To250_18', 'WJets_LNu_WPt_250To400_18', 'WJets_LNu_WPt_400To600_18', 'WJets_LNu_WPt_600Toinf_18'],
    "MCVV_18": ['WZ_1L1Nu2Q_18', 'WZ_2L2Q_18', 'WZ_3L1Nu_18', 'ZZ_2L2Nu_18', 'ZZ_2L2Q_18', 'ZZ_2Q2Nu_18', 'ZZ_4L_18', 'WW_2L2Nu_18', 'WW_1L1Nu2Q_18'],
    "MCHiggs_18": ['VBFHToBB_18', 'ttHTobb_18', 'WminusH_HToBB_WToLNu_18', 'WplusH_HToBB_WToLNu_18', 'ggZH_HToBB_ZToNuNu_18', 'ggZH_HToBB_ZToLL_18', 'ZH_HToBB_ZToLL_18', 'ZH_HToBB_ZToNuNu_18'],
    "MCQCD_18": ['QCD_HT100To200_18', 'QCD_HT200To300_18', 'QCD_HT300To500_18', 'QCD_HT500To700_18', 'QCD_HT700To1000_18', 'QCD_HT1000To1500_18', 'QCD_HT1500To2000_18', 'QCD_HT2000Toinf_18'],
    "MCDYJetsZpT1_18": ['DYJets_LL_ZpT_0To50_18', 'DYJets_LL_ZpT_50To100_18'],
    "MCDYJetsZpT2_18": ['DYJets_LL_ZpT_100To250_18', 'DYJets_LL_ZpT_250To400_18'],
    "MCDYJetsZpT3_18": ['DYJets_LL_ZpT_400To650_18', 'DYJets_LL_ZpT_650Toinf_18'],

    "MCTTbar1l1v_17": ['TTToSemiLeptonic_17'],
    "MCTTbar0l0v_17": ['TTToHadronic_17'],
    "MCTTbar2l2v_17": ['TTTo2L2Nu_17'],
    "MCSingleTop1_17": ['ST_tchannel_top_17', 'ST_tchannel_antitop_17'],
    "MCSingleTop2_17": ['ST_tW_top_17', 'ST_tW_antitop_17'],
    "MCWlvJets_17": ['WJets_LNu_WPt_100To250_17', 'WJets_LNu_WPt_250To400_17', 'WJets_LNu_WPt_400To600_17', 'WJets_LNu_WPt_600Toinf_17'],
    "MCVV_17": ['WZ_1L1Nu2Q_17', 'WZ_2L2Q_17', 'WZ_3L1Nu_17', 'ZZ_2L2Nu_17', 'ZZ_2L2Q_17', 'ZZ_2Q2Nu_17', 'ZZ_4L_17', 'WW_2L2Nu_17', 'WW_1L1Nu2Q_17'],
    "MCHiggs_17": ['VBFHToBB_17', 'ttHTobb_17', 'WminusH_HToBB_WToLNu_17', 'WplusH_HToBB_WToLNu_17', 'ggZH_HToBB_ZToNuNu_17', 'ggZH_HToBB_ZToLL_17', 'ZH_HToBB_ZToLL_17', 'ZH_HToBB_ZToNuNu_17'],
    "MCQCD_17": ['QCD_HT100To200_17', 'QCD_HT200To300_17', 'QCD_HT300To500_17', 'QCD_HT500To700_17', 'QCD_HT700To1000_17', 'QCD_HT1000To1500_17', 'QCD_HT1500To2000_17', 'QCD_HT2000Toinf_17'],
    "MCDYJetsZpT1_17": ['DYJets_LL_ZpT_0To50_17', 'DYJets_LL_ZpT_50To100_17'],
    "MCDYJetsZpT2_17": ['DYJets_LL_ZpT_100To250_17', 'DYJets_LL_ZpT_250To400_17'],
    "MCDYJetsZpT3_17": ['DYJets_LL_ZpT_400To650_17', 'DYJets_LL_ZpT_650Toinf_17'],

    "Data_18": ['Data_MET_Run2018A', 'Data_MET_Run2018B', 'Data_MET_Run2018C', 'Data_MET_Run2018D'],

    "Data_17": ['Data_MET_Run2017B', 'Data_MET_Run2017C', 'Data_MET_Run2017D', 'Data_MET_Run2017E', 'Data_MET_Run2017F'],
}


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
#parser.add_argument("--isMC", type=bool, default=True, help="MC or Data (isMC True or False)")
parser.add_argument("--chunksize", type=int, default=3000, help="Chunk size")
parser.add_argument("--maxchunks", type=int, default=None, help="Max chunks")
parser.add_argument("--workers", type=int, default=1, help="Number of workers")

args = parser.parse_args()


isMC=False
if (args.mcGroup == "Data_18"):
    from samples.files_Data18 import fileList_Data_18 as filelist
elif (args.mcGroup == "Data_17"):
    from samples.files_Data17 import filesetData_17_all as filelist 
else:
    isMC=True
    if(args.era==2018):
        from samples.files_MC18 import fileList_MC_18 as filelist
    elif(args.era==2017):
        from samples.files_MC17 import fileset_MC_Bkgs as filelist

listOfFiles = {key: filelist[key] for key in mc_group_mapping[args.mcGroup]}

runner = processor.Runner(
    executor=processor.FuturesExecutor(workers=args.workers),
    schema=NanoAODSchema,
    chunksize=args.chunksize,
    maxchunks=args.maxchunks,
    skipbadfiles=True,
    xrootdtimeout=1000,
)
output = runner(
    listOfFiles,
    treename="Events",
    processor_instance=TriggerEfficiencies(isMC=isMC, era=args.era),
)


#Luminosity Scaling for MC
for dataset_name,dataset_files in listOfFiles.items():
    print(dataset_name,":",output[dataset_name]["EventCount"].value)
    lumi_sf = 1.000
    if(isMC):
        lumi_sf = (
            crossSections[dataset_name]
            * lumis[args.era]
            / output[dataset_name]["EventCount"].value
        )
        for key, obj in output[dataset_name].items():
            if isinstance(obj, hist.Hist):
                obj *= lumi_sf

util.save(output, f"monoHbb/efficiencies/TriggerEfficiency_{args.mcGroup}_Path120.coffea")
