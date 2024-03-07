import time
import hist
import numpy as np
import awkward as ak
import coffea.processor as processor
from coffea.nanoevents.methods import nanoaod, vector
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

from .scalefactors import (
    jerjesCorrection,
    pileupSF,
    triggerEffLookup_18,
    triggerEffLookup_17,
    taggingEffLookupMediumWP_18,
    taggingEffLookupMediumWP_17,
)

from collections import defaultdict
import correctionlib 

from functools import partial
import numba
import pickle
import re
NanoAODSchema.warn_missing_crossrefs = False


#----------------------------------------------------------------------------------------------------
# Scale Factors using Correctionlib, from POG: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG

def Muon_SFs(muon, Wp, syst, era=2018):
    mu, nmu = ak.flatten(muon), ak.num(muon)
    if (era==2017):
        era_arg = "2017_UL"
        MuonZ_evaluator = correctionlib.CorrectionSet.from_file("monoHbb/scalefactors/POG/MUO/2017_UL/muon_Z.json")
    elif (era==2018):
        era_arg = "2018_UL"
        MuonZ_evaluator = correctionlib.CorrectionSet.from_file("monoHbb/scalefactors/POG/MUO/2018_UL/muon_Z.json")
    else:
        raise Exception(f"Error: Unknown era \"{era}\".")
    if (Wp=="Tight"):
        sf_muID = MuonZ_evaluator["NUM_TightID_DEN_genTracks"].evaluate(era_arg, np.array(abs(mu.eta)), np.array(mu.pt), syst)
        sf_muIso = MuonZ_evaluator["NUM_TightRelIso_DEN_TightIDandIPCut"].evaluate(era_arg, np.array(abs(mu.eta)), np.array(mu.pt), syst)
        muID = ak.unflatten(sf_muID, nmu)
        muIso = ak.unflatten(sf_muIso, nmu)
    elif (Wp=="Loose"):
        sf_muID = MuonZ_evaluator["NUM_LooseID_DEN_genTracks"].evaluate(era_arg, np.array(abs(mu.eta)), np.array(mu.pt), syst)
        sf_muIso = MuonZ_evaluator["NUM_LooseRelIso_DEN_LooseID"].evaluate(era_arg, np.array(abs(mu.eta)), np.array(mu.pt), syst)
        muID = ak.unflatten(sf_muID, nmu)
        muIso = ak.unflatten(sf_muIso, nmu)
    return muID, muIso

def Electron_SFs(elec, Wp="Loose", era=2018):
    el, nel = ak.flatten(elec), ak.num(elec)
    if (era==2017):
        era_arg = "2017"
        MuonZ_evaluator = correctionlib.CorrectionSet.from_file("monoHbb/scalefactors/POG/EGM/2017_UL/electron.json")
    elif (era==2018):
        era_arg = "2018"
        MuonZ_evaluator = correctionlib.CorrectionSet.from_file("monoHbb/scalefactors/POG/EGM/2018_UL/electron.json")
    else:
        raise Exception(f"Error: Unknown era \"{era}\".")
    sf_elID = Electron_evaluator["UL-Electron-ID-SF"].evaluate(era_arg, "sf", Wp, np.array(abs(el.eta)), np.array(el.pt))
    elID = ak.unflatten(sf_elID, nel)
    return elID, elIso

def Btag_SFs(Jet, sf_type, Wp, syst="central", era=2018):
    j, nj = ak.flatten(Jet), ak.num(Jet)
    if (era==2017):
        btagSF_evaluator = correctionlib.CorrectionSet.from_file("monoHbb/scalefactors/POG/BTV/2017_UL/btagging.json")
    elif (era==2018):
        btagSF_evaluator = correctionlib.CorrectionSet.from_file("monoHbb/scalefactors/POG/BTV/2018_UL/btagging.json")
    else:
        raise Exception(f"Error: Unknown era \"{era}\".")
    sf = btagSF_evaluator[sf_type].evaluate(syst, Wp, np.array(j.hadronFlavour), np.array(abs(j.eta)), np.array(j.pt))
    return ak.unflatten(sf, nj)

#--------------------------------------------------------------------------------------------------------------
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
        (events.Photon.pt>20) #changed from 15GeV in test4 as per ANv6
        & (abs(events.Photon.eta)<2.5)
        & (events.Photon.cutBased >= 1) #looseID (efficiency of 90%) (which has rho corrected PF photon isolation)
    )    
    return events.Photon[PhotonSelection]

#For MET with other object
def Delta_Phi_func(a,b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi

# -----------------------------------------------------------------------


class monoHbbProcessor(processor.ProcessorABC):

    def __init__(self, isMC=True, era=2018):
        ################################
        # INITIALIZE COFFEA PROCESSOR
        ################################
        ak.behavior.update(nanoaod.behavior)

        self.isMC = isMC
        self.era = era

        # Categories
        dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary dataset")
        cut_axis = hist.axis.Regular(15, 0, 15, name="cut", label="Cut")
        cutflow_axis = hist.axis.StrCategory([], growth=True, name="cutflow",label="Cutflow")
        systematic_axis = hist.axis.StrCategory([], growth=True, name="systematic", label="Systematic Uncertainty")
        met_region_axis =  hist.axis.StrCategory([], growth=True, name="met_region", label="MET region")
        jetInd_axis =  hist.axis.StrCategory([], growth=True, name="jetInd", label="Jet Index")

        # Variables
        cut = hist.axis.Regular(14, 0, 14, name="cut", label=r"Cutflow")
        met_pt_axis = hist.axis.Regular(100, 0, 1000, name="met", label=r"MET [GeV]")
        ht_axis = hist.axis.Regular(100, 0, 5000, name="ht", label=r"HT [GeV]")
        recoil_axis = hist.axis.Regular(100, 0.0, 1000., name="recoil", label=r"Recoil [GeV]")
        num_axis = hist.axis.Regular(10, 0, 10, name="num", label=r"N jets")
        nj_axis = hist.axis.Regular(30, 0, 30, name="nj", label=r"N jets")
        pt_axis = hist.axis.Regular(200, 0.0, 1000, name="pt", label=r"$p_{T}$ [GeV]")
        eta_axis = hist.axis.Regular(300, -3.0, 3.0, name="eta", label=r"$\eta$")
        phi_axis = hist.axis.Regular(200, -4.0, 4.0, name="phi", label=r"$\phi$")
        e_axis = hist.axis.Regular(200, 0.0, 1000., name="energy", label=r"$E$ [GeV]")
        mass_axis = hist.axis.Regular(80, 70., 150., name="mass", label=r"$M$ [GeV]")
        jmass_axis = hist.axis.Regular(250, 0., 500., name="jmass", label=r"$M$ [GeV]")
        dphi_axis = hist.axis.Regular(40, 0., 4., name="dphi", label=r"$\Delta \phi$")
        dr_axis = hist.axis.Regular(50, 0, 5.0, name="dr", label=r"$\Delta R$")
        deta_axis = hist.axis.Regular(50, 0, 5.0, name="deta", label=r"$\Delta \eta$")
        btagDeepB_axis = hist.axis.Regular(50, 0., 1.0, name="btag", label=r"btagDeepB")


        # Accumulator for holding histograms
        self.make_output = lambda: {
            "EventCount": processor.value_accumulator(int),
            "Cutflow_RCat_CRTopmu" : hist.Hist(dataset_axis, cut_axis, label="Events"),

            #split into MET regions
            "Mbb": hist.Hist(dataset_axis, met_region_axis, mass_axis, systematic_axis, label="Events"),

            #kinematics
            "MET_pT" : hist.Hist(dataset_axis, met_pt_axis, systematic_axis, label="Events"),
            "MET_Phi" : hist.Hist(dataset_axis, phi_axis, systematic_axis, label="Events"),
            "Recoil": hist.Hist(dataset_axis, recoil_axis, systematic_axis, label="Events"),
            "Recoil_Phi": hist.Hist(dataset_axis, phi_axis, systematic_axis, label="Events"),
            "HT": hist.Hist(dataset_axis, ht_axis, systematic_axis, label="Events"),

            "Nbj_": hist.Hist(dataset_axis, nj_axis, systematic_axis, label="Events"),
            "N_additionalJets": hist.Hist(dataset_axis, nj_axis, systematic_axis, label="Events"),
            
            "bJet_pT" : hist.Hist(dataset_axis, jetInd_axis, pt_axis, systematic_axis, label="Events"),
            "bJet_Eta" : hist.Hist(dataset_axis, jetInd_axis, eta_axis, systematic_axis, label="Events"),
            "bJet_Phi" : hist.Hist(dataset_axis, jetInd_axis, phi_axis, systematic_axis, label="Events"),
            "bJet_M" : hist.Hist(dataset_axis, jetInd_axis, jmass_axis, systematic_axis, label="Events"),
            "bJet_btagDeepB": hist.Hist(dataset_axis, jetInd_axis, btagDeepB_axis, systematic_axis, label="Events"),

            "dR_bbJets": hist.Hist(dataset_axis, dr_axis, systematic_axis, label="Events"),
            "dPhi_bbJets": hist.Hist(dataset_axis, dphi_axis, systematic_axis, label="Events"),
            "dEta_bbJets": hist.Hist(dataset_axis, deta_axis, systematic_axis, label="Events"),

            "Dijet_pT" : hist.Hist(dataset_axis, pt_axis, systematic_axis, label="Events"),
            "Dijet_Eta" : hist.Hist(dataset_axis, eta_axis, systematic_axis, label="Events"),
            "Dijet_Phi" : hist.Hist(dataset_axis, phi_axis, systematic_axis, label="Events"),
            "Dijet_M" : hist.Hist(dataset_axis, jmass_axis, systematic_axis, label="Events"),

            "Muon_pT": hist.Hist(dataset_axis, pt_axis, systematic_axis, label="Events"),
            "Muon_Eta": hist.Hist(dataset_axis, eta_axis, systematic_axis, label="Events"),
            "Muon_Phi": hist.Hist(dataset_axis, phi_axis, systematic_axis, label="Events"),

            "dPhi_met_Muon" : hist.Hist(dataset_axis, dphi_axis, systematic_axis, label="Events"),
            "dPhi_met_Dijet" : hist.Hist(dataset_axis, dphi_axis, systematic_axis, label="Events"),
            "dPhi_met_bJet" : hist.Hist(dataset_axis, jetInd_axis, dphi_axis, systematic_axis, label="Events"),
        }

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):

        shift_systs = [None]
        if self.isMC:
            shift_systs += ["JESUp", "JESDown", "JERUp", "JERDown"]

        return processor.accumulate(self.process_shift(events, name) for name in shift_systs)


    def process_shift(self, events, shift_syst=None):

        output = self.make_output()
        dataset = events.metadata["dataset"]
        cutflow = defaultdict(int)
        cutflow['TotalEvents'] += len(events)
        #for lumi*xsec normalization. use same value for all shift_syst 
        if shift_syst==None:
            output['EventCount'] += len(events)

        #era = 2017
        #era = 2018
        era = self.era

        # For Data and MC, apply MET Phi modulation correction
        #Twiki link: https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETRun2Corrections#xy_Shift_Correction_MET_phi_modu 
        #Correction factors from https://lathomas.web.cern.ch/lathomas/METStuff/XYCorrections/XYMETCorrection_withUL17andUL18andUL16.h
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

            
        ######################
        # OBJECT SELECTION
        ######################

        ## Electron Corrections: Up to NanoAOD10, residual energy scale and resolution corrections are applied 
        ## to the stored electrons to match the data. Twiki: https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD#Electrons 
        
        ## Muon Corrections: No residual correction to the muon momentum is applied. Twiki: https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD#Muons
        ## Rochester corrections are yet to be applied. Since SR has no leptons, debating if necessary.


        selection = PackedSelection()                
        # muon and electron selections are broken out into standalone functions
        tightMuons, looseMuons = selectMuons(events)
        tightElectrons, looseElectrons = selectElectrons(events)
        leptons = ak.concatenate([looseMuons, looseElectrons], axis=1)
        # tau and photon selections
        taus = selectTaus(events)
        photons = selectPhotons(events)

        jets_ak4 = events.Jet

        #JECs already applied in nanoAOD-v9 (with corr available at time of generation)
        #However we uncorrect and re-correct using latest POG recommendation
        #use rawFactor to get uncorrected jet
        if self.isMC:
            events["Jet", "pt_raw"] = (1 - events.Jet.rawFactor) * events.Jet.pt
            events["Jet", "mass_raw"] = (1 - events.Jet.rawFactor) * events.Jet.mass
            events["Jet", "pt_gen"] = ak.values_astype(ak.fill_none(events.Jet.matched_gen.pt, 0), np.float32)
            events["Jet", "rho"] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, events.Jet.pt)[0]
            events_cache = events.caches[0]
            jet_factory = jerjesCorrection.get_jet_factory(self.era)
            corrected_jets = jet_factory.build(events.Jet, lazy_cache=events_cache)            

            # If processing a jet systematic, we need to update the
            # jets to reflect the jet systematic uncertainty variations
            if shift_syst == "JERUp":
                jets_ak4 = corrected_jets.JER.up 
            elif shift_syst == "JERDown":
                jets_ak4 = corrected_jets.JER.down
            elif shift_syst == "JESUp":
                jets_ak4 = corrected_jets.JES_jes.up
            elif shift_syst == "JESDown":
                jets_ak4 = corrected_jets.JES_jes.down
            else:
                # either nominal or some shift systematic unrelated to jets
                jets_ak4 = corrected_jets
                
        #btag-ak4jet loose and medium WP
        # UL: see https://btv-wiki.docs.cern.ch/ScaleFactors
        if era == 2017:
            btag_WP_loose  = 0.0532
            btag_WP_medium = 0.3040            

        if era == 2018:
            btag_WP_loose = 0.0490
            btag_WP_medium = 0.2783

        #look for jets isolated from electron and muon passing loose selection 
        jetMuMask = ak.all(jets_ak4.metric_table(looseMuons) > 0.4, axis=-1)
        jetEleMask = ak.all(jets_ak4.metric_table(looseElectrons) > 0.4, axis=-1)

        selectjets_ak4 = ((jets_ak4.pt > 30)
                          & (abs(jets_ak4.eta) < 2.5)
                          & (jets_ak4.jetId >= 2)
                          & (jetMuMask)
                          & (jetEleMask)
        )
        AK4jets = jets_ak4[selectjets_ak4]

        #select AK4 jets with btag
        AK4jet_btagWPmed = jets_ak4[selectjets_ak4 & (jets_ak4.btagDeepFlavB > btag_WP_medium) ]
        #OR..
        AK4jets["btagged"] = AK4jets.btagDeepFlavB > btag_WP_medium

        #Photon cancidate away from Elec, Muon, AK4jet in dR 0.4
        PhoElecMask = ak.all(photons.metric_table(looseElectrons) > 0.4, axis=-1)
        PhoMuonMask = ak.all(photons.metric_table(looseMuons) > 0.4, axis=-1)
        PhoAK4jetMask = ak.all(photons.metric_table(AK4jets) > 0.4, axis=-1)
        photons = photons[PhoElecMask & PhoMuonMask & PhoAK4jetMask]
        

        # HEM cleaning
        # veto events if any jet present in HEM affected region
        if era == 2017:
            HEM_cut = np.ones(len(events.MET.pt), dtype=bool)
        elif era == 2018:
            if self.isMC==False:
                HEM_cut = jerjesCorrection.HEM_veto(isMC=self.isMC, nrun=events.run, HEMaffected=False, obj=AK4jets)
            elif self.isMC==True:
                HEM_cut = jerjesCorrection.HEM_veto(isMC=self.isMC, nrun=np.ones(len(events)), HEMaffected=HEM_MCbool, obj=AK4jets)
            

        ##################
        # EVENT VARIABLES
        ##################
        
        event_HT = ak.sum(AK4jets.pt,-1)
        event_bHT = ak.sum(AK4jets.pt[AK4jets.btagged],-1)

        leading_bjet = ak.mask(AK4jet_btagWPmed, ak.num(AK4jet_btagWPmed)>=2)[:,0]
        subleading_bjet = ak.mask(AK4jet_btagWPmed, ak.num(AK4jet_btagWPmed)>=2)[:,1]
        Dijet = leading_bjet + subleading_bjet
        #replace entries of 'None' with '-1'. So an event which doesn't have atleast two bjets will have None as its leading_bjet entry.
        leading_bjet_pt = ak.fill_none(leading_bjet.pt, -1)
        subleading_bjet_pt = ak.fill_none(subleading_bjet.pt, -1)
        Dijet_new_pt = ak.fill_none(Dijet.pt, -1)
        Dijet_new_mass = ak.fill_none(Dijet.mass, -1)

        #Additional AK4jets away from leading and subleading bjets
        ak4j_b1_Mask = ak.all(AK4jets.metric_table(leading_bjet) > 0.4, axis=-1)
        ak4j_b2_Mask = ak.all(AK4jets.metric_table(subleading_bjet) > 0.4, axis=-1)
        NaddAK4jets = ak.sum(ak.fill_none(AK4jets[ak4j_b1_Mask & ak4j_b2_Mask].pt, -1) > 30, axis=-1)

        #Top_mu CR Recoil U
        metpt_muonTopCR = ak.mask(corr_MET_pt, ak.num(tightMuons)==1)
        metphi_muonTopCR = ak.mask(corr_MET_phi, ak.num(tightMuons)==1)
        muon_TopCR = ak.mask(tightMuons, ak.num(tightMuons)==1)
        vec1 = ak.zip( {"x": muon_TopCR.pt*np.cos(muon_TopCR.phi), "y": muon_TopCR.pt*np.sin(muon_TopCR.phi), }, with_name="TwoVector", behavior=vector.behavior,)
        vec2 = ak.zip( {"x": metpt_muonTopCR*np.cos(metphi_muonTopCR), "y": metpt_muonTopCR*np.sin(metphi_muonTopCR), }, with_name="TwoVector", behavior=vector.behavior,)
        Recoil_muTopCR = ak.firsts(vec1.add(vec2))

        
        ######################
        # EVENT SELECTIONS 
        ######################
        evtweight=np.ones(len(events))

        #MET Triggers
        if era==2017:
            selection.add("metTrigger", (events.HLT.PFMETNoMu120_PFMHTNoMu120_IDTight) )
            taggingEffLookup = taggingEffLookupMediumWP_17
            triggerEffLookup = triggerEffLookup_17
            
        elif era==2018:
            selection.add("metTrigger", (events.HLT.PFMETNoMu120_PFMHTNoMu120_IDTight) )
            taggingEffLookup = taggingEffLookupMediumWP_18
            triggerEffLookup = triggerEffLookup_18
            
        #MET Filters
        # Twiki link: https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2#2018_2017_data_and_MC_UL
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
        # Above twiki suggests to not use Flag_BadChargedCandidateFilter and to keep Flag_hfNoisyHitsFilter as optional.
        #selection.add("metFilters", ((events.Flag.goodVertices) & (events.Flag.globalSuperTightHalo2016Filter) & (events.Flag.HBHENoiseFilter) & (events.Flag.HBHENoiseIsoFilter) & (events.Flag.eeBadScFilter) & (events.Flag.EcalDeadCellTriggerPrimitiveFilter) & (events.Flag.BadPFMuonFilter) & (events.Flag.BadPFMuonDzFilter) & (events.Flag.hfNoisyHitsFilter) & (events.Flag.BadChargedCandidateFilter) & (events.Flag.ecalBadCalibFilter)))

        # Event selections for Resolved Category SingleMuon (also written as Topmu) Control Region
        selection.add("HEM_veto", HEM_cut) 
        selection.add("Nphotons=0", ak.num(photons)==0) 
        selection.add("Ntaus=0", ak.num(taus)==0)
        selection.add("NtightMuon=1", ak.num(tightMuons)==1)
        selection.add("NlooseElec=0", ak.num(looseElectrons)==0)
        selection.add("MET>50GeV", corr_MET_pt>50)
        selection.add("Recoil_muTopCR>200GeV", Recoil_muTopCR.pt > 200)
        selection.add("lbjetPT>50GeV", leading_bjet_pt>50)
        selection.add("slbjetPT>30GeV", subleading_bjet_pt>30)
        selection.add("dijetMass[70-150]GeV", (Dijet_new_mass>70) & (Dijet_new_mass<150))
        selection.add("dijetPt100GeV", Dijet_new_pt>100)
        selection.add("NaddAK4j>=1", NaddAK4jets>=1)

        RCat_Topmu_CR = {
            "metTrigger",
            "metFilters",
            "HEM_veto",
            "Ntaus=0",
            "Nphotons=0",
            "NtightMuon=1",
            "NlooseElec=0",
            "MET>50GeV",
            "Recoil_muTopCR>200GeV",
            "lbjetPT>50GeV",
            "slbjetPT>30GeV",
            "dijetMass[70-150]GeV",
            "dijetPt100GeV",
            "NaddAK4j>=1",
        }        
        selection.add("ResolvedCatSels_CR_Topmu", selection.all(*RCat_Topmu_CR))

        #For Mbb distribution
        #RCat recoil regions for CR - Topmu
        selection6 = PackedSelection()
        selection6.add("recoilTopmuCR200_250", (Recoil_muTopCR.pt>200) & (Recoil_muTopCR.pt<=250))
        selection6.add("recoilTopmuCR250_290", (Recoil_muTopCR.pt>250) & (Recoil_muTopCR.pt<=290))
        selection6.add("recoilTopmuCR290_360", (Recoil_muTopCR.pt>290) & (Recoil_muTopCR.pt<=360))
        selection6.add("recoilTopmuCR360_420", (Recoil_muTopCR.pt>360) & (Recoil_muTopCR.pt<=420))
        selection6.add("recoilTopmuCR420_1000", (Recoil_muTopCR.pt>420) & (Recoil_muTopCR.pt<=1000))


        ######################
        # CUTFLOW Histogram
        ######################

        #RCat CR Topmu cutflow
        if shift_syst is None:
            selectionRCatCRTopmu = PackedSelection()
            selectionRCatCRTopmu.add("0", (events.MET.pt>0) )
            selectionRCatCRTopmu.add("1", selection.all("metTrigger") )
            selectionRCatCRTopmu.add("2", (selectionRCatCRTopmu.all("1")) & (selection.all("metFilters")) )
            selectionRCatCRTopmu.add("3", (selectionRCatCRTopmu.all("2")) & (selection.all("Ntaus=0")) )
            selectionRCatCRTopmu.add("4", (selectionRCatCRTopmu.all("3")) & (selection.all("Nphotons=0")) )
            selectionRCatCRTopmu.add("5", (selectionRCatCRTopmu.all("4")) & (selection.all("HEM_veto")) )
            selectionRCatCRTopmu.add("6", (selectionRCatCRTopmu.all("5")) & (selection.all("NtightMuon=1")) & (selection.all("NlooseElec=0")) )
            selectionRCatCRTopmu.add("7", (selectionRCatCRTopmu.all("6")) & (selection.all("MET>50GeV")) )
            selectionRCatCRTopmu.add("8", (selectionRCatCRTopmu.all("7")) & (selection.all("Recoil_muTopCR>200GeV")) )
            selectionRCatCRTopmu.add("9", (selectionRCatCRTopmu.all("8")) & (selection.all("lbjetPT>50GeV")) )
            selectionRCatCRTopmu.add("10", (selectionRCatCRTopmu.all("9")) & (selection.all("slbjetPT>30GeV")) )
            selectionRCatCRTopmu.add("11", (selectionRCatCRTopmu.all("10")) & (selection.all("dijetPt100GeV")) )
            selectionRCatCRTopmu.add("12", (selectionRCatCRTopmu.all("11")) & (selection.all("dijetMass[70-150]GeV")) )
            selectionRCatCRTopmu.add("13", (selectionRCatCRTopmu.all("12")) & (selection.all("NaddAK4j>=1")) )
            bin=0
            for n in selectionRCatCRTopmu.names:
                output["Cutflow_RCat_CRTopmu"].fill(
                    dataset=dataset,
                    cut=np.asarray(bin),
                    weight=selectionRCatCRTopmu.all(n).sum(),
                )
                bin = bin+1


        ####################################
        # EVENT WEIGHTS AND SCALE FACTORS
        ####################################

        # create a processor Weights object, with the same length as the number of events in the chunk
        weights_CR = Weights(len(events))
        weights_CR.add("NoWeight", weight=np.ones(len(events)))
        
        if(self.isMC):

            #########################
            # Pileup Re-weighting
            #########################
            
            puWeight = pileupSF.getPUSF(events.Pileup.nTrueInt,self.era,'nominal')
            puWeight_Up = pileupSF.getPUSF(events.Pileup.nTrueInt,self.era,'up')
            puWeight_Down = pileupSF.getPUSF(events.Pileup.nTrueInt,self.era,'down')
            weights_CR.add("puWeight",weight=puWeight,weightUp=puWeight_Up,weightDown=puWeight_Down,)
            
            #########################
            # Trigger SF
            #########################

            # Given the Recoil or MET value in an event, the event will have an SF value which we read from a lookup table 
            triggerSFWeight = triggerEffLookup(Recoil_muTopCR.pt)
            # Since Data/MC ratio (SF) is within 1% of unity for Recoil>250GeV and within 2% of unity for 200-250GeV
            # Assign 1% systematic uncertainty for events with recoil values >250GeV, and for 200-250GeV use 2% uncertainty
            #triggerSF_err = ak.where(Recoil_muTopCR.pt>250, 0.01, 0.02)

            #With consistent MET-Trigger paths for both years, variation within 1% of unity (2018) and within 1% of 0.98 (2017) #modified January 2024
            triggerSF_err = ak.where(Recoil_muTopCR.pt>200, 0.01, 0.00)

            triggerSFWeight_Up = (triggerSFWeight + triggerSF_err)
            triggerSFWeight_Down = (triggerSFWeight - triggerSF_err)
            weights_CR.add("TriggerSFWeight",weight=triggerSFWeight,weightUp=triggerSFWeight_Up,weightDown=triggerSFWeight_Down,)

            ##################
            # Muon SFs
            ##################
            
            #tightMuon SF
            muID_sf, muIso_sf = Muon_SFs(tightMuons, Wp="Tight", syst="sf", era=era)
            muID_up, muIso_up = Muon_SFs(tightMuons, Wp="Tight", syst="systup", era=era) 
            muID_down, muIso_down = Muon_SFs(tightMuons, Wp="Tight", syst="systdown", era=era) 

            muSF = ak.prod(muID_sf * muIso_sf, axis=-1) # ak.prod(Array, axis=-1): multiplies the SFs for each element in the Array in an event, where Array has product of diff SFs
            muSF_up = ak.prod(muID_up * muIso_up, axis=-1)
            muSF_down = ak.prod(muID_down * muIso_down, axis=-1)

            weights_CR.add("muEffWeight", weight=muSF, weightUp=muSF_up, weightDown=muSF_down)
            
            #########################
            # B-tag efficiency SFs
            #########################
            
            #Scale factors depend on jet flavor
            # "deepJet_comb" for c and b jets; "deepJet_incl" for light(udcg) jets
            jets_cb = AK4jets[(AK4jets.hadronFlavour==4) | (AK4jets.hadronFlavour==5)]
            jets_l = AK4jets[AK4jets.hadronFlavour==0]

            #SFs from Btag_SFs function  
            bJetSF_l = Btag_SFs(jets_l, sf_type="deepJet_incl", Wp="M", syst="central", era=era)
            bJetSF_cb = Btag_SFs(jets_cb, sf_type="deepJet_comb", Wp="M", syst="central", era=era)

            bJetSF_up_l = Btag_SFs(jets_l, sf_type="deepJet_incl", Wp="M", syst="up", era=era)
            bJetSF_up_cb = Btag_SFs(jets_cb, sf_type="deepJet_comb", Wp="M", syst="up", era=era)

            bJetSF_down_l = Btag_SFs(jets_l, sf_type="deepJet_incl", Wp="M", syst="down", era=era)
            bJetSF_down_cb = Btag_SFs(jets_cb, sf_type="deepJet_comb", Wp="M", syst="down", era=era)

            ## MC efficiency lookup
            #create an array of dataset names to read from lookup table
            dataname = ak.ones_like(jets_l.pt)
            dataname = ak.where(dataname==1, str(dataset), '0')
            btagEffi_l = ak.firsts(taggingEffLookup(dataname, jets_l.hadronFlavour, jets_l.pt, abs(jets_l.eta)), axis=2)
            dataname = ak.ones_like(jets_cb.pt)
            dataname = ak.where(dataname==1, str(dataset), '0')
            btagEffi_cb = ak.firsts(taggingEffLookup(dataname, jets_cb.hadronFlavour, jets_cb.pt, abs(jets_cb.eta)), axis=2)

            ## Data efficiency is eff* scale factor
            btagEffiData_l = btagEffi_l * bJetSF_l
            btagEffiData_cb = btagEffi_cb * bJetSF_cb

            btagEffiData_up_l = btagEffi_l * bJetSF_up_l
            btagEffiData_up_cb = btagEffi_cb * bJetSF_up_cb

            btagEffiData_down_l = btagEffi_l * bJetSF_down_l
            btagEffiData_down_cb = btagEffi_cb * bJetSF_down_cb
            
            #Tagging a jet as btag-jet using DeepJet algo at Medium WP
            jets_l["btagged"] = jets_l.btagDeepFlavB > btag_WP_medium
            jets_cb["btagged"] = jets_cb.btagDeepFlavB > btag_WP_medium

            ##probability is the product of all efficiencies of tagged jets, times product of 1-eff for all untagged jets
            # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods#1a_Event_reweighting_using_scale
            #p_MC = ak.prod(Eff) * ak.prod(1-Eff)
            p_MC = ak.prod(btagEffi_l[jets_l.btagged], axis=-1) * ak.prod((1.0-btagEffi_l[np.invert(jets_l.btagged)]), axis=-1) * ak.prod(btagEffi_cb[jets_cb.btagged], axis=-1) * ak.prod((1.0-btagEffi_cb[np.invert(jets_cb.btagged)]), axis=-1)
            p_Data = ak.prod(btagEffiData_l[jets_l.btagged], axis=-1) * ak.prod((1.0-btagEffiData_l[np.invert(jets_l.btagged)]), axis=-1) * ak.prod(btagEffiData_cb[jets_cb.btagged], axis=-1) * ak.prod((1.0-btagEffiData_cb[np.invert(jets_cb.btagged)]), axis=-1)
            p_Data_up = ak.prod(btagEffiData_up_l[jets_l.btagged], axis=-1) * ak.prod((1.0-btagEffiData_up_l[np.invert(jets_l.btagged)]), axis=-1) * ak.prod(btagEffiData_up_cb[jets_cb.btagged], axis=-1) * ak.prod((1.0-btagEffiData_up_cb[np.invert(jets_cb.btagged)]), axis=-1)
            p_Data_down = ak.prod(btagEffiData_down_l[jets_l.btagged], axis=-1) * ak.prod((1.0-btagEffiData_down_l[np.invert(jets_l.btagged)]), axis=-1) * ak.prod(btagEffiData_down_cb[jets_cb.btagged], axis=-1) * ak.prod((1.0-btagEffiData_down_cb[np.invert(jets_cb.btagged)]), axis=-1)
            p_MC = ak.where(p_MC == 0, 1, p_MC)
            btagWeight = p_Data/p_MC
            btagWeight_up = p_Data_up/p_MC
            btagWeight_down = p_Data_down/p_MC

            weights_CR.add('btagWeight', weight=btagWeight, weightUp=btagWeight_up, weightDown=btagWeight_down)            


            #########################
            # L1PreFiringWeight
            #########################

            if era==2017:
                #EvtWeight = (events.L1PreFiringWeight.Nom[selection.all(evtSels)]
                weights_CR.add('L1prefiringWeight', weight=events.L1PreFiringWeight.Nom )
            if era==2018:
                #keep weight=1 i.e no weight factor
                #EvtWeight = np.ones(len(events.MET.pt[selection.all(evtSels)]))
                weights_CR.add('L1prefiringWeight', weight=np.ones(len(events.MET.pt)) )

            ########################
            # Top pT reweighting
            ########################
            if 'TTTo' in dataset:
                top = events.GenPart[(events.GenPart.pdgId == 6) & (events.GenPart.status == 62)]
                antitop = events.GenPart[(events.GenPart.pdgId == -6) & (events.GenPart.status == 62)]

                SF_top = 0.103*np.exp(-0.0118*top.pt) - 0.000134*top.pt + 0.973
                SF_antitop = 0.103*np.exp(-0.0118*antitop.pt) - 0.000134*antitop.pt + 0.973
                topPtWeight = ak.firsts(np.sqrt( SF_top*SF_antitop ))
                weights_CR.add('topPtReweight', weight=topPtWeight)
                
        ###################
        # FILL HISTOGRAMS
        ###################

        ### Add systematics ###

        evtSels = "ResolvedCatSels_CR_Topmu"
        systList = []
        if(self.isMC):         
            if shift_syst is None:
                systList = [
                    "nominal",
                    "muEffWeightUp",
                    "muEffWeightDown",
                    "btagWeightUp",
                    "btagWeightDown",
                    "puWeightUp",
                    "puWeightDown",
                    "TriggerSFWeightUp",
                    "TriggerSFWeightDown",
                ]
            else:
                # if we are currently processing a shift systematic, we don't need to process any of the weight systematics
                # since those are handled in the "nominal" run
                systList = [shift_syst]
        else:
            systList = ["noweight"]
        
        for syst in systList:
            # find the event weight to be used when filling the histograms
            weightSyst = syst

            # in the case of 'nominal', or the jet energy systematics, no weight systematic variation is used (weightSyst=None)
            if syst in ["nominal", "JERUp", "JERDown", "JESUp", "JESDown"]:
                weightSyst = None
            if syst == "noweight": #for isMC=False (i.e. for Data only)
                evtWeight = np.ones(len(events.MET.pt))
            else:
                # call weights.weight() with the name of the systematic to be varied
                evtWeight = weights_CR.weight(weightSyst)
         
   
            ### Filling Histograms ###
            
            # Mbb in various MET regions
            recoil_regions = ["recoilTopmuCR200_250", "recoilTopmuCR250_290", "recoilTopmuCR290_360", "recoilTopmuCR360_420", "recoilTopmuCR420_1000"]                
            for recoilregion in recoil_regions:
                output['Mbb'].fill(dataset=dataset, met_region=recoilregion, mass=np.asarray(Dijet_new_mass[selection.all(evtSels) & selection6.all(recoilregion)]), systematic=syst, weight=evtWeight[selection.all(evtSels) & selection6.all(recoilregion)] )

            #MET pt and phi
            output['MET_pT'].fill(dataset=dataset, met=ak.flatten(corr_MET_pt[selection.all(evtSels)], axis=None), systematic=syst, weight=evtWeight[selection.all(evtSels)])
            output['MET_Phi'].fill(dataset=dataset, phi=ak.flatten(corr_MET_phi[selection.all(evtSels)], axis=None), systematic=syst, weight=evtWeight[selection.all(evtSels)])

            output['HT'].fill(dataset=dataset, ht=event_HT[selection.all(evtSels)], systematic=syst, weight=evtWeight[selection.all(evtSels)])

            output['Recoil'].fill(dataset=dataset, recoil=Recoil_muTopCR.pt[selection.all(evtSels)], systematic=syst, weight=evtWeight[selection.all(evtSels)])
            output['Recoil_Phi'].fill(dataset=dataset, phi=Recoil_muTopCR.phi[selection.all(evtSels)], systematic=syst, weight=evtWeight[selection.all(evtSels)])

            output['N_additionalJets'].fill(dataset=dataset, nj=NaddAK4jets[selection.all(evtSels)], systematic=syst, weight=evtWeight[selection.all(evtSels)])

            # Muon kinematics 
            output['Muon_pT'].fill(dataset=dataset, pt=ak.flatten(tightMuons.pt[selection.all(evtSels)], axis=None), systematic=syst, weight=evtWeight[selection.all(evtSels)])
            output['Muon_Eta'].fill(dataset=dataset, eta=ak.flatten(tightMuons.eta[selection.all(evtSels)], axis=None), systematic=syst, weight=evtWeight[selection.all(evtSels)])
            output['Muon_Phi'].fill(dataset=dataset, phi=ak.flatten(tightMuons.phi[selection.all(evtSels)], axis=None), systematic=syst, weight=evtWeight[selection.all(evtSels)])

            dPhi_met_muon  = abs(Delta_Phi_func(tightMuons.phi, corr_MET_phi)) 
            output['dPhi_met_Muon'].fill(dataset=dataset, dphi=ak.flatten(dPhi_met_muon[selection.all(evtSels)], axis=None), systematic=syst, weight=evtWeight[selection.all(evtSels)])



            #AK4 medium b-tagged jet kinematics
            for jetindex, jet in {"Jet1": leading_bjet, "Jet2": subleading_bjet}.items():

                dPhi_met_bjet = abs(Delta_Phi_func(jet.phi, corr_MET_phi))

                output['bJet_pT'].fill(dataset=dataset, jetInd=jetindex, pt=ak.flatten(jet.pt[selection.all(evtSels)], axis=None), systematic=syst, weight=evtWeight[selection.all(evtSels)])
                output['bJet_Eta'].fill(dataset=dataset, jetInd=jetindex, eta=ak.flatten(jet.eta[selection.all(evtSels)], axis=None), systematic=syst, weight=evtWeight[selection.all(evtSels)])
                output['bJet_Phi'].fill(dataset=dataset, jetInd=jetindex, phi=ak.flatten(jet.phi[selection.all(evtSels)], axis=None), systematic=syst, weight=evtWeight[selection.all(evtSels)])
                output['bJet_M'].fill(dataset=dataset, jetInd=jetindex, jmass=ak.flatten(jet.mass[selection.all(evtSels)], axis=None), systematic=syst, weight=evtWeight[selection.all(evtSels)])
                output['bJet_btagDeepB'].fill(dataset=dataset, jetInd=jetindex, btag=ak.flatten(jet.btagDeepFlavB[selection.all(evtSels)], axis=None), systematic=syst, weight=evtWeight[selection.all(evtSels)])
                output['dPhi_met_bJet'].fill(dataset=dataset, jetInd=jetindex, dphi=ak.flatten(dPhi_met_bjet[selection.all(evtSels)], axis=None), systematic=syst, weight=evtWeight[selection.all(evtSels)])

            dR_bb = leading_bjet.delta_r(subleading_bjet)
            dPhi_bb = leading_bjet.delta_phi(subleading_bjet)
            dEta_bb = abs(leading_bjet.eta - subleading_bjet.eta)

            output['dR_bbJets'].fill(dataset=dataset, dr=ak.flatten(dR_bb[selection.all(evtSels)], axis=None), systematic=syst, weight=evtWeight[selection.all(evtSels)])
            output['dPhi_bbJets'].fill(dataset=dataset, dphi=ak.flatten(dPhi_bb[selection.all(evtSels)], axis=None), systematic=syst, weight=evtWeight[selection.all(evtSels)])
            output['dEta_bbJets'].fill(dataset=dataset, deta=ak.flatten(dEta_bb[selection.all(evtSels)], axis=None), systematic=syst, weight=evtWeight[selection.all(evtSels)])

            #Dijet kinematics
            dPhi_met_dijet = abs(Delta_Phi_func(Dijet.phi, corr_MET_phi))
            output['Dijet_pT'].fill(dataset=dataset, pt=ak.flatten(Dijet.pt[selection.all(evtSels)], axis=None), systematic=syst, weight=evtWeight[selection.all(evtSels)])
            output['Dijet_Eta'].fill(dataset=dataset, eta=ak.flatten(Dijet.eta[selection.all(evtSels)], axis=None), systematic=syst, weight=evtWeight[selection.all(evtSels)])
            output['Dijet_Phi'].fill(dataset=dataset, phi=ak.flatten(Dijet.phi[selection.all(evtSels)], axis=None), systematic=syst, weight=evtWeight[selection.all(evtSels)])
            output['Dijet_M'].fill(dataset=dataset, jmass=ak.flatten(Dijet.mass[selection.all(evtSels)], axis=None), systematic=syst, weight=evtWeight[selection.all(evtSels)])
            output['dPhi_met_Dijet'].fill(dataset=dataset, dphi=ak.flatten(dPhi_met_dijet[selection.all(evtSels)], axis=None), systematic=syst, weight=evtWeight[selection.all(evtSels)])
                    


        return {dataset:output}


    def postprocess(self, accumulator):
        return accumulator
