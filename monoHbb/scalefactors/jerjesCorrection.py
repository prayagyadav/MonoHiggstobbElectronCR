import os.path
import correctionlib

from coffea import util
from coffea.jetmet_tools import CorrectedJetsFactory, JECStack
from coffea.lookup_tools import extractor

import numpy as np
import awkward as ak

# For AK4 PFchs Jet:
def get_jet_factory(era):

    cwd = os.path.dirname(__file__)

    if era==2017: 
        jec_tag='Summer19UL17_V5'; jer_tag='Summer19UL17_JRV2'
    elif era==2018: 
        jec_tag='Summer19UL18_V5'; jer_tag='Summer19UL18_JRV2'
    else: 
        raise Exception(f"Error: Unknown era \"{era}\".")

    
    # UL JEC/JER files taken from :
    # https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC
    # https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution

    Jetext = extractor()
    Jetext.add_weight_sets(
        [
            f"* * {cwd}/JEC/"+str(era)+"_UL/"+jec_tag+"_MC_L1FastJet_AK4PFchs.jec.txt",
            f"* * {cwd}/JEC/"+str(era)+"_UL/"+jec_tag+"_MC_L2Relative_AK4PFchs.jec.txt",
            f"* * {cwd}/JEC/"+str(era)+"_UL/"+jec_tag+"_MC_Uncertainty_AK4PFchs.junc.txt",
            f"* * {cwd}/JEC/"+str(era)+"_UL/"+jer_tag+"_MC_PtResolution_AK4PFchs.jr.txt",
            f"* * {cwd}/JEC/"+str(era)+"_UL/"+jer_tag+"_MC_SF_AK4PFchs.jersf.txt",
        ]
    )

    Jetext.finalize()
    Jetevaluator = Jetext.make_evaluator()

    jec_names = [
        jec_tag+"_MC_L1FastJet_AK4PFchs",
        jec_tag+"_MC_L2Relative_AK4PFchs",
        jec_tag+"_MC_Uncertainty_AK4PFchs",
        jer_tag+"_MC_PtResolution_AK4PFchs",
        jer_tag+"_MC_SF_AK4PFchs",
    ]

    jec_inputs = {name: Jetevaluator[name] for name in jec_names}
    jec_stack = JECStack(jec_inputs)

    name_map = jec_stack.blank_name_map
    name_map["JetPt"] = "pt"
    name_map["JetMass"] = "mass"
    name_map["JetEta"] = "eta"
    name_map["JetA"] = "area"
    name_map["ptGenJet"] = "pt_gen"
    name_map["ptRaw"] = "pt_raw"
    name_map["massRaw"] = "mass_raw"
    name_map["Rho"] = "rho"

    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
   
    return jet_factory

# For AK8 PFPuppi Jet:
def get_fatjet_factory(era):

    cwd = os.path.dirname(__file__)

    # UL JEC/JER files taken from :
    # https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC
    # https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution

    ###JER not provided for AK8PFPuppi jet in UL2017, so temporarilty using same files as AK4PFchs (verified this to be the case in 2018)
    
    if era==2017: 
        jec_tag='Summer19UL17_V5'; jer_tag='Summer19UL17_JRV2' 
    elif era==2018: 
        jec_tag='Summer19UL18_V5'; jer_tag='Summer19UL18_JRV2'
    else: 
        raise Exception(f"Error: Unknown era \"{era}\".")

    Jetext = extractor()
    Jetext.add_weight_sets(
        [
            f"* * {cwd}/JEC/"+str(era)+"_UL/"+jec_tag+"_MC_L1FastJet_AK8PFPuppi.jec.txt",
            f"* * {cwd}/JEC/"+str(era)+"_UL/"+jec_tag+"_MC_L2Relative_AK8PFPuppi.jec.txt",
            f"* * {cwd}/JEC/"+str(era)+"_UL/"+jec_tag+"_MC_Uncertainty_AK8PFPuppi.junc.txt",
            f"* * {cwd}/JEC/"+str(era)+"_UL/"+jer_tag+"_MC_PtResolution_AK8PFPuppi.jr.txt",
            f"* * {cwd}/JEC/"+str(era)+"_UL/"+jer_tag+"_MC_SF_AK8PFPuppi.jersf.txt",
        ]        
    )
    Jetext.finalize()
    Jetevaluator = Jetext.make_evaluator()

    jec_names = [
        jec_tag+"_MC_L1FastJet_AK8PFPuppi",
        jec_tag+"_MC_L2Relative_AK8PFPuppi",
        jec_tag+"_MC_Uncertainty_AK8PFPuppi",
        jer_tag+"_MC_PtResolution_AK8PFPuppi",
        jer_tag+"_MC_SF_AK8PFPuppi",
    ]
    
    jec_inputs = {name: Jetevaluator[name] for name in jec_names}
    jec_stack = JECStack(jec_inputs)

    name_map = jec_stack.blank_name_map
    name_map["JetPt"] = "pt"
    name_map["JetMass"] = "mass"
    name_map["JetEta"] = "eta"
    name_map["JetA"] = "area"
    name_map["ptGenJet"] = "pt_gen"
    name_map["ptRaw"] = "pt_raw"
    name_map["massRaw"] = "mass_raw"
    name_map["Rho"] = "rho"

    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
   
    return jet_factory

'''
# Thanks Garvita!
# Reference: https://github.com/CoffeaTeam/coffea/blob/master/coffea/jetmet_tools/CorrectedMETFactory.py#L6 
def get_polar_corrected_MET(met_pt, met_phi, jet_pt, jet_phi, jet_pt_orig):
    sj, cj = np.sin(jet_phi), np.cos(jet_phi)
    x = met_pt * np.cos(met_phi) + ak.sum(
        jet_pt * cj - jet_pt_orig * cj, axis=1
    )
    y = met_pt * np.sin(met_phi) + ak.sum(
        jet_pt * sj - jet_pt_orig * sj, axis=1
    )
    # return corrected MET pt and phi
    return np.hypot(x, y), np.arctan2(y, x)
'''

# Twiki link: https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETRun2Corrections#xy_Shift_Correction_MET_phi_modu
# Corr values from https://lathomas.web.cern.ch/lathomas/METStuff/XYCorrections/XYMETCorrection_withUL17andUL18andUL16.h  
def get_polar_corrected_MET(runera, npv, met_pt, met_phi):
    
    # depends on era and npv, npv = number of primary vertices
    ### for Data ###
    #2018:
    #if(runera=='Data_MET_Run2018A'): 
    if runera.endswith('Run2018A'): 
        xcorr, ycorr = -(0.263733*npv +-1.91115), -(0.0431304*npv +-0.112043)
    #elif(runera=='Data_MET_Run2018B'): 
    elif runera.endswith('Run2018B'):
        xcorr, ycorr = -(0.400466*npv +-3.05914), -(0.146125*npv +-0.533233)
    #elif(runera=='Data_MET_Run2018C'):
    elif runera.endswith('Run2018C'):
        xcorr, ycorr = -(0.430911*npv +-1.42865), -(0.0620083*npv +-1.46021)
    #elif(runera=='Data_MET_Run2018D'):
    elif runera.endswith('Run2018D'):
        xcorr, ycorr = -(0.457327*npv +-1.56856), -(0.0684071*npv +-0.928372)

    #2017:
    #elif(runera=='Data_MET_Run2017B'):
    elif runera.endswith('Run2017B'):
        xcorr, ycorr = -(-0.211161*npv +0.419333), -(0.251789*npv +-1.28089)
    #elif(runera=='Data_MET_Run2017C'):
    elif runera.endswith('Run2017C'):
        xcorr, ycorr = -(-0.185184*npv +-0.164009), -(0.200941*npv +-0.56853)
    #elif(runera=='Data_MET_Run2017D'):
    elif runera.endswith('Run2017D'):
        xcorr, ycorr = -(-0.201606*npv +0.426502), -(0.188208*npv +-0.58313)
    #elif(runera=='Data_MET_Run2017E'):
    elif runera.endswith('Run2017E'):
        xcorr, ycorr = -(-0.162472*npv +0.176329), -(0.138076*npv +-0.250239)
    #elif(runera=='Data_MET_Run2017F'):
    elif runera.endswith('Run2017F'):
        xcorr, ycorr = -(-0.210639*npv +0.72934), -(0.198626*npv +1.028)

        
    ### for MC ###
    elif(runera==2017):
        xcorr, ycorr = -(-0.300155*npv +1.90608), -(0.300213*npv +-2.02232)
    elif(runera==2018):
        xcorr, ycorr = -(0.183518*npv +0.546754), -(0.192263*npv +-0.42121)
        
    #Add met correction factor to uncorrected component
    x = met_pt * np.cos(met_phi) + xcorr
    y = met_pt * np.sin(met_phi) + ycorr

    # return corrected MET pt and phi
    return np.hypot(x, y), np.arctan2(y, x)


# In 2018 data taking, endcaps of HCAL were not functioning (from runs>=319077) (last of 2018B and full of 2018CD). 
# HCAL deposits were not recorded for region: eta=[-3.0, -1.3] and phi=[-1.57, -0.87], hence reco jets pT is under-measured, so MET is over-measured.
# We veto jets in this region to remove fake MET 
# Reference: https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html

def HEM_veto(isMC, nrun, HEMaffected, obj):

    isHEM = ak.ones_like(obj.pt)

    #for affected runs, ensure object is not in HEM region, and for unaffected runs, do nothing
    #For Data we use run number 'nrun' to veto event
    if isMC==False:
        passHEM =  ( 
            ( ((obj.eta <= -3.0) | (obj.eta >= -1.3) | (obj.phi <= -1.57) | (obj.phi >= -0.87)) & (nrun >= 319077) ) 
            | ( (obj.pt > 0) & (nrun < 319077) ) 
        )
    else:
        #For MC, separate the events as affected and not affected; and scale accordingly
        passHEM_1 = ( ( (obj.eta <= -3.0) | (obj.eta >= -1.3) | (obj.phi <= -1.57) | (obj.phi >= -0.87) ) )
        passHEM_0 = ( (obj.pt > 0) )
        passHEM = ak.where(ak.any(HEMaffected, axis=1), passHEM_1, passHEM_0)

    # for a jet in affected region, passHEM will be False
    # we select event if all objects passHEM
    passHEM = (ak.sum(passHEM == False, axis=1)==0)

    return passHEM
def HEM_veto_total_removal(obj):
    '''
    Defined by Prayag
    Totally removes an event if any of the said 'obj' lies in the HEM 15 / 16 region
    '''
    goodobj =  ((obj.eta <= -3.0) | (obj.eta >= -1.3) | (obj.phi <= -1.57) | (obj.phi >= -0.87)) 
    passHEM = ak.all(goodobj,axis=1) 

    return passHEM
