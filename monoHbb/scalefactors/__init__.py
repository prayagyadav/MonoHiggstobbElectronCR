"""Scale factors for the analysis

This module loads and sets up the scale factor objects
"""
import os.path
from coffea import util
from coffea.btag_tools import BTagScaleFactor
from coffea.jetmet_tools import CorrectedJetsFactory, JECStack
from coffea.lookup_tools import extractor
import pickle

cwd = os.path.dirname(__file__)

#### 2018 ####
# produced using monoHbb/utils/getBtagEfficiencies.py
#taggingEffLookup_18 = pickle.load(open(f"{cwd}/taggingEfficienciesDenseLookup_MCBkgs_UL2018.pkl", 'rb'))
#taggingEffLookupTightWP_18 = pickle.load(open(f"{cwd}/taggingEfficienciesDenseLookupTightWP_MCBkgs_UL2018.pkl", 'rb'))
#taggingEffLookupMediumWP_18 = pickle.load(open(f"{cwd}/taggingEfficienciesDenseLookupMediumWP_MCBkgs_UL2018.pkl", 'rb'))
#taggingEffLookupLooseWP_18 = pickle.load(open(f"{cwd}/taggingEfficienciesDenseLookupLooseWP_MCBkgs_UL2018.pkl", 'rb'))

taggingEffLookupTightWP_18 = pickle.load(open(f"{cwd}/taggingEfficienciesDenseLookupTightWP_MCBkgs_UL2018_10Mevts.pkl", 'rb'))
taggingEffLookupMediumWP_18 = pickle.load(open(f"{cwd}/taggingEfficienciesDenseLookupMediumWP_MCBkgs_UL2018_10Mevts.pkl", 'rb'))
taggingEffLookupLooseWP_18 = pickle.load(open(f"{cwd}/taggingEfficienciesDenseLookupLooseWP_MCBkgs_UL2018_10Mevts.pkl", 'rb'))

#produced using monoHbb/utils/getTriggerEfficiencies.py and monoHbb/utils/plotTrigEff.py
#triggerEffLookup = pickle.load(open(f"{cwd}/triggerSFsDenseLookup_SingleMuon_UL2018.pkl", 'rb'))
#triggerEffLookup_18 = pickle.load(open(f"{cwd}/triggerSFsDenseLookup_SingleMuon_UL2018_wmetfilters.pkl", 'rb')) 
triggerEffLookup_18 = pickle.load(open(f"{cwd}/triggerSFsDenseLookup_WlvJetsDataUL2018_TriggerPath120_NBins20.pkl", 'rb')) #modified Jan 24th


#### 2017 #### 
#triggerEffLookup_17 = pickle.load(open(f"{cwd}/triggerSFsDenseLookup_SingleMuon_UL2017_wmetfilters.pkl", 'rb')) 
triggerEffLookup_17 = pickle.load(open(f"{cwd}/triggerSFsDenseLookup_WlvJetsDataUL2017_TriggerPath120_NBins20.pkl", 'rb')) #modified Jan 24th
#taggingEffLookupTightWP_17 = pickle.load(open(f"{cwd}/taggingEfficienciesDenseLookupTightWP_MCBkgs_UL2017.pkl", 'rb'))
#taggingEffLookupMediumWP_17 = pickle.load(open(f"{cwd}/taggingEfficienciesDenseLookupMediumWP_MCBkgs_UL2017.pkl", 'rb'))
#taggingEffLookupLooseWP_17 = pickle.load(open(f"{cwd}/taggingEfficienciesDenseLookupLooseWP_MCBkgs_UL2017.pkl", 'rb'))
taggingEffLookupTightWP_17 = pickle.load(open(f"{cwd}/taggingEfficienciesDenseLookupTightWP_MCBkgs_UL2017_10Mevts.pkl", 'rb'))
taggingEffLookupMediumWP_17 = pickle.load(open(f"{cwd}/taggingEfficienciesDenseLookupMediumWP_MCBkgs_UL2017_10Mevts.pkl", 'rb'))
taggingEffLookupLooseWP_17 = pickle.load(open(f"{cwd}/taggingEfficienciesDenseLookupLooseWP_MCBkgs_UL2017_10Mevts.pkl", 'rb'))

#Electron Reconstruction SF
ex2 = extractor()
ex2.add_weight_sets([f"ElectronrecoSF EGamma_SF2D {cwd}/egammaEffi_ptAbove20_UL2018.root"])
ex2.finalize()
evaluator = ex2.make_evaluator()
ElectronrecoEffLookup_18 = evaluator['ElectronrecoSF']
ElectronrecoEffLookup_18

#Electron Trigger SF
ex = extractor()
ex.add_weight_sets([f"ElectronTriggerSF EleTriggSF_abseta_pt {cwd}/EleTriggSF.root"])
ex.finalize()
evaluator = ex.make_evaluator()
ElectrontriggerEffLookup_18 = evaluator['ElectronTriggerSF']

