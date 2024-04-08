from coffea import util
import awkward as ak
import numpy as np
import hist
import matplotlib.pyplot as plt
import mplhep
from plotting import GroupBy
from coffea.processor import accumulate
from coffea.lookup_tools import extractor, dense_lookup
import pickle


# Select the MC process # 
MCbkg='WlvJets'
#MCbkg='TT1l1v'

# Data
outputData_18 = util.load("monoHbb/efficiencies/TriggerEfficiency_Data_18_Path120.coffea")
outputData_17 = util.load("monoHbb/efficiencies/TriggerEfficiency_Data_17_Path120.coffea")

# MC 
if(MCbkg == 'WlvJets'):
    outputMC_18 = util.load("monoHbb/efficiencies/TriggerEfficiency_MCWlvJets_18_Path120_1Mevts.coffea")
    outputMC_17 = util.load("monoHbb/efficiencies/TriggerEfficiency_MCWlvJets_17_Path120_1Mevts.coffea")    
    groupingMC_18 = {
        "W(lv)+Jets": [
            "WJets_LNu_WPt_100To250_18",
            "WJets_LNu_WPt_250To400_18",
            "WJets_LNu_WPt_400To600_18",
            "WJets_LNu_WPt_600Toinf_18",
        ],
    }
    groupingMC_17 = {
        "W(lv)+Jets": [
            "WJets_LNu_WPt_100To250_17",
            "WJets_LNu_WPt_250To400_17",
            "WJets_LNu_WPt_400To600_17",
            "WJets_LNu_WPt_600Toinf_17",
        ],
    }
elif(MCbkg == 'TT1l1v'):
    outputMC_18 = util.load("monoHbb/efficiencies/TriggerEfficiency_MCTTbar1l1v_18_Path120.coffea")
    outputMC_17 = util.load("monoHbb/efficiencies/TriggerEfficiency_MCTTbar1l1v_17_Path120.coffea")
    groupingMC_18 = {"TTbar": ["TTToSemiLeptonic_18",],}
    groupingMC_17 = {"TTbar": ["TTToSemiLeptonic_17",],}

#--------------------------------------------------------------------------------
#------2018-------
#MC
histList = []
for samp, sampList in groupingMC_18.items():
    histList += [outputMC_18[s] for s in sampList]    
outputHistMC_18 = accumulate(histList)
for key, histo in outputHistMC_18.items():
    if isinstance(histo, hist.Hist):
        outputHistMC_18[key] = GroupBy(histo, 'dataset', 'dataset', groupingMC_18)
#Data
outputHistData_18 = accumulate([histo for key, histo in outputData_18.items()])

#------2017-------
#MC
histList = []
for samp, sampList in groupingMC_17.items():
    histList += [outputMC_17[s] for s in sampList]    
outputHistMC_17 = accumulate(histList)
for key, histo in outputHistMC_17.items():
    if isinstance(histo, hist.Hist):
        outputHistMC_17[key] = GroupBy(histo, 'dataset', 'dataset', groupingMC_17)
#Data
outputHistData_17 = accumulate([histo for key, histo in outputData_17.items()])

#--------------------------------------------------------------------------------
# Basic Kinematic Plots

def plot_features(var, isMC, era):

    if (era==2018):
        outputHistMC = outputHistMC_18
        outputHistData = outputHistData_18
    elif (era==2017):
        outputHistMC = outputHistMC_17
        outputHistData = outputHistData_17

    if(isMC==False):
        hD = outputHistData[var][{'dataset': sum}]
        figr, (ax) = plt.subplots(1)
        mplhep.histplot(hD)
        plt.savefig(f'Plots_TriggerEff/{var}_TrigSFtest_Data{era}.png')

    elif(isMC==True):
        hMC = outputHistMC[var][{'dataset': sum}]
        figr, (ax) = plt.subplots(1)
        mplhep.histplot(hMC)
        plt.savefig(f'Plots_TriggerEff/{var}_TrigSFtest_MC{era}_{MCbkg}.png')
#'''
plot_features(var='hMET_pT', isMC=True, era=2018)
plot_features(var='hRecoil', isMC=True, era=2018)
plot_features(var='hHT', isMC=True, era=2018)

plot_features(var='hMET_pT', isMC=False, era=2018)
plot_features(var='hRecoil', isMC=False, era=2018)
plot_features(var='hHT', isMC=False, era=2018)

plot_features(var='hMET_pT', isMC=True, era=2017)
plot_features(var='hRecoil', isMC=True, era=2017)
plot_features(var='hHT', isMC=True, era=2017)

plot_features(var='hMET_pT', isMC=False, era=2017)
plot_features(var='hRecoil', isMC=False, era=2017)
plot_features(var='hHT', isMC=False, era=2017)
#'''
#---------------------------------------------------------------------------------
# Efficiency Plots (overlay MC and Data)

def overlay_Efficiency(era, rebinFactor):

    if (era==2018):
        outputHistMC = outputHistMC_18
        outputHistData = outputHistData_18
    elif (era==2017):
        outputHistMC = outputHistMC_17
        outputHistData = outputHistData_17

    figr, (ax) = plt.subplots(1)
    key='hBool_EvtSelnTrigger'

    for isMC in [True, False]:

        if isMC==True:
            hist1_num = outputHistMC[key][{"dataset": sum, "recoil": slice(0j, 1000j, hist.rebin(rebinFactor)), "bool_evtsel": slice(1j,2j,sum), "bool_evtseltrig": slice(1j,2j,sum)}]
            hist1_deno = outputHistMC[key][{"dataset": sum, "recoil": slice(0j, 1000j, hist.rebin(rebinFactor)), "bool_evtsel": slice(1j,2j,sum), "bool_evtseltrig": slice(0j,2j,sum)}]
        elif isMC==False:
            hist1_num = outputHistData[key][{"dataset": sum, "recoil": slice(0j, 1000j, hist.rebin(rebinFactor)), "bool_evtsel": slice(1j,2j,sum), "bool_evtseltrig": slice(1j,2j,sum)}]
            hist1_deno = outputHistData[key][{"dataset": sum, "recoil": slice(0j, 1000j, hist.rebin(rebinFactor)), "bool_evtsel": slice(1j,2j,sum), "bool_evtseltrig": slice(0j,2j,sum)}]

        numArr = ak.Array(hist1_num.values())
        denoArr = ak.Array(hist1_deno.values())
        Eff_bin = ak.where(denoArr>0, numArr/denoArr, 0)

        Num = 100/rebinFactor
        recoilbins = np.linspace(0.0, 1000.0, num=int(Num), endpoint=False)
        ax.set_xlim(0,1050)
        ax.set_ylim(-0.01,1.1)
        ax.set_xlabel('Recoil [GeV]', fontsize=15)
        ax.set_ylabel('Trigger Efficiency', fontsize=15)
    
        if isMC==True:
            if(MCbkg=='WlvJets'):
                plt.plot(recoilbins, Eff_bin, 'o', color='blue', markersize=5, label=r"MC: W($\mu \nu$) + Jets")            
            elif(MCbkg=='TT1l1v'):
                plt.plot(recoilbins, Eff_bin, 'o', color='blue', markersize=5, label=r"MC: tt(semilepton)")            
        elif isMC==False:
            plt.plot(recoilbins, Eff_bin, 'x', color='red', markersize=5, label=r"Data: MET dataset")            
            plt.xticks(np.arange(0,1100,step=100))
            plt.yticks(np.arange(0,1.2,step=0.1))
            plt.grid()
        plt.vlines(x=250, ymin=0.0, ymax=1.1, linestyles='dashed', colors='black')
        ax.legend(loc='lower right', fontsize=16)
        CMS = plt.text(0.0, 1.0, r"$\bf{CMS}$ Preliminary", fontsize=12, horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)
        if(era==2018):
            lumi = plt.text(1., 1., r"59.83 fb$^{-1}$ (2018)", fontsize=12, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)                
        elif(era==2017):
            lumi = plt.text(1., 1., r"41.48 fb$^{-1}$ (2017)", fontsize=12, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)                
        
    plt.savefig(f'Plots_TriggerEff/TriggEfficiency_test_DataMC{era}_{MCbkg}_rebin{rebinFactor}.png')

overlay_Efficiency(era=2018, rebinFactor=4)
overlay_Efficiency(era=2017, rebinFactor=4)
overlay_Efficiency(era=2018, rebinFactor=5)
overlay_Efficiency(era=2017, rebinFactor=5)
overlay_Efficiency(era=2018, rebinFactor=10)
overlay_Efficiency(era=2017, rebinFactor=10)


def plot_trigSF(era, rebinFactor):

    if (era==2018):
        outputHistMC = outputHistMC_18
        outputHistData = outputHistData_18
    elif (era==2017):
        outputHistMC = outputHistMC_17
        outputHistData = outputHistData_17

    triggerEffMC_num = outputHistMC[key][{"dataset": sum, "recoil": slice(0j, 1000j, hist.rebin(rebinFactor)), "bool_evtsel": slice(1j,2j,sum), "bool_evtseltrig": slice(1j,2j,sum)}]
    triggerEffMC_deno = outputHistMC[key][{"dataset": sum, "recoil": slice(0j, 1000j, hist.rebin(rebinFactor)), "bool_evtsel": slice(1j,2j,sum), "bool_evtseltrig": slice(0j,2j,sum)}]
    triggerEffMC_numArr = ak.Array(triggerEffMC_num.values())
    triggerEffMC_denoArr = ak.Array(triggerEffMC_deno.values())
    triggerEff_MC = ak.where(triggerEffMC_denoArr>0, triggerEffMC_numArr/triggerEffMC_denoArr, 0)

    triggerEffData_num = outputHistData[key][{"dataset": sum, "recoil": slice(0j, 1000j, hist.rebin(rebinFactor)), "bool_evtsel": slice(1j,2j,sum), "bool_evtseltrig": slice(1j,2j,sum)}]
    triggerEffData_deno = outputHistData[key][{"dataset": sum, "recoil": slice(0j, 1000j, hist.rebin(rebinFactor)), "bool_evtsel": slice(1j,2j,sum), "bool_evtseltrig": slice(0j,2j,sum)}]
    triggerEffData_numArr = ak.Array(triggerEffData_num.values())
    triggerEffData_denoArr = ak.Array(triggerEffData_deno.values())
    triggerEff_Data = ak.where(triggerEffData_denoArr>0, triggerEffData_numArr/triggerEffData_denoArr, 0)

    #compute the SF (Data/MC efficiency ratio)
    NBins = int(100/rebinFactor)
    triggerSF = triggerEff_Data/triggerEff_MC
    recoilbins = np.linspace(5.0, 1005.0, num=NBins, endpoint=False)
    recoilbin_edges = np.linspace(0.0, 1000.0, num=NBins, endpoint=False)

    #Store the SFs in a lookup table
    triggerEffLookup = dense_lookup.dense_lookup( np.array(triggerSF), np.array(recoilbin_edges) )
    with open(f"monoHbb/scalefactors/triggerSFsDenseLookup_{MCbkg}DataUL{era}_TriggerPath120_NBins{NBins}.pkl", "wb") as _file:
        pickle.dump(triggerEffLookup, _file)


    #make SF plot
    figr, (ax) = plt.subplots(1)
    plt.plot(recoilbins, triggerSF, 'o', color='orange', markersize='4', label='Trigger Scale Factor')
    ax.set_xlabel('Recoil [GeV]', fontsize=15)
    ax.set_ylabel('Data/MC SF', fontsize=15)
    plt.xticks(np.arange(0, 1100, step=100))
    plt.yticks(np.arange(0.95, 1.03, step=0.01))
    plt.grid()
    ax.set_xlim(200, 1050)
    ax.set_ylim(0.95, 1.03)
    ax.legend(loc='upper right', fontsize=16)
    plt.vlines(x=250, ymin=0.0, ymax=1.1, linestyles='dashed', colors='black')
    plt.axhline(y=1.01, linestyle='-', color='purple')
    plt.axhline(y=0.99, linestyle='-', color='purple')
    CMS = plt.text(0.0, 1.0, r"$\bf{CMS}$ Preliminary", fontsize=12, horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)
    if(era==2018):
        lumi = plt.text(1., 1., r"59.83 fb$^{-1}$ (2018)", fontsize=12, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)                
    elif(era==2017):
        lumi = plt.text(1., 1., r"41.48 fb$^{-1}$ (2017)", fontsize=12, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)                
    plt.savefig(f'Plots_TriggerEff/trigSF_test_DataMC{era}_{MCbkg}_rebin{rebinFactor}.png')

plot_trigSF(era=2018, rebinFactor=4)
plot_trigSF(era=2017, rebinFactor=4)
plot_trigSF(era=2018, rebinFactor=5)
plot_trigSF(era=2017, rebinFactor=5)
plot_trigSF(era=2018, rebinFactor=10)
plot_trigSF(era=2017, rebinFactor=10)
