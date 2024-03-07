import hist
import numpy as np
import awkward as ak
from coffea import util
from coffea.processor import accumulate
from coffea.lookup_tools import extractor, dense_lookup
import mplhep as hep
import matplotlib.pyplot as plt
from plotting import GroupBy
from plottingTool import plotWithRatio

extraText = r"$t \bar{t}$($\mu \nu$) + 2b CR"+" \n"+" Resolved"
LumiVal = 41.48
Year = 2017


outputData = accumulate([
    util.load("2017/Outputs_Test5e_RCatTopmu/output_DataB_Test5eRCatTopmu2017_run20231031_095744.coffea"),
    util.load("2017/Outputs_Test5e_RCatTopmu/output_DataC_Test5eRCatTopmu2017_run20231031_085412.coffea"),
    util.load("2017/Outputs_Test5e_RCatTopmu/output_DataD_Test5eRCatTopmu2017_run20231031_090024.coffea"),
    util.load("2017/Outputs_Test5e_RCatTopmu/output_DataE_Test5eRCatTopmu2017_run20231031_091244.coffea"),
    util.load("2017/Outputs_Test5e_RCatTopmu/output_DataF_Test5eRCatTopmu2017_run20231031_094716.coffea"),
])
outputMC = accumulate([
    util.load("2017/Outputs_Test6c_RCatTopmu/output_MCTTbar1l1v_Test6cRCatTopmu2017_run20240202_033941.coffea"),
    util.load("2017/Outputs_Test6c_RCatTopmu/output_MCTTbar2l2v_Test6cRCatTopmu2017_run20240202_041215.coffea"),
    util.load("2017/Outputs_Test6c_RCatTopmu/output_MCTTbar0l0v_Test6cRCatTopmu2017_run20240202_054641.coffea"),
    util.load("2017/Outputs_Test6c_RCatTopmu/output_MCSingleTop1_Test6cRCatTopmu2017_run20240202_070140.coffea"),
    util.load("2017/Outputs_Test6c_RCatTopmu/output_MCSingleTop2_Test6cRCatTopmu2017_run20240202_072910.coffea"),
    util.load("2017/Outputs_Test6c_RCatTopmu/output_MCWlvJets1_Test6cRCatTopmu2017_run20240202_085216.coffea"),
    util.load("2017/Outputs_Test6c_RCatTopmu/output_MCWlvJets2_Test6cRCatTopmu2017_run20240202_082158.coffea"),
    util.load("2017/Outputs_Test6c_RCatTopmu/output_MCWlvJets3_Test6cRCatTopmu2017_run20240202_093754.coffea"),
    util.load("2017/Outputs_Test6c_RCatTopmu/output_MCWlvJets4_Test6cRCatTopmu2017_run20240202_085622.coffea"),
])

groupingMC = {
    "WJets_WpT": [
        "WJets_LNu_WPt_100To250_17",
        "WJets_LNu_WPt_250To400_17",
        "WJets_LNu_WPt_400To600_17",
        "WJets_LNu_WPt_600Toinf_17",
    ],
    "SingleTop": [
        "ST_tW_top_17",  
        "ST_tW_antitop_17",
        "ST_tchannel_top_17",  
        "ST_tchannel_antitop_17",  
    ],
    "tt": [
        "TTToSemiLeptonic_17",
        "TTTo2L2Nu_17",
        "TTToHadronic_17",
    ],
}

#----------------------------------------
## Group MC samples and Data eras ##
#----------------------------------------
histList = []
for samp, sampList in groupingMC.items():
    histList += [outputMC[s] for s in sampList]
outputHistMC = accumulate(histList)
for key, histo in outputHistMC.items():
    if isinstance(histo, hist.Hist):
        outputHistMC[key] = GroupBy(histo, 'dataset', 'dataset', groupingMC)

outputHistData = accumulate([histo for key, histo in outputData.items()])

#----------------------------------------
## Make Cutflow Table ##
#----------------------------------------
for key in outputHistMC.keys():
    if key.startswith('Cutflow_RCat_CRTopmu'): 
        
        #make cutflow table
        Nevents_MC = outputHistMC[key][{"dataset": sum}].values()
        Nevents_Data = outputHistData[key][{"dataset": sum}].values()
        Ratio_DataMC = Nevents_Data[:14]/Nevents_MC[:14]

        bins_R_1muCR = np.linspace(0,13,14)
        sels_R_1muCR = ["NoCut", "MET-Trigger", "MET-Filter", r"$N_{tau}=0$", r"$N_{\gamma}=0$", "HEM-veto", r"$N_{\mu}=1$", r"$p_{T}^{miss}>50$GeV", r"Recoil$>200$GeV", r"$p_{T}(b_{1})>$50GeV", r"$p_{T}(b_{2})>$30GeV", r"$p_{T}(b_{1}b_{2})>$100GeV", r"$70<M(b_{1}b_{2})<150$GeV", r"$N_{ajet}\geq1$"]
        import pandas as pd 
        dict = {'Bin': bins_R_1muCR, 'Selection': sels_R_1muCR, 'NEvts_data': Nevents_Data[:14], 'NEvts_bkg': Nevents_MC[:14], 'Ratio': Ratio_DataMC}

        df = pd.DataFrame(dict)
        np.savetxt('2017/Plots_Test6c_RCatTopmu/'+str(key)+'_table_2017.txt', df.values, delimiter="\t", fmt='%d\t%s\t%.2e\t%.2e\t%.3f') 

        # make cutflow plot
        hMC = outputHistMC[key]
        hData = outputHistData[key][{'dataset':sum}]
        plotWithRatio(h=hMC, hData=hData, overlay='dataset', logY=True, xLabel='Selection Bin', xRange=None, extraText=None, lumi=LumiVal, year=Year, colors_cat='Topmu')
        plt.savefig('2017/Plots_Test6c_RCatTopmu/'+str(key)+'_plot_2017.png')

#----------------------------------------
## Kinematic Plots ##
#----------------------------------------
def make_kinematicplot(var, Xlabel, rebin_factor, logY, xRange=None):
    h1 = outputHistMC[var][{"systematic": 'nominal'}][...,::hist.rebin(rebin_factor)]
    hData = outputHistData[var][{'dataset':sum}][{"systematic": 'noweight'}][...,::hist.rebin(rebin_factor)]
    
    figr, (ax) = plt.subplots(1)
    plotWithRatio(h=h1, hData=hData, overlay='dataset', logY=logY, xLabel=Xlabel, xRange=xRange, colors_cat='Topmu', extraText=extraText, lumi=LumiVal, year=Year)
    plt.savefig('2017/Plots_Test6c_RCatTopmu/'+var+'_RCatTopmu_2017.png')


make_kinematicplot(var="MET_pT", Xlabel="MET [GeV]", rebin_factor=4, logY=True, xRange=[0.,800.])
make_kinematicplot(var="MET_Phi", Xlabel=r"MET $\phi$ [GeV]", rebin_factor=8, logY=False)
make_kinematicplot(var="Recoil", Xlabel="Recoil [GeV]", rebin_factor=2, logY=True, xRange=[0.,1000.])
make_kinematicplot(var="Recoil_Phi", Xlabel=r"Recoil $\phi$ [GeV]", rebin_factor=8, logY=False)
make_kinematicplot(var="HT", Xlabel="HT [GeV]", rebin_factor=4, logY=True,) #xRange=[0.,1000.])

make_kinematicplot(var="Muon_pT", Xlabel=r"Muon $p_{T}$ [GeV]", rebin_factor=5, logY=True, xRange=[0.,800.])
make_kinematicplot(var="Muon_Eta", Xlabel=r"Muon $\eta$", rebin_factor=8, logY=False)
make_kinematicplot(var="Muon_Phi", Xlabel=r"Muon $\phi$", rebin_factor=8, logY=False)
make_kinematicplot(var="dPhi_met_Muon", Xlabel=r"$\Delta \phi$ (Muon, MET) ", rebin_factor=2, logY=False, xRange=[0., 3.2])

make_kinematicplot(var="Dijet_pT", Xlabel=r"Dijet $p_{T}$ [GeV]", rebin_factor=5, logY=True, xRange=[0.,1000.])
make_kinematicplot(var="Dijet_Eta", Xlabel=r"Dijet $\eta$", rebin_factor=8, logY=False)
make_kinematicplot(var="Dijet_Phi", Xlabel=r"Dijet $\phi$", rebin_factor=8, logY=False)
make_kinematicplot(var="Dijet_M", Xlabel=r"Dijet Mass [GeV]", rebin_factor=4, logY=False, xRange=[0.,200.])
make_kinematicplot(var="dPhi_met_Dijet", Xlabel=r"$\Delta \phi$ (Dijet, MET) ", rebin_factor=2, logY=False, xRange=[0., 3.2])

make_kinematicplot(var="dR_bbJets", Xlabel=r"$\Delta R$ (bb)", rebin_factor=2, logY=False, xRange=[0., 3.5])
make_kinematicplot(var="dPhi_bbJets", Xlabel=r"$\Delta \phi$ (bb)", rebin_factor=2, logY=False, xRange=[0., 3.2])
make_kinematicplot(var="dEta_bbJets", Xlabel=r"$\Delta \eta$ (bb)", rebin_factor=2, logY=False, xRange=[0., 2.55])


#----------------------------------------
def make_kinematicplot_jet(var, Xlabel, rebin_factor, logY, xRange=None):
    # Jet-1
    hMC_j1 = outputHistMC[var][{"jetInd": 'Jet1', "systematic": 'nominal'}][...,::hist.rebin(rebin_factor)]
    hData_j1 = outputHistData[var][{'dataset':sum}][{"jetInd": 'Jet1', "systematic": 'noweight'}][...,::hist.rebin(rebin_factor)]
    figr, (ax) = plt.subplots(1)
    plotWithRatio(h=hMC_j1, hData=hData_j1, overlay='dataset', logY=logY, xLabel="Jet1 "+Xlabel, xRange=xRange, colors_cat='Topmu', extraText=extraText, lumi=LumiVal, year=Year)
    plt.savefig('2017/Plots_Test6c_RCatTopmu/'+var+'Jet1_2017.png')

    # Jet-2
    hMC_j2 = outputHistMC[var][{"jetInd": 'Jet2', "systematic": 'nominal'}][...,::hist.rebin(rebin_factor)]
    hData_j2 = outputHistData[var][{'dataset':sum}][{"jetInd": 'Jet2', "systematic": 'noweight'}][...,::hist.rebin(rebin_factor)]
    figr, (ax) = plt.subplots(1)
    plotWithRatio(h=hMC_j2, hData=hData_j2, overlay='dataset', logY=logY, xLabel="Jet2 "+Xlabel, xRange=xRange, colors_cat='Topmu', extraText=extraText, lumi=LumiVal, year=Year)
    plt.savefig('2017/Plots_Test6c_RCatTopmu/'+var+'Jet2_2017.png')


make_kinematicplot_jet(var="bJet_pT", Xlabel=r"$p_{T}$ [GeV]", rebin_factor=4, logY=True, xRange=[0.,1000.])
make_kinematicplot_jet(var="bJet_Eta", Xlabel=r"$\eta$", rebin_factor=8, logY=False)
make_kinematicplot_jet(var="bJet_Phi", Xlabel=r"$\phi$", rebin_factor=8, logY=False)
make_kinematicplot_jet(var="bJet_M", Xlabel=r" Mass", rebin_factor=4, logY=False, xRange=[0.,200.])
make_kinematicplot_jet(var="bJet_btagDeepB", Xlabel=r" btagDeepB", rebin_factor=4, logY=False)
make_kinematicplot_jet(var="dPhi_met_bJet", Xlabel=r" $\Delta \phi$(MET, jet)", rebin_factor=4, logY=False, xRange=[0., 3.2])


#----------------------------------------
## Systematics variation ##
#----------------------------------------
def checkSyst_nomUpDown(var, var_axis, syste, Title):

    histo = outputHistMC[var]

    h0 = histo[{"systematic": 'nominal'}].project(var_axis)[...,::hist.rebin(4)]
    h1 = histo[{"systematic": str(syste)+'Up'}].project(var_axis)[...,::hist.rebin(4)]
    h2 = histo[{"systematic": str(syste)+'Down'}].project(var_axis)[...,::hist.rebin(4)]

    figr, (ax) = plt.subplots(1)
    hep.histplot([h0,h1,h2], ax=ax, stack=False, label=['nominal', 'Up', 'Down'], histtype='step', yerr=False)
    ax.set_xlim(0,500)
    ax.set_ylabel("Events")
    ax.set_title(Title, fontsize=16)
    ax.legend(fontsize=15)
    plt.savefig('2017/Plots_Test6c_RCatTopmu/'+str(var)+'_'+str(syste)+'_updown_2017.png') 

checkSyst_nomUpDown(var="MET_pT", var_axis="met", syste='JES', Title='JES')
checkSyst_nomUpDown(var="MET_pT", var_axis="met", syste='JER', Title='JER')
checkSyst_nomUpDown(var="MET_pT", var_axis="met", syste='muEffWeight', Title='Muon (Iso+ID) SF')
checkSyst_nomUpDown(var="MET_pT", var_axis="met", syste='btagWeight', Title='btag SF')
checkSyst_nomUpDown(var="MET_pT", var_axis="met", syste='puWeight', Title='Pileup weight')
checkSyst_nomUpDown(var="MET_pT", var_axis="met", syste='TriggerSFWeight', Title='Trigger SF')

checkSyst_nomUpDown(var="Recoil", var_axis="recoil", syste='JES', Title='JES')
checkSyst_nomUpDown(var="Recoil", var_axis="recoil", syste='JER', Title='JER')
checkSyst_nomUpDown(var="Recoil", var_axis="recoil", syste='muEffWeight', Title='Muon (Iso+ID) SF')
checkSyst_nomUpDown(var="Recoil", var_axis="recoil", syste='btagWeight', Title='btag SF')
checkSyst_nomUpDown(var="Recoil", var_axis="recoil", syste='puWeight', Title='Pileup weight')
checkSyst_nomUpDown(var="Recoil", var_axis="recoil", syste='TriggerSFWeight', Title='Trigger SF')
