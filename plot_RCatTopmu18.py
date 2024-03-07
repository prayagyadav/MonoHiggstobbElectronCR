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

outputData = accumulate([
    util.load("Outputs_Test5d_RCatTopmu/output_DataA_Test5dRCatTopmuCorrFull_run20231012_085337.coffea"),
    util.load("Outputs_Test5d_RCatTopmu/output_DataA_Test5dRCatTopmuCorrFull_run20231012_121307.coffea"),
    util.load("Outputs_Test5d_RCatTopmu/output_DataB_Test5dRCatTopmuCorrFull_run20231012_081159.coffea"),
    util.load("Outputs_Test5d_RCatTopmu/output_DataC_Test5dRCatTopmuCorrFull_run20231012_081136.coffea"),
    util.load("Outputs_Test5d_RCatTopmu/output_DataD1_Test5dRCatTopmuCorrFull_run20231012_084543.coffea"),
    util.load("Outputs_Test5d_RCatTopmu/output_DataD11_Test5dRCatTopmuCorrFull_run20231012_091444.coffea"),
    util.load("Outputs_Test5d_RCatTopmu/output_DataD2_Test5dRCatTopmuCorrFull_run20231012_110831.coffea"),
])

outputMC = accumulate([
    util.load("Outputs_Test5e_RCatTopmu/output_MCTTbar1l1v_Test5eRCatTopmuCorrFull_run20231025_123824_metphiMC.coffea"),
    util.load("Outputs_Test5e_RCatTopmu/output_MCTTbar2l2v_Test5eRCatTopmuCorrFull_run20231025_135454_metphiMC.coffea"),
    util.load("Outputs_Test5e_RCatTopmu/output_MCWlvJets_Test5eRCatTopmuCorrFull_run20231025_155046_metphiMC.coffea"),
    util.load("Outputs_Test5e_RCatTopmu/output_MCSingleTop2_Test5eRCatTopmuCorrFull_run20231025_162054_metphiMC.coffea"),
])


# specify the MC grouping
groupingMC = {
    "W(lv)+Jets": [
        "WJets_LNu_WPt_100To250_18",
        "WJets_LNu_WPt_250To400_18",
        "WJets_LNu_WPt_400To600_18",
        "WJets_LNu_WPt_600Toinf_18",
    ],
    "SingleTop": [
        "ST_tW_top_18",  
        "ST_tW_antitop_18",  
        "ST_tchannel_top_18",  
        "ST_tchannel_antitop_18",  
    ],
    "tt": [
        "TTToSemiLeptonic_18",
        "TTTo2L2Nu_18",
        "TTToHadronic_18",
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
        sels_R_1muCR = ["NoCut", "MET-Trigger", "MET-Filter", r"$N_{tau}=0$", r"$N_{\gamma}=0$", "HEM-veto", r"$N_{\mu}=1$", r"$p_{T}^{miss}>50$GeV", r"Recoil$>200$GeV", r"$p_{T}(b_{1})>$50GeV", r"$p_{T}(b_{2})>$30GeV", r"$p_{T}(b_{1}b_{2})>$100GeV", r"$100<M(b_{1}b_{2})<150$GeV", r"$N_{ajet}\geq1$"]
        import pandas as pd 
        dict = {'Bin': bins_R_1muCR, 'Selection': sels_R_1muCR, 'NEvts_data': Nevents_Data[:14], 'NEvts_bkg': Nevents_MC[:14], 'Ratio': Ratio_DataMC}

        df = pd.DataFrame(dict)
        np.savetxt('Plots_Test5eTopmu/'+str(key)+'_table_2018.txt', df.values, delimiter="\t", fmt='%d\t%s\t%.2e\t%.2e\t%.3f') 

        # make cutflow plot
        hMC = outputHistMC[key]
        hData = outputHistData[key][{'dataset':sum}]
        plotWithRatio(h=hMC, hData=hData, overlay='dataset', logY=True, xLabel='Selection Bin', xRange=None, colors_cat='Topmu', extraText=None, lumi=59.83, year=2018)
        plt.savefig('Plots_Test5eTopmu/'+str(key)+'_plot_2018.png')


#----------------------------------------
## Kinematic Plots ##
#----------------------------------------
def make_kinematicplot(var, Xlabel, rebin_factor, logY, xRange=None):
    h1 = outputHistMC[var][{"systematic": 'nominal'}][...,::hist.rebin(rebin_factor)]
    hData = outputHistData[var][{'dataset':sum}][{"systematic": 'noweight'}][...,::hist.rebin(rebin_factor)]
    
    figr, (ax) = plt.subplots(1)
    plotWithRatio(h=h1, hData=hData, overlay='dataset', logY=logY, xLabel=Xlabel, xRange=xRange, colors_cat='Topmu', extraText=extraText, lumi=59.83, year=2018)
    plt.savefig('Plots_Test5eTopmu/'+var+'_RCatTopmu_2018.png')


make_kinematicplot(var="MET_pT", Xlabel="MET [GeV]", rebin_factor=4, logY=True, xRange=[0.,800.])
make_kinematicplot(var="MET_Phi", Xlabel=r"MET $\phi$ [GeV]", rebin_factor=5, logY=False)
make_kinematicplot(var="Recoil", Xlabel="Recoil [GeV]", rebin_factor=2, logY=True, xRange=[0.,1000.])
make_kinematicplot(var="Recoil_Phi", Xlabel=r"Recoil $\phi$ [GeV]", rebin_factor=5, logY=False)
make_kinematicplot(var="HT", Xlabel="HT [GeV]", rebin_factor=4, logY=True,) #xRange=[0.,1000.])

make_kinematicplot(var="Muon_pT", Xlabel=r"Muon $p_{T}$ [GeV]", rebin_factor=5, logY=True, xRange=[0.,800.])
make_kinematicplot(var="Muon_Eta", Xlabel=r"Muon $\eta$", rebin_factor=6, logY=False)
make_kinematicplot(var="Muon_Phi", Xlabel=r"Muon $\phi$", rebin_factor=6, logY=False)
make_kinematicplot(var="dPhi_met_Muon", Xlabel=r"$\Delta \phi$ (Muon, MET) ", rebin_factor=2, logY=False, xRange=[0., 3.2])

make_kinematicplot(var="Dijet_pT", Xlabel=r"Dijet $p_{T}$ [GeV]", rebin_factor=5, logY=True, xRange=[0.,1000.])
make_kinematicplot(var="Dijet_Eta", Xlabel=r"Dijet $\eta$", rebin_factor=6, logY=False)
make_kinematicplot(var="Dijet_Phi", Xlabel=r"Dijet $\phi$", rebin_factor=6, logY=False)
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
    plotWithRatio(h=hMC_j1, hData=hData_j1, overlay='dataset', logY=logY, xLabel="Jet1 "+Xlabel, xRange=xRange, colors_cat='Topmu', extraText=extraText, lumi=59.83, year=2018)
    plt.savefig('Plots_Test5eTopmu/'+var+'Jet1_2018.png')

    # Jet-2
    hMC_j2 = outputHistMC[var][{"jetInd": 'Jet2', "systematic": 'nominal'}][...,::hist.rebin(rebin_factor)]
    hData_j2 = outputHistData[var][{'dataset':sum}][{"jetInd": 'Jet2', "systematic": 'noweight'}][...,::hist.rebin(rebin_factor)]
    figr, (ax) = plt.subplots(1)
    plotWithRatio(h=hMC_j2, hData=hData_j2, overlay='dataset', logY=logY, xLabel="Jet2 "+Xlabel, xRange=xRange, colors_cat='Topmu', extraText=extraText, lumi=59.83, year=2018)
    plt.savefig('Plots_Test5eTopmu/'+var+'Jet2_2018.png')


make_kinematicplot_jet(var="bJet_pT", Xlabel=r"$p_{T}$ [GeV]", rebin_factor=5, logY=True, xRange=[0.,1000.])
make_kinematicplot_jet(var="bJet_Eta", Xlabel=r"$\eta$", rebin_factor=6, logY=False)
make_kinematicplot_jet(var="bJet_Phi", Xlabel=r"$\phi$", rebin_factor=6, logY=False)
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
    plt.savefig('Plots_Test5eTopmu/'+str(var)+'_'+str(syste)+'_updown_2018.png') 

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
