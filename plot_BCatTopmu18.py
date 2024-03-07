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

extraText = r"$t \bar{t}$($\mu \nu$) + FatJet CR"+" \n"+" Boosted"

outputData = accumulate([
    util.load("Outputs_Test6b_BCatTopmu/output_DataA_Test6bBCatTopmu2018_run20231201_080119.coffea"),
    util.load("Outputs_Test6b_BCatTopmu/output_DataB_Test6bBCatTopmu2018_run20231201_081241.coffea"),
    util.load("Outputs_Test6b_BCatTopmu/output_DataC_Test6bBCatTopmu2018_run20231201_090050.coffea"),
    util.load("Outputs_Test6b_BCatTopmu/output_DataD_Test6bBCatTopmu2018_run20231201_084221.coffea"),
])
outputMC = accumulate([
    util.load("Outputs_Test6b_BCatTopmu/output_MCTTbar1l1v_Test6bBCatTopmu2018_run20231201_083932.coffea"),
    util.load("Outputs_Test6b_BCatTopmu/output_MCTTbar2l2v_Test6bBCatTopmu2018_run20231201_091453.coffea"),
    util.load("Outputs_Test6b_BCatTopmu/output_MCTTbar0l0v_Test6bBCatTopmu2018_run20231206_082136.coffea"),
    util.load("Outputs_Test6b_BCatTopmu/output_MCSingleTop1_Test6bBCatTopmu2018_run20231202_074543.coffea"),
    util.load("Outputs_Test6b_BCatTopmu/output_MCSingleTop2_Test6bBCatTopmu2018_run20231201_143200.coffea"),
    util.load("Outputs_Test6b_BCatTopmu/output_MCWlvJets_Test6bBCatTopmu2018_run20231203_104140.coffea"),
])


# specify the MC grouping
groupingMC = {
    "WJets_WpT": [
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
    if key.startswith('Cutflow_BCat_CRTopmu'): 
        
        #make cutflow table
        Nevents_MC = outputHistMC[key][{"dataset": sum}].values()
        Nevents_Data = outputHistData[key][{"dataset": sum}].values()
        Ratio_DataMC = Nevents_Data[:12]/Nevents_MC[:12]

        bins_B_1muCR = np.linspace(0,11,12)
        sels_B_1muCR = ["NoCut", "MET-Trigger", "MET-Filter", r"$N_{tau}=0$", r"$N_{\gamma}=0$", "HEM-veto", r"$N_{\mu}=1$", r"$p_{T}^{miss}>50$GeV", r"Recoil$>250$GeV", r"N(FatJet)=1", r"$N_{IsoAddjet}\leq2$", r"$N_{IsoLooseBtagjet}=1$"]
        import pandas as pd 
        dict = {'Bin': bins_B_1muCR, 'Selection': sels_B_1muCR, 'NEvts_data': Nevents_Data[:12], 'NEvts_bkg': Nevents_MC[:12], 'Ratio': Ratio_DataMC}

        df = pd.DataFrame(dict)
        np.savetxt('Plots_Test6b_BCatTopmu/'+str(key)+'_table_2018.txt', df.values, delimiter="\t", fmt='%d\t%s\t%.2e\t%.2e\t%.3f') 

        # make cutflow plot
        hMC = outputHistMC[key]
        hData = outputHistData[key][{'dataset':sum}]
        plotWithRatio(h=hMC, hData=hData, overlay='dataset', logY=True, xLabel='Selection Bin', xRange=None, colors_cat='Topmu', extraText=None, lumi=59.83, year=2018)
        plt.savefig('Plots_Test6b_BCatTopmu/'+str(key)+'_plot_2018.png')


#----------------------------------------
## Kinematic Plots ##
#----------------------------------------
def make_kinematicplot(var, Xlabel, rebin_factor, logY, xRange=None):
    h1 = outputHistMC[var][{"systematic": 'nominal'}][...,::hist.rebin(rebin_factor)]
    hData = outputHistData[var][{'dataset':sum}][{"systematic": 'noweight'}][...,::hist.rebin(rebin_factor)]
    
    figr, (ax) = plt.subplots(1)
    plotWithRatio(h=h1, hData=hData, overlay='dataset', logY=logY, xLabel=Xlabel, xRange=xRange, colors_cat='Topmu', extraText=extraText, lumi=59.83, year=2018)
    plt.savefig('Plots_Test6b_BCatTopmu/'+var+'_BCatTopmu_2018.png')

make_kinematicplot(var="MET_pT", Xlabel="MET [GeV]", rebin_factor=4, logY=True, xRange=[0.,800.])
make_kinematicplot(var="MET_Phi", Xlabel=r"MET $\phi$ [GeV]", rebin_factor=6, logY=False)
make_kinematicplot(var="Recoil", Xlabel="Recoil [GeV]", rebin_factor=4, logY=True, xRange=[0.,1000.])
make_kinematicplot(var="Recoil_Phi", Xlabel=r"Recoil $\phi$ [GeV]", rebin_factor=5, logY=False)
make_kinematicplot(var="HT", Xlabel="HT [GeV]", rebin_factor=4, logY=True,) #xRange=[0.,1000.])

make_kinematicplot(var="Muon_pT", Xlabel=r"Muon $p_{T}$ [GeV]", rebin_factor=5, logY=True, xRange=[0.,800.])
make_kinematicplot(var="Muon_Eta", Xlabel=r"Muon $\eta$", rebin_factor=6, logY=False)
make_kinematicplot(var="Muon_Phi", Xlabel=r"Muon $\phi$", rebin_factor=6, logY=False)
make_kinematicplot(var="dPhi_met_Muon", Xlabel=r"$\Delta \phi$ (Muon, MET)", rebin_factor=2, logY=False, xRange=[0., 3.2])
make_kinematicplot(var="dPhi_recoil_Muon", Xlabel=r"$\Delta \phi$ (Muon, Recoil)", rebin_factor=2, logY=False, xRange=[0., 3.2])

make_kinematicplot(var="FJet_pT", Xlabel=r"FatJet $p_{T}$ [GeV]", rebin_factor=5, logY=True, xRange=[0.,1000.])
make_kinematicplot(var="FJet_Eta", Xlabel=r"FatJet $\eta$", rebin_factor=6, logY=False)
make_kinematicplot(var="FJet_Phi", Xlabel=r"FatJet $\phi$", rebin_factor=6, logY=False)
make_kinematicplot(var="FJet_M", Xlabel=r"FatJet $Mass$ [GeV]", rebin_factor=4, logY=False, xRange=[0.,200.])
make_kinematicplot(var="FJet_Msd", Xlabel=r"FatJet $Mass_{softdrop}$ [GeV]", rebin_factor=4, logY=False, xRange=[0.,200.])
make_kinematicplot(var="dPhi_met_FJet", Xlabel=r"$\Delta \phi$ (FatJet, MET)", rebin_factor=2, logY=False, xRange=[0., 3.2])
make_kinematicplot(var="dPhi_Muon_FJet", Xlabel=r"$\Delta \phi$ (FatJet, Muon)", rebin_factor=2, logY=False, xRange=[0., 3.2])
make_kinematicplot(var="FJet_Area", Xlabel=r"FatJet Area", rebin_factor=5, logY=False)
make_kinematicplot(var="FJet_btagHbb", Xlabel=r"FatJet btagHbb", rebin_factor=2, logY=False)
make_kinematicplot(var="FJet_deepTag_H", Xlabel=r"FatJet deepTag_H", rebin_factor=2, logY=False)
make_kinematicplot(var="FJet_particleNet_HbbvsQCD", Xlabel=r"FatJet particleNet_HbbvsQCD", rebin_factor=2, logY=False)

make_kinematicplot(var="FJet_pT_BCatMinus2", Xlabel=r"FatJet $p_{T}$ [GeV]", rebin_factor=5, logY=True, xRange=[0.,1000.])
make_kinematicplot(var="FJet_Eta_BCatMinus2", Xlabel=r"FatJet $\eta$", rebin_factor=6, logY=False)
make_kinematicplot(var="FJet_Phi_BCatMinus2", Xlabel=r"FatJet $\phi$", rebin_factor=6, logY=False)
make_kinematicplot(var="FJet_M_BCatMinus2", Xlabel=r"FatJet $Mass$ [GeV]", rebin_factor=4, logY=False, xRange=[0.,200.])
make_kinematicplot(var="FJet_Msd_BCatMinus2", Xlabel=r"FatJet $Mass_{softdrop}$ [GeV]", rebin_factor=4, logY=False, xRange=[0.,200.])
make_kinematicplot(var="dPhi_met_FJet_BCatMinus2", Xlabel=r"$\Delta \phi$ (FatJet, MET)", rebin_factor=2, logY=False, xRange=[0., 3.2])
make_kinematicplot(var="dPhi_Muon_FJet_BCatMinus2", Xlabel=r"$\Delta \phi$ (FatJet, Muon)", rebin_factor=2, logY=False, xRange=[0., 3.2])
make_kinematicplot(var="FJet_Area_BCatMinus2", Xlabel=r"FatJet Area", rebin_factor=5, logY=False)
make_kinematicplot(var="FJet_btagHbb_BCatMinus2", Xlabel=r"FatJet btagHbb", rebin_factor=2, logY=False)
make_kinematicplot(var="FJet_deepTag_H_BCatMinus2", Xlabel=r"FatJet deepTag_H", rebin_factor=2, logY=False)
make_kinematicplot(var="FJet_particleNet_HbbvsQCD_BCatMinus2", Xlabel=r"FatJet particleNet_HbbvsQCD", rebin_factor=2, logY=False)

make_kinematicplot(var="bJet_N", Xlabel=r"N iso Loose b-tag jets", rebin_factor=1, logY=False)
make_kinematicplot(var="Jet_N", Xlabel=r"N iso additional jets", rebin_factor=1, logY=False)
make_kinematicplot(var="bJet_N_BCatMinus2", Xlabel=r"N iso Loose b-tag jets [AK8-Selection]", rebin_factor=1, logY=False)
make_kinematicplot(var="Jet_N_BCatMinus2", Xlabel=r"N iso additional jets [AK8-Selection]", rebin_factor=1, logY=False)
make_kinematicplot(var="bJet_N_BCatMinus1", Xlabel=r"N iso Loose b-tag jets [AK8-Selection + Nj$\leq$2]", rebin_factor=1, logY=False)

make_kinematicplot(var="Jet_pT", Xlabel=r"Jets $p_{T}$ [GeV]", rebin_factor=5, logY=False)
make_kinematicplot(var="Jet_Eta", Xlabel=r"Jets $\eta$", rebin_factor=6, logY=False)
make_kinematicplot(var="Jet_Phi", Xlabel=r"Jets $\phi$", rebin_factor=6, logY=False)
make_kinematicplot(var="Jet_M", Xlabel=r"Jets Mass [GeV]", rebin_factor=4, logY=False)
make_kinematicplot(var="Jet_btagDeepB", Xlabel=r"Jets btagDeepB", rebin_factor=2, logY=False)
make_kinematicplot(var="dPhi_met_Jet", Xlabel=r"$\Delta \phi$ (Jet, MET)", rebin_factor=2, logY=False)

make_kinematicplot(var="bJet_pT", Xlabel=r"bJets $p_{T}$ [GeV]", rebin_factor=5, logY=False)
make_kinematicplot(var="bJet_Eta", Xlabel=r"bJets $\eta$", rebin_factor=6, logY=False)
make_kinematicplot(var="bJet_Phi", Xlabel=r"bJets $\phi$", rebin_factor=6, logY=False)
make_kinematicplot(var="bJet_M", Xlabel=r"bJets Mass [GeV]", rebin_factor=4, logY=False)
make_kinematicplot(var="bJet_btagDeepB", Xlabel=r"bJets btagDeepB", rebin_factor=2, logY=False)
make_kinematicplot(var="dPhi_met_bJet", Xlabel=r"$\Delta \phi$ (bJet, MET)", rebin_factor=2, logY=False)

make_kinematicplot(var="Jet_pT_BCatMinus2", Xlabel=r"Jets $p_{T}$ [GeV] [AK8-Selection-only]", rebin_factor=5, logY=False)
make_kinematicplot(var="Jet_Eta_BCatMinus2", Xlabel=r"Jets $\eta$ [AK8-Selection-only]", rebin_factor=6, logY=False)
make_kinematicplot(var="Jet_Phi_BCatMinus2", Xlabel=r"Jets $\phi$ [AK8-Selection-only]", rebin_factor=6, logY=False)
make_kinematicplot(var="Jet_M_BCatMinus2", Xlabel=r"Jets Mass [GeV] [AK8-Selection-only]", rebin_factor=4, logY=False)
make_kinematicplot(var="Jet_btagDeepB_BCatMinus2", Xlabel=r"Jets btagDeepB [AK8-Selection-only]", rebin_factor=2, logY=False)
make_kinematicplot(var="dPhi_met_Jet_BCatMinus2", Xlabel=r"$\Delta \phi$ (Jet, MET) [AK8-Selection-only]", rebin_factor=2, logY=False)

make_kinematicplot(var="bJet_pT_BCatMinus2", Xlabel=r"bJets $p_{T}$ [GeV] [AK8-Selection-only]", rebin_factor=5, logY=False)
make_kinematicplot(var="bJet_Eta_BCatMinus2", Xlabel=r"bJets $\eta$ [AK8-Selection-only]", rebin_factor=6, logY=False)
make_kinematicplot(var="bJet_Phi_BCatMinus2", Xlabel=r"bJets $\phi$ [AK8-Selection-only]", rebin_factor=6, logY=False)
make_kinematicplot(var="bJet_M_BCatMinus2", Xlabel=r"bJets Mass [GeV] [AK8-Selection-only]", rebin_factor=4, logY=False)
make_kinematicplot(var="bJet_btagDeepB_BCatMinus2", Xlabel=r"bJets btagDeepB [AK8-Selection-only]", rebin_factor=2, logY=False)
make_kinematicplot(var="dPhi_met_bJet_BCatMinus2", Xlabel=r"$\Delta \phi$ (bJet, MET) [AK8-Selection-only]", rebin_factor=2, logY=False)

make_kinematicplot(var="bJet_pT_BCatMinus1", Xlabel=r"bJets $p_{T}$ [GeV] [AK8-Selection-only]", rebin_factor=5, logY=False)
make_kinematicplot(var="bJet_Eta_BCatMinus1", Xlabel=r"bJets $\eta$ [AK8-Selection-only]", rebin_factor=6, logY=False)
make_kinematicplot(var="bJet_Phi_BCatMinus1", Xlabel=r"bJets $\phi$ [AK8-Selection-only]", rebin_factor=6, logY=False)
make_kinematicplot(var="bJet_M_BCatMinus1", Xlabel=r"bJets Mass [GeV] [AK8-Selection-only]", rebin_factor=4, logY=False)
make_kinematicplot(var="bJet_btagDeepB_BCatMinus1", Xlabel=r"bJets btagDeepB [AK8-Selection-only]", rebin_factor=2, logY=False)

#----------------------------------------
def make_kinematicplot_2d(var, whichtau, Xlabel, rebin_factor, logY, xRange=None):

    hMC_ = outputHistMC[var][{"labelname": whichtau, "systematic": 'nominal'}][...,::hist.rebin(rebin_factor)]
    hData_ = outputHistData[var][{'dataset':sum}][{"labelname": whichtau, "systematic": 'noweight'}][...,::hist.rebin(rebin_factor)]
    figr, (ax) = plt.subplots(1)
    plotWithRatio(h=hMC_, hData=hData_, overlay='dataset', logY=logY, xLabel=Xlabel, xRange=xRange, colors_cat='Topmu', extraText=extraText, lumi=59.83, year=2018)
    plt.savefig('Plots_Test6b_BCatTopmu/FJet_'+whichtau+'_2018.png')

make_kinematicplot_2d(var="FJet_TauN", whichtau='tau1', Xlabel=r"FatJet $\tau_{1}$", rebin_factor=1, logY=False)
make_kinematicplot_2d(var="FJet_TauN", whichtau='tau2', Xlabel=r"FatJet $\tau_{2}$", rebin_factor=1, logY=False)
make_kinematicplot_2d(var="FJet_TauN", whichtau='tau3', Xlabel=r"FatJet $\tau_{3}$", rebin_factor=1, logY=False)
make_kinematicplot_2d(var="FJet_TauN", whichtau='tau4', Xlabel=r"FatJet $\tau_{4}$", rebin_factor=1, logY=False)
make_kinematicplot_2d(var="FJet_TauNM", whichtau='tau21', Xlabel=r"FatJet $\tau_{21}$", rebin_factor=1, logY=False)
make_kinematicplot_2d(var="FJet_TauNM", whichtau='tau31', Xlabel=r"FatJet $\tau_{31}$", rebin_factor=1, logY=False)
make_kinematicplot_2d(var="FJet_TauNM", whichtau='tau32', Xlabel=r"FatJet $\tau_{32}$", rebin_factor=1, logY=False)
make_kinematicplot_2d(var="FJet_n2b1_n3b1", whichtau='n2b1', Xlabel=r"FatJet $N_{2} \beta_{1}$", rebin_factor=1, logY=False)
make_kinematicplot_2d(var="FJet_n2b1_n3b1", whichtau='n3b1', Xlabel=r"FatJet $N_{3} \beta_{1}$", rebin_factor=1, logY=False)

make_kinematicplot_2d(var="FJet_TauN_BCatMinus2", whichtau='tau1', Xlabel=r"FatJet $\tau_{1}$", rebin_factor=1, logY=False)
make_kinematicplot_2d(var="FJet_TauN_BCatMinus2", whichtau='tau2', Xlabel=r"FatJet $\tau_{2}$", rebin_factor=1, logY=False)
make_kinematicplot_2d(var="FJet_TauN_BCatMinus2", whichtau='tau3', Xlabel=r"FatJet $\tau_{3}$", rebin_factor=1, logY=False)
make_kinematicplot_2d(var="FJet_TauN_BCatMinus2", whichtau='tau4', Xlabel=r"FatJet $\tau_{4}$", rebin_factor=1, logY=False)
make_kinematicplot_2d(var="FJet_TauNM_BCatMinus2", whichtau='tau21', Xlabel=r"FatJet $\tau_{21}$", rebin_factor=1, logY=False)
make_kinematicplot_2d(var="FJet_TauNM_BCatMinus2", whichtau='tau31', Xlabel=r"FatJet $\tau_{31}$", rebin_factor=1, logY=False)
make_kinematicplot_2d(var="FJet_TauNM_BCatMinus2", whichtau='tau32', Xlabel=r"FatJet $\tau_{32}$", rebin_factor=1, logY=False)
make_kinematicplot_2d(var="FJet_n2b1_n3b1_BCatMinus2", whichtau='n2b1', Xlabel=r"FatJet $N_{2} \beta_{1}$", rebin_factor=1, logY=False)
make_kinematicplot_2d(var="FJet_n2b1_n3b1_BCatMinus2", whichtau='n3b1', Xlabel=r"FatJet $N_{3} \beta_{1}$", rebin_factor=1, logY=False)


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
    plt.savefig('Plots_Test6b_BCatTopmu/'+str(var)+'_'+str(syste)+'_updown_2018.png') 

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
