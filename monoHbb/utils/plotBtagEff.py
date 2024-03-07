from coffea import util
import awkward as ak
import numpy as np
import hist
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep
from plotting import GroupBy
from coffea.processor import accumulate
from coffea.lookup_tools import extractor, dense_lookup


#outputMC_17 = util.load("monoHbb/efficiencies/btagEfficiencyLMTwps_MCBkgs_UL2017.coffea")
#outputMC_18 = util.load("monoHbb/efficiencies/btagEfficiencyLMTwps_MCBkgs_UL2018.coffea")

outputMC_17 = util.load("monoHbb/efficiencies/btagEfficiencyLMTwps_MCBkgs_UL2017_10Mevts.coffea")
outputMC_18 = util.load("monoHbb/efficiencies/btagEfficiencyLMTwps_MCBkgs_UL2018_10Mevts.coffea")

#year='2018'
year='2017'

groupingMC_18 = {
    "WJets_WpT": [
        "WJets_LNu_WPt_100To250_18",
        "WJets_LNu_WPt_250To400_18",
        "WJets_LNu_WPt_400To600_18",
        "WJets_LNu_WPt_600Toinf_18",
    ],
    "ZJets_ZpT": [
        "Z1Jets_NuNu_ZpT_50To150_18",
        "Z1Jets_NuNu_ZpT_150To250_18",
        "Z1Jets_NuNu_ZpT_250To400_18",
        "Z1Jets_NuNu_ZpT_400Toinf_18",
        "Z2Jets_NuNu_ZpT_50To150_18",
        "Z2Jets_NuNu_ZpT_150To250_18",
        "Z2Jets_NuNu_ZpT_250To400_18",
        "Z2Jets_NuNu_ZpT_400Toinf_18",
    ],    
    "DYJets_HT": [
        "DYJets_LL_HT_70To100_18",
        "DYJets_LL_HT_100To200_18",
        "DYJets_LL_HT_200To400_18",
        "DYJets_LL_HT_400To600_18",
        "DYJets_LL_HT_600To800_18",
        "DYJets_LL_HT_800To1200_18",
        "DYJets_LL_HT_1200To2500_18",
        "DYJets_LL_HT_2500ToInf_18",
    ],
    "DYJets_ZpT": [
        "DYJets_LL_ZpT_0To50_18",
        "DYJets_LL_ZpT_50To100_18",
        "DYJets_LL_ZpT_100To250_18",
        "DYJets_LL_ZpT_250To400_18",
        "DYJets_LL_ZpT_400To650_18",
        "DYJets_LL_ZpT_650Toinf_18",
    ],
    "SingleTop": [
        "ST_tW_top_18",  
        "ST_tW_antitop_18",
        "ST_tchannel_top_18",  
        "ST_tchannel_antitop_18",  
    ],
    "tt": [
        "TTToSemiLeptonic_18",
        "TTToHadronic_18",
        "TTTo2L2Nu_18",
    ],
    "VV": [
        "WZ_2L2Q_18",
        "WZ_3L1Nu_18",
        "ZZ_2L2Q_18",
        "ZZ_4L_18",
        "WW_2L2Nu_18",
        "WZ_1L1Nu2Q_18",
        "ZZ_2L2Nu_18",
        "ZZ_2Q2Nu_18",
    ],
    "QCD": [
        "QCD_HT100To200_18",
        "QCD_HT200To300_18",
        "QCD_HT300To500_18",
        "QCD_HT500To700_18",
        "QCD_HT700To1000_18",
        "QCD_HT1000To1500_18",
        "QCD_HT1500To2000_18",
        "QCD_HT2000Toinf_18",
    ],
    "SMHiggs": [
        "ttHTobb_18",
        "WminusH_HToBB_WToLNu_18",
        "WplusH_HToBB_WToLNu_18",
        "ggZH_HToBB_ZToLL_18",
        "ZH_HToBB_ZToLL_18",
        "VBFHToBB_18",
        "ggZH_HToBB_ZToNuNu_18",
        "ZH_HToBB_ZToNuNu_18",        
    ],
}

groupingMC_17 = {
    "ZJets_ZpT": [
        "Z1Jets_NuNu_ZpT_50To150_17",
        "Z1Jets_NuNu_ZpT_150To250_17",
        "Z1Jets_NuNu_ZpT_250To400_17",
        "Z1Jets_NuNu_ZpT_400Toinf_17",
        "Z2Jets_NuNu_ZpT_50To150_17",
        "Z2Jets_NuNu_ZpT_150To250_17",
        "Z2Jets_NuNu_ZpT_250To400_17",
        "Z2Jets_NuNu_ZpT_400Toinf_17",
    ],    
    "DYJets_HT": [
        "DYJets_LL_HT_70To100_17",
        "DYJets_LL_HT_100To200_17",
        "DYJets_LL_HT_200To400_17",
        "DYJets_LL_HT_400To600_17",
        "DYJets_LL_HT_600To800_17",
        "DYJets_LL_HT_800To1200_17",
        "DYJets_LL_HT_1200To2500_17",
        "DYJets_LL_HT_2500ToInf_17",
    ],
    "SingleTop": [
        "ST_tW_top_17",  
        "ST_tW_antitop_17",
        "ST_tchannel_top_17",  
        "ST_tchannel_antitop_17",  
    ],
    "tt": [
        "TTToSemiLeptonic_17",
        "TTToHadronic_17",
        "TTTo2L2Nu_17",
    ],
    "VV": [
        "WW_2L2Nu_17",
        "WW_1L1Nu2Q_17",
        "WW_4Q_17",
        "WZ_1L1Nu2Q_17",
        "WZ_2L2Q_17",
        "WZ_3L1Nu_17",
        "ZZ_2L2Nu_17",
        "ZZ_2L2Q_17",
        "ZZ_2Q2Nu_17",
        "ZZ_4L_17",
    ],
    "QCD": [
        "QCD_HT100To200_17",
        "QCD_HT200To300_17",
        "QCD_HT300To500_17",
        "QCD_HT500To700_17",
        "QCD_HT700To1000_17",
        "QCD_HT1000To1500_17",
        "QCD_HT1500To2000_17",
        "QCD_HT2000Toinf_17",
    ],
    "SMHiggs": [
        "ttHTobb_17",
        "WminusH_HToBB_WToLNu_17",
        "WplusH_HToBB_WToLNu_17",
        "ggZH_HToBB_ZToLL_17",
        "ZH_HToBB_ZToLL_17",
        "VBFHToBB_17",
        "ggZH_HToBB_ZToNuNu_17",
        "ZH_HToBB_ZToNuNu_17",        
    ],
}
groupingMC_17_TT = {
    "tt1l": [
        "TTToSemiLeptonic_17",
    ],
    "tt2l": [
        "TTTo2L2Nu_17",
    ],
    "tt0l": [
        "TTToHadronic_17",
    ],
}


if year == '2018':
    outputMC = outputMC_18
    groupingMC = groupingMC_18
elif year == '2017':
    outputMC = outputMC_17
    groupingMC = groupingMC_17
    #groupingMC = groupingMC_17_TT

histList = []
for samp, sampList in groupingMC.items():
    histList += [outputMC[s] for s in sampList]    
outputHistMC = accumulate(histList)
for key, histo in outputHistMC.items():
    if isinstance(histo, hist.Hist):
        outputHistMC[key] = GroupBy(histo, 'dataset', 'dataset', groupingMC)


# ------------------------------------------------------------------------------

btagEff_ptBins = np.array([0, 30, 50, 70, 100, 140, 200, 300, 500, 1000])
btagEff_etaBins = np.array([0., 0.5, 1., 1.5, 2., 2.5])

#Plot btag efficiency for light, c and b jets 
def plot_btagEff(bjetWP, WP, year):

    for dataset_name in groupingMC.keys():
        light_total = outputHistMC["hJets"][{"dataset": dataset_name, "flavor": slice(0j, 4j,sum)}]
        c_total = outputHistMC["hJets"][{"dataset": dataset_name, "flavor": slice(4j, 5j,sum)}]
        b_total = outputHistMC["hJets"][{"dataset": dataset_name, "flavor": slice(5j, 6j,sum)}]
        light_tagged = outputHistMC[bjetWP][{"dataset": dataset_name, "flavor": slice(0j, 4j,sum)}]
        c_tagged = outputHistMC[bjetWP][{"dataset": dataset_name, "flavor": slice(4j, 5j,sum)}]
        b_tagged = outputHistMC[bjetWP][{"dataset": dataset_name, "flavor": slice(5j, 6j,sum)}]        

        Eff_1 = light_tagged.project("pt", "eta")/light_total.project("pt", "eta")
        Eff_2 = c_tagged.project("pt", "eta")/c_total.project("pt", "eta")
        Eff_3 = b_tagged.project("pt", "eta")/b_total.project("pt", "eta")

        fig, (ax) = plt.subplots(1,3, figsize=(15, 5), sharey=True, sharex=True)
        fig.subplots_adjust(wspace=0.15)

        hep.hist2dplot(Eff_1, ax=ax[0], vmin=0.0, vmax=1.0)
        hep.hist2dplot(Eff_2, ax=ax[1], vmin=0.0, vmax=1.0)
        hep.hist2dplot(Eff_3, ax=ax[2], vmin=0.0, vmax=1.0)

        ax[0].set_title("HadronFlavor=0 (light)", fontsize=15)
        ax[1].set_title("HadronFlavor=4 (c)", fontsize=15)
        ax[2].set_title("HadronFlavor=5 (b)", fontsize=15)
        ax[0].set_xlim(0,1000)
        ax[1].set_xlim(0,1000)
        ax[2].set_xlim(0,1000)
        ax[0].set_ylim(0,2.5)
        ax[1].set_ylim(0,2.5)
        ax[2].set_ylim(0,2.5)
        ax[2].set_xlabel(r"$p_{T}$ [GeV]", fontsize=14)
        ax[0].set_ylabel(r"$\eta$", fontsize=14)
        ax[0].set_xlabel(r" ", fontsize=14)
        ax[2].set_ylabel(r" ", fontsize=14)
        ax[1].set_xlabel(r" ", fontsize=14)
        ax[1].set_ylabel(r" ", fontsize=14)
        plt.savefig(f'monoHbb/utils/plotsBtag/{year}_btagEff{WP}_MC{dataset_name}_10Mevts.png')

if year == '2018':
    plot_btagEff(bjetWP='hBJets_looseWP', WP='LooseWP', year='2018')
    plot_btagEff(bjetWP='hBJets_mediumWP', WP='MediumWP', year='2018')
    plot_btagEff(bjetWP='hBJets_tightWP', WP='TightWP', year='2018')

elif year == '2017':
    plot_btagEff(bjetWP='hBJets_looseWP', WP='LooseWP', year='2017')
    plot_btagEff(bjetWP='hBJets_mediumWP', WP='MediumWP', year='2017')
    plot_btagEff(bjetWP='hBJets_tightWP', WP='TightWP', year='2017')
