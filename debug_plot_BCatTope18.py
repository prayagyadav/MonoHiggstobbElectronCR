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

outputData = accumulate([
    util.load("coffea_files/ver2/output_DataA_BCatTope2018_run20240305_140317.coffea"),
    util.load("coffea_files/ver2/output_DataB_BCatTope2018_run20240305_135829.coffea"),
    util.load("coffea_files/ver2/output_DataC_BCatTope2018_run20240305_140012.coffea"),
    util.load("coffea_files/ver2/output_DataD_BCatTope2018_run20240305_142908.coffea"),
])
outputMC = accumulate([
    util.load("coffea_files/ver2/output_MCTTbar1l1v_BCatTope2018_run20240305_140626.coffea"),
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
        #"TTTo2L2Nu_18",
        #"TTToHadronic_18",
    ],
}


def makehist(histo,sample, isData,jettype,coord,time):
    #print(isData," "key," ",jettype," ",coord," ",time)
    #print(histo)
    fig,ax=plt.subplots()
    hep.histplot(
        histo,
        ax=ax
    )
    ax.set_xlabel(coord)
    ax.set_ylabel(f"Number of {jettype} jets")
    ax.set_title(f"{sample} {jettype} {coord} {time} HEM Veto")
    filename = f"debug_{sample}_{jettype}_{coord}_{time}.png"
    path = "plots/ver2/"
    fig.savefig(path+filename)
    print(filename, " created at ", path)
    plt.close()


for key in outputData.keys():
    for subkey in outputData[key].keys():
        if subkey.startswith('debug'):
            arr = subkey.split("_")
            #print(arr)
            makehist(outputData[key][subkey],sample=key, isData=True,jettype=arr[1],coord=arr[2],time=arr[3])
for key in outputMC.keys():
    for subkey in outputMC[key].keys():
        if subkey.startswith('debug'):
            arr = subkey.split("_")
            #print(arr)
            makehist(outputMC[key][subkey],sample=key, isData=False,jettype=arr[1],coord=arr[2],time=arr[3])







