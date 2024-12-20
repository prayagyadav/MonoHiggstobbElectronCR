from coffea import util
import hist
import coffea.processor as processor
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.nanoevents.methods import nanoaod, vector
from coffea.lookup_tools import extractor, dense_lookup
from collections import defaultdict
import awkward as ak
import numpy as np
import pickle

class BjetEfficiencies(processor.ProcessorABC):
    def __init__(self):
        ak.behavior.update(nanoaod.behavior)

        dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary dataset")
        jetPt_axis = hist.axis.Variable([0,30,50,70,100,140,200,300,500,1000], name="pt", label=r"$p_{T}$ [GeV]")  
        jetEta_axis = hist.axis.Variable([0,0.5,1.,1.5,2.,2.5], name="eta", label=r"$\eta$")
        jetFlav_axis = hist.axis.Variable([0,4.,5.,6.], name="flavor", label=r"hadron flavor")

        self.make_output = lambda: {
            'hJets': hist.Hist(dataset_axis, jetPt_axis, jetEta_axis, jetFlav_axis, label="Counts"),
            'hBJets_looseWP': hist.Hist(dataset_axis, jetPt_axis, jetEta_axis, jetFlav_axis, label="Counts"),
            'hBJets_mediumWP': hist.Hist(dataset_axis, jetPt_axis, jetEta_axis, jetFlav_axis, label="Counts"),
            'hBJets_tightWP': hist.Hist(dataset_axis, jetPt_axis, jetEta_axis, jetFlav_axis, label="Counts"),
        }
        
    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):

        output = self.make_output()
        cutflow = defaultdict(int)
        cutflow['TotalEvents'] += len(events)
        dataset = events.metadata['dataset']

        
        ##################
        # Jet selections #
        jets = events.Jet
        jetSelect = (
            (jets.pt > 30) & (abs(jets.eta) < 2.5) & (jets.jetId >= 2)
        )

        btag_WP_loose = 0.0532
        btag_WP_medium = 0.3040
        btag_WP_tight = 0.7476

        Jets = jets[jetSelect]
        bJets_looseWP = jets[jetSelect & (jets.btagDeepFlavB > btag_WP_loose)]
        bJets_mediumWP = jets[jetSelect & (jets.btagDeepFlavB > btag_WP_medium)]
        bJets_tightWP = jets[jetSelect & (jets.btagDeepFlavB > btag_WP_tight)]
        
        output['hJets'].fill(
            dataset=dataset,
            pt=ak.flatten(Jets.pt),
            eta=ak.flatten(abs(Jets.eta)),
            flavor=ak.flatten(Jets.hadronFlavour),
        )
        output['hBJets_looseWP'].fill(
            dataset=dataset,
            pt=ak.flatten(bJets_looseWP.pt),
            eta=ak.flatten(abs(bJets_looseWP.eta)),
            flavor=ak.flatten(bJets_looseWP.hadronFlavour),
        )
        output['hBJets_mediumWP'].fill(
            dataset=dataset,
            pt=ak.flatten(bJets_mediumWP.pt),
            eta=ak.flatten(abs(bJets_mediumWP.eta)),
            flavor=ak.flatten(bJets_mediumWP.hadronFlavour),
        )
        output['hBJets_tightWP'].fill(
            dataset=dataset,
            pt=ak.flatten(bJets_tightWP.pt),
            eta=ak.flatten(abs(bJets_tightWP.eta)),
            flavor=ak.flatten(bJets_tightWP.hadronFlavour),
        )

        return {dataset:output}


    def postprocess(self, accumulator):
        return accumulator


from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from fileList_MCbkg_17 import *
listOfFiles = fileset_MC_17

runner = processor.Runner(
    executor=processor.FuturesExecutor(workers=30),
    schema=NanoAODSchema,
    chunksize=100000,
    maxchunks=100,
    skipbadfiles=True,
    xrootdtimeout=1000,
)
output = runner(
    listOfFiles,
    treename="Events",
    processor_instance=BjetEfficiencies(),
)

#util.save(output, "monoHbb/efficiencies/btagEfficiencyLMTwps_MCBkgs_UL2017.coffea")
util.save(output, "monoHbb/efficiencies/btagEfficiencyLMTwps_MCBkgs_UL2017_10Mevts.coffea")


#--------------------------------------------------------------------------
# Save in taggingEfficienciesDenseLookup.coffea file

#for Loose WP b-tag jets:
btagEff_ptBins = np.array([0, 30, 50, 70, 100, 140, 200, 300, 500, 1000])
btagEff_etaBins = np.array([0., 0.5, 1., 1.5, 2., 2.5])
btagEff = []
samples = []
for dataset_name in listOfFiles.keys():    
    l_total = output[dataset_name]["hJets"][{"flavor": slice(0j, 4j,sum)}]
    c_total = output[dataset_name]["hJets"][{"flavor": slice(4j, 5j,sum)}]
    b_total = output[dataset_name]["hJets"][{"flavor": slice(5j, 6j,sum)}]
    l_tagged = output[dataset_name]["hBJets_looseWP"][{"flavor": slice(0j, 4j,sum)}]
    c_tagged = output[dataset_name]["hBJets_looseWP"][{"flavor": slice(4j, 5j,sum)}]
    b_tagged = output[dataset_name]["hBJets_looseWP"][{"flavor": slice(5j, 6j,sum)}]
    l_Total = l_total.values()
    c_Total = c_total.values()
    b_Total = b_total.values()
    l_Tagged = l_tagged.values()
    c_Tagged = c_tagged.values()
    b_Tagged = b_tagged.values()
    btagEff3 = ak.where((b_Tagged > 0) & (b_Total > 0), b_Tagged/b_Total, 0.0)
    btagEff2 = ak.where((c_Tagged > 0) & (c_Total > 0), c_Tagged/c_Total, 0.0)
    btagEff1 = ak.where((l_Tagged > 0) & (l_Total > 0), l_Tagged/l_Total, 0.0)
    btagEff.append( np.array([btagEff1[0], btagEff2[0], btagEff3[0]]) )
    samples.append(l_total.axes[0][0])
taggingEffLookup = dense_lookup.dense_lookup(np.array(btagEff), (samples, [0,4,5], btagEff_ptBins, btagEff_etaBins))
#with open("monoHbb/scalefactors/taggingEfficienciesDenseLookupLooseWP_MCBkgs_UL2017.pkl", "wb") as _file:
with open("monoHbb/scalefactors/taggingEfficienciesDenseLookupLooseWP_MCBkgs_UL2017_10Mevts.pkl", "wb") as _file:
    pickle.dump(taggingEffLookup, _file)

#------------------------------------------------------------------------------------------------------

#same for Medium WP b-tag jets:
btagEff_ptBins = np.array([0, 30, 50, 70, 100, 140, 200, 300, 500, 1000])
btagEff_etaBins = np.array([0., 0.5, 1., 1.5, 2., 2.5])
btagEff = []
samples = []
for dataset_name in listOfFiles.keys():    
    l_total = output[dataset_name]["hJets"][{"flavor": slice(0j, 4j,sum)}]
    c_total = output[dataset_name]["hJets"][{"flavor": slice(4j, 5j,sum)}]
    b_total = output[dataset_name]["hJets"][{"flavor": slice(5j, 6j,sum)}]
    l_tagged = output[dataset_name]["hBJets_mediumWP"][{"flavor": slice(0j, 4j,sum)}]
    c_tagged = output[dataset_name]["hBJets_mediumWP"][{"flavor": slice(4j, 5j,sum)}]
    b_tagged = output[dataset_name]["hBJets_mediumWP"][{"flavor": slice(5j, 6j,sum)}]
    l_Total = l_total.values()
    c_Total = c_total.values()
    b_Total = b_total.values()
    l_Tagged = l_tagged.values()
    c_Tagged = c_tagged.values()
    b_Tagged = b_tagged.values()
    btagEff3 = ak.where((b_Tagged > 0) & (b_Total > 0), b_Tagged/b_Total, 0.0)
    btagEff2 = ak.where((c_Tagged > 0) & (c_Total > 0), c_Tagged/c_Total, 0.0)
    btagEff1 = ak.where((l_Tagged > 0) & (l_Total > 0), l_Tagged/l_Total, 0.0)
    btagEff.append( np.array([btagEff1[0], btagEff2[0], btagEff3[0]]) )
    samples.append(l_total.axes[0][0])
taggingEffLookup = dense_lookup.dense_lookup(np.array(btagEff), (samples, [0,4,5], btagEff_ptBins, btagEff_etaBins))
#with open("monoHbb/scalefactors/taggingEfficienciesDenseLookupMediumWP_MCBkgs_UL2017.pkl", "wb") as _file:
with open("monoHbb/scalefactors/taggingEfficienciesDenseLookupMediumWP_MCBkgs_UL2017_10Mevts.pkl", "wb") as _file:
    pickle.dump(taggingEffLookup, _file)


#------------------------------------------------------------------------------------------------------

#same for Tight WP b-tag jets:
btagEff_ptBins = np.array([0, 30, 50, 70, 100, 140, 200, 300, 500, 1000])
btagEff_etaBins = np.array([0., 0.5, 1., 1.5, 2., 2.5])
btagEff = []
samples = []
for dataset_name in listOfFiles.keys():    
    l_total = output[dataset_name]["hJets"][{"flavor": slice(0j, 4j,sum)}]
    c_total = output[dataset_name]["hJets"][{"flavor": slice(4j, 5j,sum)}]
    b_total = output[dataset_name]["hJets"][{"flavor": slice(5j, 6j,sum)}]
    l_tagged = output[dataset_name]["hBJets_tightWP"][{"flavor": slice(0j, 4j,sum)}]
    c_tagged = output[dataset_name]["hBJets_tightWP"][{"flavor": slice(4j, 5j,sum)}]
    b_tagged = output[dataset_name]["hBJets_tightWP"][{"flavor": slice(5j, 6j,sum)}]
    l_Total = l_total.values()
    c_Total = c_total.values()
    b_Total = b_total.values()
    l_Tagged = l_tagged.values()
    c_Tagged = c_tagged.values()
    b_Tagged = b_tagged.values()
    btagEff3 = ak.where((b_Tagged > 0) & (b_Total > 0), b_Tagged/b_Total, 0.0)
    btagEff2 = ak.where((c_Tagged > 0) & (c_Total > 0), c_Tagged/c_Total, 0.0)
    btagEff1 = ak.where((l_Tagged > 0) & (l_Total > 0), l_Tagged/l_Total, 0.0)
    btagEff.append( np.array([btagEff1[0], btagEff2[0], btagEff3[0]]) )
    samples.append(l_total.axes[0][0])
taggingEffLookup = dense_lookup.dense_lookup(np.array(btagEff), (samples, [0,4,5], btagEff_ptBins, btagEff_etaBins))
#with open("monoHbb/scalefactors/taggingEfficienciesDenseLookupTightWP_MCBkgs_UL2017.pkl", "wb") as _file:
with open("monoHbb/scalefactors/taggingEfficienciesDenseLookupTightWP_MCBkgs_UL2017_10Mevts.pkl", "wb") as _file:
    pickle.dump(taggingEffLookup, _file)

