import correctionlib

filename = "monoHbb/scalefactors/POG/JME/2018_UL/jet_jerc.json"
#filename = "monoHbb/scalefactors/POG/JME/2018_UL/fatJet_jerc.json"
#filename = "/afs/hep.wisc.edu/home/slomte/monoHbb_coffea/trial/testJSONPOG/jsonpog-integration/POG/JME/2018_UL/fatJet_jerc.json"

ceval = correctionlib.CorrectionSet.from_file(filename)
list(ceval.keys())

for corr in ceval.values():
    print(corr.name, ' \n')
