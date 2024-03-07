import os.path
import correctionlib

from coffea import util
from coffea.lookup_tools import extractor

import numpy as np

def getPUSF(nTrueInt, era, var='nominal'):

    ## json files from: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/LUM

    if era==2017: 
        fname='monoHbb/scalefactors/POG/LUM/2017_UL/puWeights.json'
        hname = "Collisions17_UltraLegacy_goldenJSON"
    elif era==2018: 
        fname='monoHbb/scalefactors/POG/LUM/2018_UL/puWeights.json'
        hname = "Collisions18_UltraLegacy_goldenJSON"
    else: 
        raise Exception(f"Error: Unknown era \"{era}\".")

    evaluator = correctionlib.CorrectionSet.from_file(fname)
    return evaluator[hname].evaluate(np.array(nTrueInt), var)



