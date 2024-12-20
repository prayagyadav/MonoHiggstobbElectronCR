import numpy as np
import os
import correctionlib 

btvjson = correctionlib.CorrectionSet.from_file("monoHbb/scalefactors/POG/BTV/2018_UL/btagging.json")
'''
# case 1: fixedWP correction with mujets (here medium WP)
# evaluate('systematic', 'working_point', 'flavor', 'abseta', 'pt')
bc_jet_sf = btvjson["deepJet_mujets"].evaluate("central", "M",
            5, 1.2, 60.)
light_jet_sf = btvjson["deepJet_incl"].evaluate("central", "M",
            0, 2.2, 100.)
print("\njet SFs for mujets at medium WP:")
print("SF b/c: {bc_jet_sf}".format(bc_jet_sf=bc_jet_sf))
print("SF light: {light_jet_sf}".format(light_jet_sf=light_jet_sf))

# case 2: fixedWP correction uncertainty (here tight WP and comb SF)
# evaluate('systematic', 'working_point', 'flavor', 'abseta', 'pt')
bc_jet_sf = btvjson["deepJet_comb"].evaluate("up_correlated", "T", 
            5, 1.2, 60.)
light_jet_sf = btvjson["deepJet_incl"].evaluate("up_correlated", "T", 
            0, 2.2, 100.)
print("\njet SF up_correlated for comb at tight WP:")
print("SF b/c: {bc_jet_sf}".format(bc_jet_sf=bc_jet_sf))
print("SF light: {light_jet_sf}".format(light_jet_sf=light_jet_sf))

# case 3: shape correction SF
# evaluate('systematic', 'flavor', 'eta', 'pt', 'discriminator')
jet_sf = btvjson["deepJet_shape"].evaluate("central",
        5, 1.2, 60., 0.95)
print("\njet SF for shape correction:")
print("SF: {jet_sf}".format(jet_sf=jet_sf))

# case 4: shape correction SF uncertainties
# evaluate('systematic', 'flavor', 'eta', 'pt', 'discriminator')
b_jet_sf = btvjson["deepJet_shape"].evaluate("up_hfstats2",
        5, 1.2, 60., 0.95)
c_jet_sf = btvjson["deepJet_shape"].evaluate("up_cferr1", 
        4, 2.2, 100., 0.45)
print("\njet SF up_hfstats2 for shape correction b/light jets:")
print("SF b/light: {b_jet_sf}".format(b_jet_sf=b_jet_sf))
print("jet SF up_cferr1 for shape correction c jets:")
print("SF c: {c_jet_sf}".format(c_jet_sf=c_jet_sf))
'''


#------------------------------
bc_jet_sf = btvjson["deepJet_comb"].evaluate("central", "M", [[5,5],[5,5]], [[1.3,1],[0.2,0.8]], [[35., 70], [89,100]])
print(bc_jet_sf, ' \n')


'''
list(btvjson.keys())
for corr in btvjson.values():
    print(corr.name, ' \n')
'''
