# Everything in picobarns

hbb_BR = 0.582

#https://twiki.cern.ch/twiki/bin/viewauth/CMS/LumiRecommendationsRun2
#Use golden json (legacy) values
lumis = {
    2016: 35860.0,
    2017: 41480.0,
    2018: 59832.0,
}

crossSections = {

    ### 2017 ###

    #https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO #Total xsec: 833.9pb#
    #https://twiki.cern.ch/twiki/bin/view/CMS/SummaryTable1G25ns#TTbar (updated section)
    #@NNLO
    "TTToSemiLeptonic_17": 366.3 ,
    "TTTo2L2Nu_17": 88.5 ,
    "TTToHadronic_17": 379.1 ,

    #@NLO
    #from monojet
    #used until Test5e
    #"ST_tchannel_top_17": 137.458, 
    #"ST_tchannel_antitop_17": 83.0066,
    #"ST_tW_top_17": 35.85,
    #"ST_tW_antitop_17": 35.85,
    #@NNLO
    #https://twiki.cern.ch/twiki/bin/view/LHCPhysics/SingleTopNNLORef
    #used from Test6a
    "ST_tchannel_top_17": 134.2 ,
    "ST_tchannel_antitop_17": 80.0,
    "ST_tW_top_17": 39.65,
    "ST_tW_antitop_17": 39.65,

    #from monojet
    "Z1Jets_NuNu_ZpT_50To150_17": 598.9 ,
    "Z1Jets_NuNu_ZpT_150To250_17": 18.04 ,
    "Z1Jets_NuNu_ZpT_250To400_17": 2.051 ,
    "Z1Jets_NuNu_ZpT_400Toinf_17": 0.2251 ,
    "Z2Jets_NuNu_ZpT_50To150_17": 326.3 ,
    "Z2Jets_NuNu_ZpT_150To250_17": 29.6 ,
    "Z2Jets_NuNu_ZpT_250To400_17": 5.174 ,
    "Z2Jets_NuNu_ZpT_400Toinf_17": 0.8472 ,

    #from monojet
    #NLO
    "WJets_LNu_WPt_50To100_17": 3569.0,
    "WJets_LNu_WPt_100To250_17": 769.8,
    "WJets_LNu_WPt_250To400_17": 27.62,
    "WJets_LNu_WPt_400To600_17": 3.591,
    "WJets_LNu_WPt_600Toinf_17": 0.549,


    #from SUS-23-007 AN-22-181 table8:
    #Ref: https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV
    "DYJetsToLL_inclu_v1_17": 6025.2,   #6077.2, #NNLO
    "WJetsToLNu_incl_17": 61526.7, #NNLO

    #https://indico.cern.ch/event/1033836/contributions/4350242/attachments/2241954/3801452/GenMeeting_DYInclVsPtBinned_100521.pdf
    #mg265: https://cernbox.cern.ch/files/link/public/l5eFog2GLv7lmXj?tiles-size=1&items-per-page=100&view-mode=resource-table
    "DYJets_LL_ZpT_0To50_17": 5882.0,
    "DYJets_LL_ZpT_50To100_17": 392.1,
    "DYJets_LL_ZpT_100To250_17": 91.23,
    "DYJets_LL_ZpT_250To400_17": 3.499,
    "DYJets_LL_ZpT_400To650_17": 0.4764,
    "DYJets_LL_ZpT_650Toinf_17": 0.04489,
    
    #from SUS-23-002 AN-19-256-v14 table21&22
    #used from Test6a
    #@NLO
    "DYJets_LL_HT_70To100_17": 169.9,
    "DYJets_LL_HT_100To200_17": 161.1,
    "DYJets_LL_HT_200To400_17": 48.66,
    "DYJets_LL_HT_400To600_17": 6.968,
    "DYJets_LL_HT_600To800_17": 1.743,
    "DYJets_LL_HT_800To1200_17": 0.8052,
    "DYJets_LL_HT_1200To2500_17": 0.1933,
    "DYJets_LL_HT_2500ToInf_17": 0.003468,

    "DYJets_LL_M4to50_HT_70To100_17": 307.0,
    "DYJets_LL_M4to50_HT_100To200_17": 204.0,
    "DYJets_LL_M4to50_HT_200To400_17": 54.39,
    "DYJets_LL_M4to50_HT_400To600_17": 5.697,
    "DYJets_LL_M4to50_HT_600ToInf_17": 1.85,    


    #from Varun's repo: 
    #DY HTbinned is at LO accuracy:
    #used until Test5e
    #"DYJets_LL_HT_70To100_17": 147.0, #typo here?
    #"DYJets_LL_HT_100To200_17": 161.0, 
    #"DYJets_LL_HT_200To400_17": 48.58,
    #"DYJets_LL_HT_400To600_17": 6.983,
    #"DYJets_LL_HT_600To800_17": 1.747,
    #"DYJets_LL_HT_800To1200_17": 0.8052,
    #"DYJets_LL_HT_1200To2500_17": 0.1927,
    #"DYJets_LL_HT_2500ToInf_17": 0.003478,

    "QCD_HT100To200_17": 23680000.0,
    "QCD_HT200To300_17": 1556000.0,
    "QCD_HT300To500_17": 323600.0,
    "QCD_HT500To700_17": 29950.0,
    "QCD_HT700To1000_17": 6351.0,
    "QCD_HT1000To1500_17": 1094.0,
    "QCD_HT1500To2000_17": 98.99,
    "QCD_HT2000Toinf_17": 20.23,
    "WW_2L2Nu_17": 11.08,
    "WW_1L1Nu2Q_17": 45.99,
    "WW_4Q_17": 47.73,
    #from mono-Higgs AN:
    "WZ_1L1Nu2Q_17": 10.74,
    "WZ_2L2Q_17": 5.60,
    "WZ_2Q2Nu_17": 6.858,#not on DAS
    "WZ_3L1Nu_17" : 4.43,
    "ZZ_2L2Nu_17": 0.56,
    "ZZ_2L2Q_17": 3.22,
    "ZZ_2Q2Nu_17": 4.73,
    "ZZ_4L_17": 1.25,
    #SM HTobb processes: from mono-Higgs AN
    "VBFHToBB_17": 3.861,
    "ttHTobb_17": 0.5269,
    "WminusH_HToBB_WToLNu_17": 0.177,
    "WplusH_HToBB_WToLNu_17": 0.2819,
    "ggZH_HToBB_ZToNuNu_17": 0.01222,
    "ggZH_HToBB_ZToLL_17": 0.006185,
    "ZH_HToBB_ZToLL_17": 0.07924,
    "ZH_HToBB_ZToNuNu_17": 0.1565,


    ### 2018 ###

    #"TTToSemiLeptonic_18": 365.34 ,
    #"TTTo2L2Nu_18": 88.29 ,
    #"TTToHadronic_18": 377.96 ,
    #https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO #Total xsec: 833.9pb#
    #https://twiki.cern.ch/twiki/bin/view/CMS/SummaryTable1G25ns#TTbar (updated section)
    #@NNLO
    "TTToSemiLeptonic_18": 366.3 ,
    "TTTo2L2Nu_18": 88.5 ,
    "TTToHadronic_18": 379.1 ,

    #@NLO
    #from monojet
    #used until Test5e
    #"ST_tchannel_top_18": 137.458,
    #"ST_tchannel_antitop_18": 83.0066,
    #"ST_tW_top_18": 35.85,
    #"ST_tW_antitop_18": 35.85,
    #@NNLO
    #https://twiki.cern.ch/twiki/bin/view/LHCPhysics/SingleTopNNLORef
    #used from Test6a
    "ST_tchannel_top_18": 134.2 ,
    "ST_tchannel_antitop_18": 80.0,
    "ST_tW_top_18": 39.65,
    "ST_tW_antitop_18": 39.65,

    #from mono-jet AN:
    "Z1Jets_NuNu_ZpT_50To150_18": 598.9 ,
    "Z1Jets_NuNu_ZpT_150To250_18": 18.04 ,
    "Z1Jets_NuNu_ZpT_250To400_18": 2.051 ,
    "Z1Jets_NuNu_ZpT_400Toinf_18": 0.2251 ,
    "Z2Jets_NuNu_ZpT_50To150_18": 326.3 ,
    "Z2Jets_NuNu_ZpT_150To250_18": 29.6 ,
    "Z2Jets_NuNu_ZpT_250To400_18": 5.174 ,
    "Z2Jets_NuNu_ZpT_400Toinf_18": 0.8472 ,
    "WJets_LNu_WPt_50To100_18": 3569.0,
    "WJets_LNu_WPt_100To250_18": 769.8,
    "WJets_LNu_WPt_250To400_18": 27.62,
    "WJets_LNu_WPt_400To600_18": 3.591,
    "WJets_LNu_WPt_600Toinf_18": 0.549,

    #from SUS-23-007 AN-22-181 table8:
    #Ref: https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV
    "DYJetsToLL_inclu_v1_18": 6025.2,  #6077.2, #NNLO
    "DYJetsToLL_inclu_v2_18": 6025.2,   #6077.2, #NNLO
    "WJetsToLNu_incl_18": 61526.7, #NNLO

    #https://indico.cern.ch/event/1033836/contributions/4350242/attachments/2241954/3801452/GenMeeting_DYInclVsPtBinned_100521.pdf
    #mg265: https://cernbox.cern.ch/files/link/public/l5eFog2GLv7lmXj?tiles-size=1&items-per-page=100&view-mode=resource-table
    "DYJets_LL_ZpT_0To50_18": 5882.0,
    "DYJets_LL_ZpT_50To100_18": 392.1,
    "DYJets_LL_ZpT_100To250_18": 91.23,
    "DYJets_LL_ZpT_250To400_18": 3.499,
    "DYJets_LL_ZpT_400To650_18": 0.4764,
    "DYJets_LL_ZpT_650Toinf_18": 0.04489,


    #from SUS-23-002 AN-19-256-v14 table21&22
    "DYJets_LL_HT_70To100_18": 169.9,
    "DYJets_LL_HT_100To200_18": 161.1,
    "DYJets_LL_HT_200To400_18": 48.66,
    "DYJets_LL_HT_400To600_18": 6.968,
    "DYJets_LL_HT_600To800_18": 1.743,
    "DYJets_LL_HT_800To1200_18": 0.8052,
    "DYJets_LL_HT_1200To2500_18": 0.1933,
    "DYJets_LL_HT_2500ToInf_18": 0.003468,

    "DYJets_LL_M4to50_HT_70To100_18": 307.0,
    "DYJets_LL_M4to50_HT_100To200_18": 204.0,
    "DYJets_LL_M4to50_HT_200To400_18": 54.39,
    "DYJets_LL_M4to50_HT_400To600_18": 5.697,
    "DYJets_LL_M4to50_HT_600ToInf_18": 1.85,



    #DiBoson
    "WZ_1L1Nu2Q_18": 10.74,
    "WZ_2L2Q_18": 5.60,
    "WZ_2Q2Nu_18": 6.858,
    "WZ_3L1Nu_18" : 4.43,
    "ZZ_2L2Nu_18": 0.56,
    "ZZ_2L2Q_18": 3.22,
    "ZZ_2Q2Nu_18": 4.73,
    "ZZ_4L_18": 1.25,
    "WW_2L2Nu_18": 12.18,
    "WW_1L1Nu2Q_18": 50.00,
    #from Varun's repo:
    #"DYJets_LL_HT_70To100_18": 146.7,
    #"DYJets_LL_HT_100To200_18": 160.8,
    #"DYJets_LL_HT_200To400_18": 48.63,
    #"DYJets_LL_HT_400To600_18": 6.978,
    #"DYJets_LL_HT_600To800_18": 1.756,
    #"DYJets_LL_HT_800To1200_18": 0.8094,
    #"DYJets_LL_HT_1200To2500_18": 0.1931,
    #"DYJets_LL_HT_2500ToInf_18": 0.003516,
    "QCD_HT100To200_18": 23660000.0,
    "QCD_HT200To300_18": 1549000.0,
    "QCD_HT300To500_18": 323000.0,
    "QCD_HT500To700_18": 29960.0,
    "QCD_HT700To1000_18": 6353.0,
    "QCD_HT1000To1500_18": 1093.0,
    "QCD_HT1500To2000_18": 99.35,
    "QCD_HT2000Toinf_18": 20.25,

    #SM HTobb processes: from mono-Higgs AN
    "VBFHToBB_18": 3.861,
    "ttHTobb_18": 0.5269,
    "WminusH_HToBB_WToLNu_18": 0.177,
    "WplusH_HToBB_WToLNu_18": 0.2819,
    "ggZH_HToBB_ZToNuNu_18": 0.01222,
    "ggZH_HToBB_ZToLL_18": 0.006185,
    "ZH_HToBB_ZToLL_18": 0.07924,
    "ZH_HToBB_ZToNuNu_18": 0.1565,


    # Xsec values for (MZprime_MChi): 
    #0.05066(1500_100), #0.04976(1500_200), #0.01413(2000_200), #0.01425(2000_1), #0.2077(1000_100), #0.0505(1500_1), #0.05066(1500_100), #3.322 ,
    "MonoHTobb_ZpBaryonic_18": (0.582*0.05066), 
    "MonoHTobb_ZpBaryonic_17": (0.582*0.05066),

}

