import hist
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import itertools
from hist import intervals
from hist.intervals import ratio_uncertainty as ratio_Uncertainty
import awkward as ak
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# Thanks for this, Nick and Andrzej ;)
def GroupBy(h, oldname, newname, grouping):
    hnew = hist.Hist(
        hist.axis.StrCategory(grouping, name=newname),
        *(ax for ax in h.axes if ax.name != oldname),
        #storage=h._storage_type,
        storage=h.storage_type,
    )
    for i, indices in enumerate(grouping.values()):
        hnew.view(flow=True)[i] = h[{oldname: indices}][{oldname: sum}].view(flow=True)
    return hnew



def plotWithRatio(
        h,
        hData=None,
        xLabel,
        overlay,
        xRange,
        stacked=True,
        invertStack=False,
        density=False,
        lumi=59.64,
        year=2018,
        label="CMS Preliminary",
        xlabel="",
        ratioRange=[0.5, 1.5],
        yRange=None,
        logY=False,
        colors_cat='Topmu',
        extraText=r'Top($\mu$) CR ',
        leg="upper right",
        binwnorm=None
):
    #print("stage 1")
    colors_CRTopmu = ['darkblue', 'maroon', 'red']
    colors_CRZmumu = ['forestgreen']
    #colors_CRZmumu = ['darkblue', 'maroon', 'red', 'mediumaquamarine', 'forestgreen']
    fill_opts_CRTopmu = {'edgecolor': (0,0,0,0.3), 'alpha': 1., 'facecolor': colors_CRTopmu}
    fill_opts_CRZmumu = {'edgecolor': (0,0,0,0.3), 'alpha': 1., 'facecolor': colors_CRZmumu}
    data_err_opts = {'linestyle': 'none', 'marker': '.', 'markersize': 10., 'color': 'k', 'elinewidth': 0.1,}
    ratio_opts = {'linestyle': 'none', 'marker': '.', 'markersize': 6., 'color': 'k', 'elinewidth': 0.1,}


    # make a nice ratio plot
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 18,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.handletextpad": 0.1,
            "legend.handlelength": 1,
            "legend.handleheight": 1.25,
        }
    )


    if not hData is None:
        fig, (ax, rax) = plt.subplots(
            2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True
        )
        fig.subplots_adjust(hspace=0.05)
    else:
        fig, ax = plt.subplots(
            1, 1, figsize=(7, 7)
        )  # , gridspec_kw={"height_ratios": (3, 1)}, sharex=True)

    #print("stage 2")
    # Here is an example of setting up a color cycler to color the various fill patches
    # http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=6
    from cycler import cycler

    
    #if not colors is None:
    if invertStack:
        _n = len(h.identifiers(overlay)) - 1
        colors = colors[_n::-1]
    
    if colors_cat=='Zmumu':
        ax.set_prop_cycle(cycler(color=colors_CRZmumu))
        colors_category=colors_CRZmumu
    if colors_cat=='Topmu':
        ax.set_prop_cycle(cycler(color=colors_CRTopmu))
        colors_category=colors_CRTopmu
    
    h.plot(
        overlay=overlay,
        ax=ax,
        stack=stacked,
        density=density,
        histtype='fill',
        binwnorm=binwnorm,
        edgecolor=None,
        alpha=0.9,
        facecolor=colors_category,
    )
    
    #print("stage 3")
    if not hData is None:
        #pass
        
        if binwnorm:
            
            mcStatUp = np.append((h[{overlay:sum}].values() + np.sqrt(h[{overlay:sum}].variances()))/np.diff(hData.axes[0].edges),[0])
            mcStatDo = np.append((h[{overlay:sum}].values() - np.sqrt(h[{overlay:sum}].variances()))/np.diff(hData.axes[0].edges),[0])
        
            uncertainty_band = ax.fill_between(
                hData.axes[0].edges,
                mcStatUp,
                mcStatDo,
                step='post',
                hatch='///',
                facecolor='none',
                edgecolor='gray',
                linewidth=0,
        )
        else:
            
            mcStatUp = np.append(h[{overlay:sum}].values() + np.sqrt(h[{overlay:sum}].variances()),[0])
            mcStatDo = np.append(h[{overlay:sum}].values() - np.sqrt(h[{overlay:sum}].variances()),[0])
        
            uncertainty_band = ax.fill_between(
                hData.axes[0].edges,
                mcStatUp,
                mcStatDo,
                step='post',
                hatch='///',
                facecolor='none',
                edgecolor='gray',
                linewidth=0,
        )

    #print("stage 4")
    if not hData is None:
        
        if binwnorm:
            ax.errorbar(x=hData.axes[0].centers,
                        y=hData.values()/np.diff(hData.axes[0].edges),
                        xerr=np.diff(hData.axes[0].edges)/2,
                        yerr=np.sqrt(hData.values())/np.diff(hData.axes[0].edges),
                        color='black',
                        marker='.',
                        markersize=8,
                        linewidth=0,
                        elinewidth=0.5,
                        label="Data",
            )
        else:
            ax.errorbar(x=hData.axes[0].centers,
                        y=hData.values(),
                        xerr=np.diff(hData.axes[0].edges)/2,
                        yerr=np.sqrt(hData.values()),
                        color='black',
                        marker='.',
                        markersize=8,
                        linewidth=0,
                        elinewidth=1,
                        label="Data",
            )
    
    #print("stage 5")
    if not binwnorm is None:
        ax.set_ylabel(f"<Events/{binwnorm}>")
        if "[" in ax.get_xlabel():
            units = ax.get_xlabel().split("[")[-1].split("]")[0]
            ax.set_ylabel(f"<Events / {binwnorm} {units}>")
    else:
        ax.set_ylabel('Events')

    ax.autoscale(axis="x", tight=True)
    #ax.set_ylim(0, None)

    ax.set_xlabel(None)
    if hData is None:
        ax.set_xlabel(xlabel)

    if leg == "right":
        leg_anchor = (1.0, 1.0)
        leg_loc = "upper left"
    elif leg == "upper right":
        leg_anchor = (1.0, 1.0)
        leg_loc = "upper right"
    elif leg == "upper left":
        leg_anchor = (0.0, 1.0)
        leg_loc = "upper left"

    if not leg is None:
        ax.legend(bbox_to_anchor=leg_anchor, loc=leg_loc,fontsize=10)
    
        
    ratio_mcStatUp = np.append(1 + np.sqrt(h[{overlay:sum}].variances())/h[{overlay:sum}].values(),[0])
    ratio_mcStatDo = np.append(1 - np.sqrt(h[{overlay:sum}].variances())/h[{overlay:sum}].values(),[0])
    
    #print("stage 6")
    if not hData is None:
        
        ratio_uncertainty_band = rax.fill_between(
            hData.axes[0].edges,
            ratio_mcStatUp,
            ratio_mcStatDo,
            step='post',
            color='lightgray',
        )

    if not hData is None:
        
        hist_1_values, hist_2_values = hData.values(), h[{overlay:sum}].values()
        #ratios = hist_1_values/hist_1_values
        ratios = ak.where(hist_2_values>0, hist_1_values/hist_2_values, 10)
        
        ratio_uncert = ratio_Uncertainty(
            num=hist_1_values,
            denom=hist_2_values,
            uncertainty_type="poisson",
        )
        '''
        ratio: plot the ratios using Matplotlib errorbar or bar
        hist.plot.plot_ratio_array(
            hData, ratios, ratio_uncert, ax=rax, uncert_draw_type='line',
        );
        '''

        
        #print("stage 7")
        rax.errorbar(x=hData.axes[0].centers, 
                     y=ratios, 
                     xerr=np.diff(hData.axes[0].edges)/2,
                     yerr=ratio_uncert,
                     color='black', 
                     marker='.', 
                     markersize=8, 
                     linewidth=0, 
                     elinewidth=1, 
                     label="Ratio",
        )

        rax.set_xlabel(xLabel,fontsize=16,loc="right")
        rax.set_ylim(ratioRange[0], ratioRange[1])
        rax.axhline(y=1, color='black', linestyle='dashed')
        rax.axhline(y=1.1, color='grey', linestyle='dotted')
        rax.axhline(y=0.9, color='grey', linestyle='dotted')
        rax.axhline(y=1.2, color='grey', linestyle='dotted')
        rax.axhline(y=0.8, color='grey', linestyle='dotted')
        rax.axhline(y=1.3, color='grey', linestyle='dotted')
        rax.axhline(y=0.7, color='grey', linestyle='dotted')
        rax.set_ylabel('Data/MC')


    if logY:
        ax.set_yscale("log")
        ax.set_ylim(1, ax.get_ylim()[1] * 10)
        ax.yaxis.set_minor_locator(AutoMinorLocator(10))
        #ax.yaxis.set_minor_locator(MultipleLocator(10))
    else:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.5)


    if not xRange is None:
        ax.set_xlim(xRange[0], xRange[1])
    if not yRange is None:
        ax.set_ylim(yRange[0], yRange[1])

    #print("stage 8")
    CMS = plt.text(
        0.01, #0.02,
        1.0, #0.89,
        r"$\bf{CMS}$ Work in Progress",
        fontsize=14,
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )

    if not extraText is None:

        extraLabel = plt.text(
            0.05, #0.0,
            0.80, #1.0,
            extraText,
            fontsize=14,
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
        )

    lumi = plt.text(
        1.0,
        1.0,
        r"%.2f fb$^{-1}$ (%d)" % (lumi,year),
        fontsize=13,
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )


    #print("stage 9")
    ax.legend(fontsize=14, ncol=2, loc='upper right', frameon=False, bbox_to_anchor=(1.0, 1.0)).shadow=False
    ax.tick_params(axis='both', direction='in', length=5, which='minor')
    rax.tick_params(axis='both', direction='in', length=5, which='minor')
    ax.tick_params(axis='both', direction='in', length=10, which='major')
    rax.tick_params(axis='both', direction='in', length=10, which='major')
    ax.tick_params(top=True, right=True, which='both')
    rax.tick_params(top=True, right=True, which='both')
    ax.minorticks_on()
    rax.minorticks_on()

    
    #print("stage 10")
