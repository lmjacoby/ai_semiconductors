#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def plot_energy(Impurity, AB, Site):
    # Transition energy values
    # create xticks and labels for each method
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import preprocessing
    try:
        TL_energies, TL_std, FM_energies,\
            FM_std, PBE_gap, DFT_exist =\
            preprocessing.preprocessing(Impurity, AB, Site)
    except (IndexError, ValueError, TypeError):
        # raise ValueError('Try to input again')
        return
    x = [i for i in range(len(TL_energies))]

    minor_ticks = np.arange(np.min(TL_energies)-1, np.max(TL_energies)+1, 0.5)

    if DFT_exist:
        lab = ['DFT', 'KRR', 'RFR', 'NN']
    else:
        lab = ['KRR', 'RFR', 'NN']
    # plot predicted or DFT values for each output
    fig, ax = plt.subplots(figsize=(18, 12))

    if DFT_exist:
        z = np.array([0, 1, 2, 3] * 6)
    else:
        z = np.array([1, 2, 3] * 6)
    colors = np.array(['black', 'green', 'orange', 'blue'])
    ax.scatter(x, TL_energies, s=1444,
               marker="_", linewidth=3, zorder=3, c=colors[z])
    ax.set_yticks(minor_ticks, minor=False)

    # plot uncertainty errorbar
    ax.errorbar(x, TL_energies, yerr=TL_std, fmt='none', c=colors[z])
    ax.plot(np.linspace(-1, 24, 100),
            [PBE_gap]*100, label='CBM', color='r', linewidth=3, alpha=0.2)
    ax.plot(np.linspace(-1, 24, 100), [0]*100,
            label='VBM', color='r', linewidth=3, alpha=0.2)
    ax.margins(0.2)
    ax.set_ylabel('Transition level (ev)')
    ax.set_title('Transition level Prediction')

    # ylabel create
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([PBE_gap, 0])
    ax2.set_yticklabels(['CBM', 'VBM'])

    # xlabel create
    if DFT_exist:
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(ticks=[1.5, 5.5, 9.5, 13.5, 17.5, 21.5])
        ax2.set_xticklabels(['(+3,+2)', '(+2,+1)', '(+1,0)',
                             '(0,-1)', '(-1,-2)', '(-2,-3)'])
        ax2.set_xticks([3.5, 7.5, 11.5, 15.5, 19.5], minor=True)
    else:
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(ticks=[1, 4, 7, 10, 13, 16])
        ax2.set_xticklabels(['(+3,+2)', '(+2,+1)', '(+1,0)',
                             '(0,-1)', '(-1,-2)', '(-2,-3)'])
        ax2.set_xticks([2.5, 5.5, 8.5, 11.5, 14.5], minor=True)
    ax.grid(which='minor', alpha=0.8)
    ax.set_xlim(-1, len(TL_energies))

    # legend
    if DFT_exist:
        custom_lines = [Line2D([0], [0], color='black', lw=4),
                        Line2D([0], [0], color='green', lw=4),
                        Line2D([0], [0], color='orange', lw=4),
                        Line2D([0], [0], color='blue', lw=4)]
    else:
        custom_lines = [Line2D([0], [0], color='green', lw=4),
                        Line2D([0], [0], color='orange', lw=4),
                        Line2D([0], [0], color='blue', lw=4)]
    ax.legend(custom_lines, lab, loc='upper left', prop={'size': 15})
    # Formation Energy
    x = [i for i in range(len(FM_energies))]
    minor_ticks = np.arange(np.min(FM_energies)-2, np.max(FM_energies)+1, 0.5)
    # plot predicted or DFT values for each output
    fig, ax = plt.subplots(figsize=(18, 12))

    if DFT_exist:
        z = np.array([0, 1, 2, 3]*2)
    else:
        z = np.array([1, 2, 3]*2)
    colors = np.array(['black', 'green', 'orange', 'blue'])
    ax.scatter(x, FM_energies, s=300, marker=".",
               linewidth=1, zorder=3, c=colors[z])
    ax.set_yticks(minor_ticks, minor=False)

    # plot uncertainty errorbar
    ax.errorbar(x, FM_energies, yerr=FM_std, fmt='none', c=colors[z])
    ax.margins(0.2)
    ax.set_ylabel('Formation Energy (ev)')
    ax.set_title('Formation Energy Prediction')

    # xlabel create
    if DFT_exist:
        ax2 = ax.twinx()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(ticks=[1, 6])
        ax2.set_xticklabels(['dH (A-rich)', 'dH (B-rich)'])
        ax2.set_xticks([3.5], minor=True)
    else:
        ax2 = ax.twinx()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(ticks=[0.8, 4.2])
        ax2.set_xticklabels(['dH (A-rich)', 'dH (B-rich)'])
        ax2.set_xticks([2.5], minor=True)
    ax2.axes.get_yaxis().set_visible(False)
    ax.grid(which='minor', alpha=0.8)
    ax.set_xlim(-1, len(FM_energies))

    # legend
    if DFT_exist:
        custom_lines = [Line2D([0], [0], color='black', lw=4),
                        Line2D([0], [0], color='green', lw=4),
                        Line2D([0], [0], color='orange', lw=4),
                        Line2D([0], [0], color='blue', lw=4)]
    else:
        custom_lines = [Line2D([0], [0], color='green', lw=4),
                        Line2D([0], [0], color='orange', lw=4),
                        Line2D([0], [0], color='blue', lw=4)]
    ax.legend(custom_lines, lab, loc='upper left', prop={'size': 15})
    plt.show()
    if DFT_exist:
        summary = {'DFT': [], 'KRR': [], 'RFR': [], 'NN': []}
        for i in range(0, 2):
            summary['DFT'].append(np.str(round(FM_energies[i*2], 2)) +
                                  ' +/- ' + np.str(round(FM_std[i*2], 2)))
            summary['KRR'].append(np.str(round(FM_energies[i*2+1], 2)) +
                                  ' +/- ' + np.str(round(FM_std[i*2+1], 2)))
            summary['RFR'].append(np.str(round(FM_energies[i*2+2], 2)) +
                                  ' +/- ' + np.str(round(FM_std[i*2+2], 2)))
            summary['NN'].append(np.str(round(FM_energies[i*2+3], 2)) +
                                 ' +/- ' + np.str(round(FM_std[i*2+3], 2)))
        for i in range(0, 6):
            summary['DFT'].append(np.str(round(TL_energies[i*4], 2)) +
                                  ' +/- ' + np.str(round(TL_std[i*4], 2)))
            summary['KRR'].append(np.str(round(TL_energies[i*4+1], 2)) +
                                  ' +/- ' + np.str(round(TL_std[i*4+1], 2)))
            summary['RFR'].append(np.str(round(TL_energies[i*4+2], 2)) +
                                  ' +/- ' + np.str(round(TL_std[i*4+2], 2)))
            summary['NN'].append(np.str(round(TL_energies[i*4+3], 2)) +
                                 ' +/- ' + np.str(round(TL_std[i*4+3], 2)))
    else:
        summary = {'KRR': [], 'RFR': [], 'NN': []}
        for i in range(0, 2):
            summary['KRR'].append(np.str(round(FM_energies[i*2], 2)) +
                                  ' +/- ' + np.str(round(FM_std[i*2], 2)))
            summary['RFR'].append(np.str(round(FM_energies[i*2+1], 2)) +
                                  ' +/- ' + np.str(round(FM_std[i*2+1], 2)))
            summary['NN'].append(np.str(round(FM_energies[i*2+2], 2)) +
                                 ' +/- ' + np.str(round(FM_std[i*2+2], 2)))
        for i in range(0, 6):
            summary['KRR'].append(np.str(round(TL_energies[i*3], 2)) +
                                  ' +/- ' + np.str(round(TL_std[i*3], 2)))
            summary['RFR'].append(np.str(round(TL_energies[i*3+1], 2)) +
                                  ' +/- ' + np.str(round(TL_std[i*3+1], 2)))
            summary['NN'].append(np.str(round(TL_energies[i*3+2], 2)) +
                                 ' +/- ' + np.str(round(TL_std[i*3+2], 2)))
    print(pd.DataFrame(summary, index=['dHA', 'dHB', '(+3,+2)', '(+2,+1)',
                                       '(+1,0)', '(0,-1)',
                                       '(-1,-2)', '(-2,-3)']))
    return
