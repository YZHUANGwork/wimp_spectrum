import numpy as np
import os

import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import matplotlib.cm as cm

def setup_cdfpdf_ax(ax, title, xlabel, ylabel, gridTF, cdfpdf, label_size, font_size, 
                    vlines = [0], hlines = [0], xlims = [0,0], ylims = [0,0], log = [0,0]):
    ax.yaxis.set_tick_params(labelsize=label_size) 
    ax.xaxis.set_tick_params(labelsize=label_size)  
    ax.set_title(title, fontsize = font_size)
    ax.grid(gridTF)
    if cdfpdf in ['cdf']:
        ylabel = 'Event rate '+ r'[ton$^{-1}$ yr$^{-1}$]'
    elif cdfpdf in ['pdf']:
        ylabel = r'$\dfrac{d{\cal R}}{dE_{r}}$  '+ '[ton$^{-1}$ yr$^{-1}$ '+'keV'+'$^{-1}$]'
    elif cdfpdf in ['pmf']:
        ylabel = r'$d{\cal R}(E_{r})$  '+ '[ton$^{-1}$ yr$^{-1}$ '+']'
    ax.set_xlabel(xlabel, fontsize = label_size)
    ax.set_ylabel(ylabel, fontsize = label_size)
    
    if len(xlims)>0:
        if  xlims[0] != xlims[1]:
            ax.set_xlim(xlims[0], xlims[1])
    if len(ylims)>0:
        if ylims[0] != ylims[1]:
            ax.set_ylim(ylims[0], ylims[1])
    if len(vlines)>0:
        for vline in vlines:
            ax.axvline(x = vline, lw = 3, ls = '--', color = 'black')
    if len(hlines)>0:
        for hline in hlines:
            ax.axhline(y = hline, lw = 3, ls = '--', color = 'black')
    if log[0]!=0:
        ax.set_xscale('log')
    if log[1]!=0:
        ax.set_yscale('log')
    return ax 

    
def get_isotope_color(isotope):
    if isotope == 'Xe129':
        color = 'yellowgreen'
    elif isotope == 'Xe131':
        color = 'forestgreen'
    elif isotope == 'Xe132':
        color = 'darkturquoise'
    elif isotope == 'Xe134':
        color = 'indigo'
    elif isotope == 'Xe136':
        color = 'magenta'
    elif isotope == 'Cs131':
        color = 'darkred'
    elif isotope == 'Cs136':
        color = 'goldenrod'

    return color


def get_WIMPmass_colors(WIMPMASS):
    if '6GeV' in WIMPMASS:
        c = 'lightcoral'
    elif '10GeV' in WIMPMASS:
        c = 'darkred'
    elif '20GeV' in WIMPMASS:
        c = 'orange'
    elif '50GeV' in WIMPMASS:
        c = 'goldenrod'
    elif '100GeV' in WIMPMASS:
        c = 'olive'
    elif '500GeV' in WIMPMASS:
        c = 'yellowgreen'
    elif '1000GeV' in WIMPMASS:
        c = 'lightseagreen'
    elif '5000GeV' in WIMPMASS:
        c = 'darkcyan'
    elif '10000GeV' in WIMPMASS:
        c = 'violet'
    return c


def get_color_index(i):
    if i == 0:
        return 'blue'
    elif i == 1:
        return 'orange'
    elif i == 2:
        return 'brown'
    elif i == 3:
        return 'yellowgreen'
    
    
    
def assign_color(arr, color_arr_dict, colormap = cm.viridis_r):
    colors = colormap(np.linspace(0, 1, len(arr)))
    
    for color, ele in zip(colors, arr):
        color_arr_dict[ele] =  color
    return color_arr_dict


def get_ratio_color(ratio):
    if ratio == 0.00001:
        color =  'midnightblue'
    if ratio == 0.0001:
        color = 'indigo'
    elif ratio == 0.001:
        color = 'rebeccapurple' 
    elif ratio == 0.01:
        color = 'mediumvioletred'
    elif ratio == 0.1:
        color = 'dodgerblue'
    elif ratio == 1:
        color = 'lime'
    elif ratio == 10:
        color ='deeppink'
    elif ratio == 100:
        color = 'tomato'
    elif ratio == 1000:
        color = 'gold'
    elif ratio == 10000:
        color = 'chocolate'
    elif ratio == 100000:
        color = 'firebrick'
    elif ratio == 1000000:
        color = 'brown'
    elif np.isinf(ratio):
        color = 'grey'
    return color