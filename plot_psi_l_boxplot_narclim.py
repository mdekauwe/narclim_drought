#!/usr/bin/env python

"""
Plot SWP

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (18.10.2017)"
__email__ = "mdekauwe@gmail.com"

import xarray as xr
import matplotlib.pyplot as plt
import sys
import datetime as dt
import pandas as pd
import numpy as np
from matplotlib.ticker import FixedLocator
import datetime
import os
import glob
import seaborn as sns


def main(fname):

    df = pd.read_csv(fname)
    df = df[df.plc_max > -500.].reset_index()
    df['map'] = np.ones(len(df)) * np.nan

    df_map = pd.read_csv("euc_map.csv")
    df_map = df_map.sort_values(by=['map'])
    print(df_map)
    species = df_map.species
    species = species.str.replace("_", " ")
    species = species.str.replace("Eucalyptus", "E.")


    for i in range(len(df_map)):
        spp = df_map.species[i]
        map = df_map.map[i]

        for j in range(len(df)):

            if df.species[j] == spp.replace(" ", "_"):
                df['map'][j] = map


    #df = df.sort_values(by=['map'])

    for i in range(len(df)):
        print(i, df.species[i], df.psi_leaf_min[i])

    df['species'] = df['species'].str.replace("_", " ")
    df['species'] = df['species'].str.replace("Eucalyptus", "E.")
    #species = np.unique(df.species)



    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

    fig = plt.figure(figsize=(12,6))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    ax = fig.add_subplot(111)

    flierprops = dict(marker='o', markersize=3, markerfacecolor="black")

    ax = sns.boxplot(x="species", y="psi_leaf_min", data=df, flierprops=flierprops,
                     order=species)
    #ax = sns.swarmplot(x="species", y="plc_max", data=df, color=".25")
    ax.set_ylabel("$\Psi$$_{l}$ (MPa)")
    ax.set_xlabel(" ")

    ax.set_xticklabels(species, rotation=90)
    of = "/Users/mdekauwe/Desktop/psi_leaf_blah.png"
    plt.savefig(of, bbox_inches='tight', dpi=150, pad_inches=0.1)

if __name__ == "__main__":

    fname = "narclim_plcs.csv"
    main(fname)
