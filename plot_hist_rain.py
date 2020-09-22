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

def extract_data(df, dfa):

    # driest summer?
    dfa = dfa.sort_values("Rainf")
    #dfa = dfa.sort_values("weighted_psi_soil")
    year = dfa.head(1).index.year.values[0] # driest year

    rf_dry = round(dfa.head(1).Rainf.values[0],1) # driest year
    rf_wet = round(dfa.tail(1).Rainf.values[0],1) # wettest year

    other_psi_leaf = df[df.index.year != year]
    dry_psi_leaf = df[df.index.year == year]

    # we now have 3 RCMs per GCM for this year, just keep the lowest RCM psi
    # leaf
    #dry_psi_leaf = dry_psi_leaf.sort_values("weighted_psi_soil")
    #num = dry_psi_leaf.head(1).num.values[0]
    #rcm = dry_psi_leaf.head(1).rcm.values[0]
    #dry_psi_leaf = dry_psi_leaf[(dry_psi_leaf.num == num) &
    #                            (dry_psi_leaf.rcm == rcm)]

    return (dry_psi_leaf, other_psi_leaf, rf_dry, rf_wet)



def main(spp, plot_dir, GCMs, RCMs, time_slice, odir):

    spp = spp.replace(" ", "_")

    files = []
    GCM = GCMs[0]
    for RCM in RCMs:
        path = "%s/%s/%s/%s" % (odir, time_slice, GCM, RCM)
        files.append( glob.glob(os.path.join(path, '%s_*.nc' % (spp))) )

    rf_dry_GCM1 = np.nan
    rf_wet_GCM1 = np.nan
    # flatten list
    files = [item for sublist in files for item in sublist]

    if len(files) > 0:

        data = []

        for fn in files:
            try:
                num = os.path.basename(fn).split("_")[2]
                rcm = fn.split("/")[3]

                dfx, dfa = read_cable_file(fn)

                dfx['species'] = spp
                dfx['num'] = num
                dfx['rcm'] = rcm

                # store DataFrame in list
                data.append(dfx)

            except:
                pass

        if len(data) > 0:
            df = pd.concat(data)
            (dry_psi_leaf_GCM1, other_psi_leaf_GCM1,
             rf_dry_GCM1, rf_wet_GCM1) = extract_data(df, dfa)
    else:
        dry_psi_leaf_GCM1 = np.nan
        other_psi_leaf_GCM1 = np.nan



    GCM = GCMs[1]
    for RCM in RCMs:
        path = "%s/%s/%s/%s" % (odir, time_slice, GCM, RCM)
        files.append( glob.glob(os.path.join(path, '%s_*.nc' % (spp))) )

    # flatten list
    files = [item for sublist in files for item in sublist]

    rf_dry_GCM2 = np.nan
    rf_wet_GCM2 = np.nan
    if len(files) > 0:

        data = []

        for fn in files:
            try:
                num = os.path.basename(fn).split("_")[2]
                rcm = fn.split("/")[3]

                dfx, dfa = read_cable_file(fn)

                dfx['species'] = spp
                dfx['num'] = num
                dfx['rcm'] = rcm

                # store DataFrame in list
                data.append(dfx)

            except:
                pass

        if len(data) > 0:
            df = pd.concat(data)
            (dry_psi_leaf_GCM2, other_psi_leaf_GCM2,
             rf_dry_GCM2, rf_wet_GCM2) = extract_data(df, dfa)
    else:
        dry_psi_leaf_GCM2 = np.nan
        other_psi_leaf_GCM2 = np.nan


    GCM = GCMs[2]
    for RCM in RCMs:
        path = "%s/%s/%s/%s" % (odir, time_slice, GCM, RCM)
        files.append( glob.glob(os.path.join(path, '%s_*.nc' % (spp))) )

    # flatten list
    files = [item for sublist in files for item in sublist]

    rf_dry_GCM3 = np.nan
    rf_wet_GCM3 = np.nan
    if len(files) > 0:

        data = []

        for fn in files:
            try:
                num = os.path.basename(fn).split("_")[2]
                rcm = fn.split("/")[3]

                dfx, dfa = read_cable_file(fn)

                dfx['species'] = spp
                dfx['num'] = num
                dfx['rcm'] = rcm

                # store DataFrame in list
                data.append(dfx)

            except:
                pass

        if len(data) > 0:
            df = pd.concat(data)
            (dry_psi_leaf_GCM3, other_psi_leaf_GCM3,
             rf_dry_GCM3, rf_wet_GCM3) = extract_data(df, dfa)
    else:
        dry_psi_leaf_GCM3 = np.nan
        other_psi_leaf_GCM3 = np.nan


    GCM = GCMs[3]
    for RCM in RCMs:
        path = "%s/%s/%s/%s" % (odir, time_slice, GCM, RCM)
        files.append( glob.glob(os.path.join(path, '%s_*.nc' % (spp))) )

    # flatten list
    files = [item for sublist in files for item in sublist]
    rf_dry_GCM4 = np.nan
    rf_wet_GCM4 = np.nan

    if len(files) > 0:

        data = []

        for fn in files:
            try:
                num = os.path.basename(fn).split("_")[2]
                rcm = fn.split("/")[3]

                dfx, dfa = read_cable_file(fn)

                dfx['species'] = spp
                dfx['num'] = num
                dfx['rcm'] = rcm

                # store DataFrame in list
                data.append(dfx)

            except:
                pass

        if len(data) > 0:
            df = pd.concat(data)
            (dry_psi_leaf_GCM4, other_psi_leaf_GCM4,
             rf_dry_GCM4, rf_wet_GCM4) = extract_data(df, dfa)
    else:
        dry_psi_leaf_GCM4 = np.nan
        other_psi_leaf_GCM4 = np.nan


    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

    fig = plt.figure(figsize=(9,10))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.2)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    colours = plt.cm.Set2(np.linspace(0, 1, 7))

    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)
    print(spp)

    try:
        sns.distplot(other_psi_leaf_GCM1.Rainf.values, label="Other", color="royalblue", ax=ax1)
        sns.distplot(dry_psi_leaf_GCM1.Rainf.values, label="Dry", color="orange", ax=ax1)
    except:
        pass

    ax1.legend(numpoints=1, loc=(0.75, 0.1), fontsize=10)
    ax1.set_ylabel(" ")
    ax1.set_xlabel(" ")

    try:
        sns.distplot(other_psi_leaf_GCM2.Rainf.values, label="Other", color="royalblue", ax=ax2)
        sns.distplot(dry_psi_leaf_GCM2.Rainf.values, label="Dry", color="orange", ax=ax2)
    except:
        pass
    ax2.set_ylabel("Density", position=(0.5, -0.1))
    ax2.set_xlabel(" ")

    try:
        sns.distplot(other_psi_leaf_GCM3.Rainf.values, label="Other", color="royalblue", ax=ax3)
        sns.distplot(dry_psi_leaf_GCM3.Rainf.values, label="Dry", color="orange", ax=ax3)
    except:
        pass
    ax3.set_ylabel(" ")
    ax3.set_xlabel(" ")

    try:
        sns.distplot(other_psi_leaf_GCM4.Rainf.values, label="Other", color="royalblue", ax=ax4)
        sns.distplot(dry_psi_leaf_GCM4.Rainf.values, label="Dry", color="orange", ax=ax4)
    except:
        pass
    ax4.set_ylabel(" ")
    ax4.set_xlabel("Rain (mm month$^{-1}$)")

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    ax1.set_xlim(-100.0, 700.0)
    ax2.set_xlim(-100.0, 700.0)
    ax3.set_xlim(-100.0, 700.0)
    ax4.set_xlim(-100.0, 700.0)

    props = dict(boxstyle='round', facecolor='white', alpha=0.0, ec="white")
    ax1.text(0.72, 0.97, "CCCMA3.1: %.0f / %.0f mm" % (rf_dry_GCM1, rf_wet_GCM1),
             transform=ax1.transAxes, fontweight="bold",
             fontsize=10, verticalalignment='top', bbox=props)
    ax2.text(0.72, 0.97, "CSIRO-MK3.0: %.0f / %.0f mm" % (rf_dry_GCM2, rf_wet_GCM2),
             transform=ax2.transAxes,fontweight="bold",
             fontsize=10, verticalalignment='top', bbox=props)
    ax3.text(0.72, 0.97, "ECHAM5: %.0f / %.0f mm" % (rf_dry_GCM3, rf_wet_GCM3),
             transform=ax3.transAxes, fontweight="bold",
             fontsize=10, verticalalignment='top', bbox=props)
    ax4.text(0.72, 0.97, "MIROC3.2: %.0f / %.0f mm" % (rf_dry_GCM4, rf_wet_GCM4),
             transform=ax4.transAxes, fontweight="bold",
             fontsize=10, verticalalignment='top', bbox=props)

    ofname = os.path.join(plot_dir, "%s_rain_hist.png" % (spp))
    fig.savefig(ofname, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def read_cable_file(fname):

    #vars_to_keep = ['weighted_psi_soil','psi_leaf','psi_soil',\
    #                'Rainf','SoilMoist', 'TVeg']
    vars_to_keep = ['weighted_psi_soil','psi_leaf','Rainf','TVeg']

    ds = xr.open_dataset(fname, decode_times=False)

    time_jump = int(ds.time[1].values) - int(ds.time[0].values)
    if time_jump == 3600:
        freq = "H"
    elif time_jump == 1800:
        freq = "30M"
    else:
        raise("Time problem")

    units, reference_date = ds.time.attrs['units'].split('since')
    ds = ds[vars_to_keep].squeeze(dim=["x","y","patch"], drop=True)

    """
    ds['psi_soil1'] = ds['psi_soil'][:,0]
    ds['psi_soil2'] = ds['psi_soil'][:,1]
    ds['psi_soil3'] = ds['psi_soil'][:,2]
    ds['psi_soil4'] = ds['psi_soil'][:,3]
    ds['psi_soil5'] = ds['psi_soil'][:,4]
    ds['psi_soil6'] = ds['psi_soil'][:,5]
    ds = ds.drop("psi_soil")
    """

    ds['Rainf'] *= float(time_jump)
    ds['TVeg'] *= float(time_jump)

    """
    # layer thickness
    zse = [.022, .058, .154, .409, 1.085, 2.872]

    frac1 = zse[0] / (zse[0] + zse[1])
    frac2 = zse[1] / (zse[0] + zse[1])
    frac3 = zse[2] / (zse[2] + zse[3])
    frac4 = zse[3] / (zse[2] + zse[3])
    frac5 = zse[4] / (zse[4] + zse[4])
    frac6 = zse[5] / (zse[5] + zse[5])
    ds['theta1'] = (ds['SoilMoist'][:,0] * frac1) + \
                   (ds['SoilMoist'][:,1] * frac2)
    ds['theta2'] = (ds['SoilMoist'][:,2] * frac3) + \
                   (ds['SoilMoist'][:,3] * frac4)
    ds['theta3'] = (ds['SoilMoist'][:,4] * frac4) + \
                   (ds['SoilMoist'][:,5] * frac5)
    ds = ds.drop("SoilMoist")
    """

    df = ds.to_dataframe()

    start = reference_date.strip().split(" ")[0].replace("-","/")
    df['dates'] = pd.date_range(start=start, periods=len(df), freq=freq)

    # correct dates to join S.H. years...
    df['dates'] =  df['dates'] + pd.DateOffset(months=+6) # shift to join S.H years

    #df['dates2'] =  df['dates'] + pd.DateOffset(months=+5) # shift to join S.H years
    #for i in range(len(df)):
    #    print(df['dates'].values[i], df['dates2'].values[i])

    df = df.set_index('dates')

    years = np.unique(df.index.year)
    first = years[0]
    last = years[-1]
    df = df.loc[ (df.index.year > first) & (df.index.year < last) ]


    dfa = df.resample("A").agg({'Rainf': np.sum, 'TVeg': np.sum,
                                'psi_leaf': np.min,
                                'weighted_psi_soil': np.min})

    ### Just keep summers

    #Dec = 6 (june); Jan = 7 (july); Feb = 8 (aug)
    #df = df.loc[ (df.index.month == 6) | (df.index.month == 7) | \
    #             (df.index.month == 8) ]

    ### Just keep spring/summer
    #df = df.loc[ (df.index.month >=3) & (df.index.month <=8)]

    df = df.resample("M").agg({'Rainf': np.sum, 'TVeg': np.sum,
                                'psi_leaf': np.min,
                                'weighted_psi_soil': np.min})

    return df, dfa


if __name__ == "__main__":

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    odir = "outputs"

    RCMs = ["R1", "R2", "R3"]
    GCMs = ["CCCMA3.1", "CSIRO-MK3.0", "ECHAM5", "MIROC3.2"]
    time_slices = ["1990-2009", "2020-2039", "2060-2079"]

    time_slice = time_slices[0]

    df_spp = pd.read_csv("euc_species_traits.csv")
    species = np.unique(df_spp.species)

    for spp in species:
        main(spp, plot_dir, GCMs, RCMs, time_slice, odir)
        sys.exit()
