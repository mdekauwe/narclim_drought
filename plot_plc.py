#!/usr/bin/env python

"""
Plot visual benchmark (average seasonal cycle) of old vs new model runs.

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
from optparse import OptionParser

def main(fname1, plot_fname=None):

    df1 = read_cable_file(fname1)

    plc_mean = np.mean(df1.resample("D").agg("mean").values)
    plc_max = np.max(df1.resample("D").agg("max").values)
    print(plc_mean, plc_max)


    df1 = resample_timestep(df1)

    #df1 = df1[(df1.index.hour >= 12) & (df1.index.hour < 13) &
    #           (df1.index.minute < 30)].copy()

    fig = plt.figure(figsize=(9,6))
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

    ax1 = fig.add_subplot(1,1,1)


    axes = [ax1]

    vars = ["plc"]
    for a,  v in zip(axes,  vars):

        #a.plot(df1[v].index.to_pydatetime(), df1[v].rolling(window=7).mean(), c=colours[2],
        #       lw=1.5, ls="-", label="Hydraulics")

        a.plot(df1[v].index.to_pydatetime(), df1[v], c=colours[2],
               lw=1.5, ls="-", label="Hydraulics")

        #x.bar(df_met.index, df_met["Rainf"], alpha=0.3, color="black")
    #ax1.set_ylim(0, 0.2)

    labels = ["PLC (%)"]
    for a, l in zip(axes, labels):
        a.set_ylabel(l, fontsize=12)

    #for x, l in zip(axes2, labels):
    #    x.set_ylabel("Rainfall (mm d$^{-1}$)", fontsize=12)


    #plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.legend(numpoints=1, loc="best")

    for a in axes:
        a.set_xlim([datetime.date(1992,1,1), datetime.date(1995, 12, 31)])

    #for a in axes:
        #a.set_xlim([datetime.date(2002,10,1), datetime.date(2003, 4, 1)])
    #    a.set_xlim([datetime.date(2002,12,1), datetime.date(2003, 5, 1)])
        #a.set_xlim([datetime.date(2006,11,1), datetime.date(2007, 4, 1)])
        #a.set_xlim([datetime.date(2012,7,1), datetime.date(2013, 8, 1)])
        #a.set_xlim([datetime.date(2006,11,1), datetime.date(2007, 4, 1)])

    if plot_fname is None:
        plt.show()
    else:
        #fig.autofmt_xdate()
        fig.savefig(plot_fname, bbox_inches='tight', pad_inches=0.1)


def read_cable_file(fname):
    vars_to_keep = ['plc']
    ds = xr.open_dataset(fname, decode_times=False)

    time_jump = int(ds.time[1].values) - int(ds.time[0].values)
    if time_jump == 3600:
        freq = "H"
    elif time_jump == 1800:
        freq = "30M"
    else:
        raise("Time problem")

    units, reference_date = ds.time.attrs['units'].split('since')
    df = ds[vars_to_keep].squeeze(dim=["x","y"], drop=True).to_dataframe()
    start = reference_date.strip().split(" ")[0].replace("-","/")
    df['dates'] = pd.date_range(start=start, periods=len(df), freq=freq)
    df = df.set_index('dates')

    return df



def resample_timestep(df):

    UMOL_TO_MOL = 1E-6
    MOL_C_TO_GRAMS_C = 12.0
    SEC_2_HLFHOUR = 1800.
    method = {"plc":"max"}


    return df


    return dates

if __name__ == "__main__":


    #fname = "outputs/hydraulics_profitmax.nc"
    #fname = "debug/Eucalyptus_cladocalyx_10_out.nc"
    fname = "debug/Eucalyptus_viminalis_64_out.nc"
    #fname = "outputs/1990-2009/CCCMA3.1/R1/Eucalyptus_viminalis_64_profitmax_out.nc"
    main(fname)
