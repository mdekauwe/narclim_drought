#!/usr/bin/env python

"""
Turn the MAESPA input file into a CABLE netcdf file. Aim to swap MAESPA data
for the raw data later when I have more time...

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (04.08.2018)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import pandas as pd
import numpy as np
import xarray as xr
import datetime
import matplotlib.pyplot as plt

path = "/Users/mdekauwe/research/narclim_data/1990-2009/CCCMA3.1/R1"
fname = os.path.join(path, "narclim_met_Eucalyptus_viminalis_64.nc")

#"""
vars_to_keep = ['Rainf']
ds = xr.open_dataset(fname, decode_times=False)
print(ds.latitude.values[0][0], ds.longitude.values[0][0])
freq = "H"
units, reference_date = ds.time.attrs['units'].split('since')
df = ds[vars_to_keep].squeeze(dim=["x","y"], drop=True).to_dataframe()
start = reference_date.strip().split(" ")[0].replace("-","/")
df['dates'] = pd.date_range(start=start, periods=len(df), freq=freq)
df = df.set_index('dates')


SEC_2_HOUR = 3600.
df['Rainf'] *= SEC_2_HOUR

dfa = df.resample("A").agg("sum")
dfd = df.resample("D").agg("sum")
dfm = df.resample("M").agg("sum")

#df_clim = dfm.groupby(lambda x: x.month).mean()
df_clim = dfd.groupby(lambda x: x.dayofyear).mean()

print(np.mean(dfa.values))

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

#x = np.arange(1, 13)
x = np.arange(1, 367)
y = df_clim.Rainf.values
print(len(x), len(y))
ax1.bar(x, y, color=colours[2])
#ax1.set_ylabel("Rainfall (mm month$^{-1}$)")
ax1.set_ylabel("Rainfall (mm day$^{-1}$)")
plt.show()
