#!/usr/bin/env python

"""
Run CABLE either for a single site, a subset, or all the flux sites pointed to
in the met directory

- Only intended for biophysics
- Set mpi = True if doing a number of flux sites

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (02.08.2018)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import shutil
import subprocess
import multiprocessing as mp
import numpy as np
import pandas as pd

from cable_utils import adjust_nml_file
from cable_utils import get_svn_info
from cable_utils import change_LAI
from cable_utils import add_attributes_to_output_file
from cable_utils import change_traits

class RunCable(object):

    def __init__(self, met_dir=None, log_dir=None, output_dir=None,
                 restart_dir=None, aux_dir=None, namelist_dir=None,
                 nml_fname="cable.nml",
                 grid_fname="gridinfo_CSIRO_1x1.nc",
                 phen_fname="modis_phenology_csiro.txt",
                 cnpbiome_fname="pftlookup_csiro_v16_17tiles.csv",
                 elev_fname="GSWP3_gwmodel_parameters.nc", fwsoil="standard",
                 lai_dir=None, fixed_lai=None, co2_conc=400.0,
                 met_subset=[], cable_src=None, cable_exe="cable", mpi=True,
                 num_cores=None, verbose=True, traits=None):

        self.met_dir = met_dir
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.restart_dir = restart_dir
        self.aux_dir = aux_dir
        self.namelist_dir = namelist_dir
        self.nml_fname = nml_fname
        self.biogeophys_dir = os.path.join(self.aux_dir, "core/biogeophys")
        self.grid_dir = os.path.join(self.aux_dir, "offline")
        self.biogeochem_dir = os.path.join(self.aux_dir, "core/biogeochem/")
        self.grid_fname = os.path.join(self.grid_dir, grid_fname)
        self.phen_fname = os.path.join(self.biogeochem_dir, phen_fname)
        self.cnpbiome_fname = os.path.join(self.biogeochem_dir, cnpbiome_fname)
        self.elev_fname = elev_fname
        self.co2_conc = co2_conc
        self.met_subset = met_subset
        self.cable_src = cable_src
        self.cable_exe = os.path.join(cable_src, "offline/%s" % (cable_exe))
        self.verbose = verbose
        self.mpi = mpi
        self.num_cores = num_cores
        self.lai_dir = lai_dir
        self.fixed_lai = fixed_lai
        self.fwsoil = fwsoil
        self.traits = traits


    def main(self):

        (met_files, url, rev) = self.initialise_stuff()

        # Setup multi-processor jobs
        if self.mpi:
            if self.num_cores is None: # use them all!
                self.num_cores = mp.cpu_count()
            chunk_size = int(np.ceil(len(met_files) / float(self.num_cores)))
            pool = mp.Pool(processes=self.num_cores)
            processes = []

            for i in range(self.num_cores):
                start = chunk_size * i
                end = chunk_size * (i + 1)
                if end > len(met_files):
                    end = len(met_files)

                # setup a list of processes that we want to run
                p = mp.Process(target=self.worker,
                               args=(met_files[start:end], url, rev, ))
                processes.append(p)

            # Run processes
            for p in processes:
                p.start()
        else:
            self.worker(met_files, url, rev)

    def worker(self, met_files, url, rev):

        for fname in met_files:

            species = '_'.join((os.path.basename(fname).split("_")[2:4]))
            num = os.path.basename(fname).split("_")[-1].split(".")[0]
            site = "%s_%s" % (species, num)

            base_nml_fn = os.path.join(self.grid_dir, "%s" % (self.nml_fname))
            nml_fname = "cable_%s.nml" % (site)
            shutil.copy(base_nml_fn, nml_fname)

            (out_fname, out_log_fname) = self.clean_up_old_files(site)

            spp_traits = self.traits[self.traits.species == \
                                     species.replace("_", " ")]

            if len(spp_traits) > 0:
                b_plant = float(spp_traits.b_plant.values[0])
                c_plant = float(spp_traits.c_plant.values[0])
                vcmax = float(spp_traits.vcmax.values[0])
                fname_new = change_traits(fname, id, site, b_plant,
                                          c_plant, vcmax)

            replace_dict = {
                            "filename%met": "'%s'" % (fname_new),
                            "filename%out": "'%s'" % (out_fname),
                            "filename%log": "'%s'" % (out_log_fname),
                            "filename%restart_out": "' '",
                            "filename%type": "'%s'" % (self.grid_fname),
                            "output%restart": ".FALSE.",
                            "fixedCO2": "%.2f" % (self.co2_conc),
                            "casafile%phen": "'%s'" % (self.phen_fname),
                            "casafile%cnpbiome": "'%s'" % (self.cnpbiome_fname),
                            "cable_user%GS_SWITCH": "'medlyn'",
                            "cable_user%GW_MODEL": ".FALSE.",
                            "cable_user%or_evap": ".FALSE.",
                            "redistrb": ".FALSE.",
                            "spinup": ".FALSE.",
                            "cable_user%litter": ".TRUE.",
                            "cable_user%FWSOIL_SWITCH": "'%s'" % (self.fwsoil),
            }
            adjust_nml_file(nml_fname, replace_dict)
            self.run_me(nml_fname)

            add_attributes_to_output_file(nml_fname, out_fname, url, rev)
            shutil.move(nml_fname, os.path.join(self.namelist_dir, nml_fname))

            os.remove(fname_new)
            os.remove("new_sumbal")

    def initialise_stuff(self):

        if not os.path.exists(self.restart_dir):
            os.makedirs(self.restart_dir)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not os.path.exists(self.namelist_dir):
            os.makedirs(self.namelist_dir)

        # Run all the met files in the directory
        if len(self.met_subset) == 0:
            met_files = glob.glob(os.path.join(self.met_dir, "*.nc"))
        else:
            met_files = [os.path.join(self.met_dir, i) for i in self.met_subset]

        cwd = os.getcwd()
        (url, rev) = get_svn_info(cwd, self.cable_src)

        # delete local executable, copy a local copy and use that
        local_exe = "cable"
        if os.path.isfile(local_exe):
            os.remove(local_exe)
        shutil.copy(self.cable_exe, local_exe)
        self.cable_exe = local_exe

        return (met_files, url, rev)

    def clean_up_old_files(self, site):
        out_fname = os.path.join(self.output_dir, "%s_out.nc" % (site))
        if os.path.isfile(out_fname):
            os.remove(out_fname)

        out_log_fname = os.path.join(self.log_dir, "%s_log.txt" % (site))
        if os.path.isfile(out_log_fname):
            os.remove(out_log_fname)

        return (out_fname, out_log_fname)

    def run_me(self, nml_fname):
        # run the model
        if self.verbose:
            cmd = './%s %s' % (self.cable_exe, nml_fname)
            error = subprocess.call(cmd, shell=True)
            if error == 1:
                print("Job failed to submit")
                raise
        else:
            # No outputs to the screen: stout and stderr to dev/null
            cmd = './%s %s > /dev/null 2>&1' % (self.cable_exe, nml_fname)
            error = subprocess.call(cmd, shell=True)
            if error == 1:
                print("Job failed to submit")

if __name__ == "__main__":

    met_dir = "/Users/mdekauwe/research/narclim_data/1990-2009/CCCMA3.1/R1"
    #met_dir = os.getcwd()
    log_dir = "logs"
    output_dir = "debug"
    restart_dir = "restart_files"
    namelist_dir = "namelists"

    #cable_src = "../../src/trunk_DESICA_PFTs/trunk_DESICA_PFTs/"
    cable_src = "../../src/profit_max/profit_max/"
    aux_dir = "../../src/CABLE-AUX/"
    mpi = False
    num_cores = 4 # set to a number, if None it will use all cores...!
    # if empty...run all the files in the met_dir
    met_subset = ['narclim_met_Eucalyptus_viminalis_64.nc']

    # ------------------------------------------- #
    """
    C = RunCable(met_dir=met_dir, log_dir=log_dir, output_dir=output_dir,
                 restart_dir=restart_dir, aux_dir=aux_dir,
                 namelist_dir=namelist_dir, met_subset=met_subset,
                 cable_src=cable_src, mpi=mpi, num_cores=num_cores,
                 fwsoil="standard")
    C.main()
    site = met_subset[0].split(".")[0]
    out_fname = os.path.join(output_dir, "%s_out.nc" % (site))
    shutil.move(out_fname, os.path.join(output_dir, "standard.nc"))

    """
    df_traits = pd.read_csv("euc_species_traits.csv")


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    C = RunCable(met_dir=met_dir, log_dir=log_dir, output_dir=output_dir,
                 restart_dir=restart_dir, aux_dir=aux_dir,
                 namelist_dir=namelist_dir, met_subset=met_subset,
                 cable_src=cable_src, mpi=mpi, num_cores=num_cores,
                 fwsoil="profitmax", traits=df_traits)
    C.main()

    #site = met_subset[0].split(".")[0]
    #out_fname = os.path.join(output_dir, "%s_out.nc" % (site))
    #shutil.move(out_fname, os.path.join(output_dir, "hydraulics_profitmax.nc"))
    #"""
