#!/usr/bin/env python

"""
Created on Fri Jun 19 10:46:32 2015

@author: ruizca
"""
#from __future__ import print_function

import os
import argparse
import json
import logging

import numpy as np
from scipy.stats import binned_statistic

import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.gridspec
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#from astropy.io import fits as pyfits
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u

import sherpa.astro.ui as shp
from sherpa.astro import datastack as dsmod

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('axes.formatter', limits=(-3, 3))
plt.style.use('bmh')

shp.set_conf_opt('sigma', 1.6)
shp.set_conf_opt('numcores', 4)
shp.set_proj_opt('sigma', 1.6)
shp.set_proj_opt('numcores', 4)
#shp.set_preference("window.display", "false")


def properFit():
     while True:
          dsmod.fit()
          fitresult = shp.get_fit_results()

          if fitresult.dstatval <= 1e-1:#2e-2:
             break


def save_goodness(stats, prefix):
    """
    Save goodness of fit stats into a json file.
    """
    dict_goodness = {"rchi2": stats.rstat,
                     "dof": stats.dof,
                     "npar": stats.numpoints - stats.dof,
                     "chi2": stats.statval}

    json_filename = "{}_bestfit_goodness.json".format(prefix)
    with open(json_filename, 'w') as json_file:
        json.dump(dict_goodness, json_file, indent=2)


def save_params(dict_params, prefix):
    """
    Save best-fit parameters into a json file.
    """
    json_filename = "{}_bestfit_params.json".format(prefix)
    with open(json_filename, 'w') as json_file:
        json.dump(dict_params, json_file, indent=2)


def save_fluxes(dict_fluxes, prefix):
    """
    Save best-fit parameters into a json file.
    """
    json_filename = "{}_bestfit_fluxes.json".format(prefix)
    with open(json_filename, 'w') as json_file:
        json.dump(dict_fluxes, json_file, indent=2)


def nH_uplimit():
    # Fix nH to 1e19 cm-2 and find the best fit
    shp.set_par('abs1.nH', val=0.001, frozen=True)
    properFit()
    dsmod.freeze('po1.PhoIndex')

    properFit()
    chi2_min = shp.get_fit_results().statval

    # Estimate the probability distribution of NH using chi2
    NHtest = np.logspace(-3, 3, num=120)
    PNHtest = np.full(len(NHtest), np.nan)
    cumNHtest = np.full(len(NHtest), np.nan)

    for j in range(len(NHtest)):
        shp.set_par('abs1.nH', val=NHtest[j])
        properFit()

        Dchi2 = shp.get_fit_results().statval - chi2_min
        PNHtest[j] = np.exp(-Dchi2/2)
        cumNHtest[j] = np.trapz(PNHtest[:j+1], NHtest[:j+1])

    # Normalize
    Pnorm = 1/np.trapz(PNHtest, NHtest)
    C = Pnorm*cumNHtest
    #P = Pnorm*PNHtest
    nH = 10**np.interp(np.log10(0.955), np.log10(C),
                       np.log10(NHtest), left=0, right=1)

#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.loglog(NHtest, P, linewidth=0, marker='o')
#    ax.loglog(NHtest, C, linewidth=0, marker='o')
#    ax.axvline(nH)
#    plt.show()

    return nH

def ticks_format(value, index):
    """
    get the value and returns the value as:
       integer: [0,99]
       1 digit float: [0.1, 0.99]
       n*10^m: otherwise
    To have all the number of the same size they are all returned as latex strings
    """
    exp = np.floor(np.log10(value))
    base = value/10**exp
    if exp == 0 or exp == 1:
        return '${0:d}$'.format(int(value))
    if exp == -1:
        return '${0:.1f}$'.format(value)
    else:
        if base == 1:
            return '$10^{{{0:d}}}$'.format(int(exp))
        else:
            return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))

def my_plot_fit(ids, plot_folder, emin=0, emax=np.inf,
                save=True, label_model="", z=None):

    all_model = []
    all_emodel = []
    all_data = []
    all_dataxerr = []
    all_datayerr = []
    all_edata = []
    all_ratio = []
    all_ratioerr = []

    # Get data and model for each spectrum
    for sid in ids:
        d = shp.get_data_plot(sid)
        m = shp.get_model_plot(sid)
        e = (m.xhi + m.xlo)/2
        bins = np.concatenate((d.x - d.xerr/2, [d.x[-1] + d.xerr[-1]]))

        model = m.y
        model_de = model * (m.xhi - m.xlo)
        model_binned, foo1, foo2 = binned_statistic(e, model_de, bins=bins,
                                                    statistic='sum')
        model_binned = model_binned/d.xerr

        #delchi = resid/d.yerr
        ratio = d.y/model_binned

        mask_data = np.logical_and(d.x+d.xerr/2 >= emin, d.x-d.xerr/2 <= emax)
        mask_model = np.logical_and(e >= emin, e <= emax)

        all_model.append(model[mask_model])
        all_emodel.append(e[mask_model])
        all_data.append(d.y[mask_data])
        all_dataxerr.append(d.xerr[mask_data])
        all_datayerr.append(d.yerr[mask_data])
        all_edata.append(d.x[mask_data])
        all_ratio.append(ratio[mask_data])
        all_ratioerr.append(d.yerr[mask_data]/model_binned[mask_data])


    # Show all spectra in one plot
    fig = plt.figure()
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    ax0 = plt.subplot(gs[0])
    for i in range(len(ids)):
        ax0.errorbar(all_edata[i], all_data[i],
                     xerr=all_dataxerr[i]/2, yerr=all_datayerr[i],
                     fmt="o", ms=5, elinewidth=1.25, capsize=2,
                     ls="None", zorder=1000)
        ax0.loglog(all_emodel[i], all_model[i], c='red', alpha=0.5)

    ax0.set_ylabel("count rate / $\mathrm{s}^{-1}\:\mathrm{keV}^{-1}$")
    ax0.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(ticks_format))

    ax1 = plt.subplot(gs[1], sharex=ax0)
    for i in range(len(ids)):
        ax1.errorbar(all_edata[i], all_ratio[i],
                     xerr=all_dataxerr[i]/2,
                     yerr=all_ratioerr[i],
                     elinewidth=1.25, capsize=2, ls="None", zorder=1000)

    ax1.axhline(1, ls="--", c="gray")

    plt.setp(ax0.get_xticklabels(), visible=False)

    ax1.set_yscale("log")
    ax1.set_xlabel("Energy / keV")
    ax1.set_ylabel("ratio")

    ax1.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(ticks_format))
    ax1.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 3.0, 5.0)))
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(ticks_format))
    ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    plt.xlim(0.2, 12)
    plt.tight_layout()

    if save:
        plot_file = '{}_srcspec_ALL.png'.format(label_model)
        plot_file = os.path.join(plot_folder, plot_file)

        fig.savefig(plot_file)
    else:
        fig.show()

    plt.close(fig)

def calc_fluxes(ids, z=0, dataScale=None, num=10):
    flux = np.full((len(shp.list_data_ids()), 3), np.nan)
    rf_flux = np.full((len(shp.list_data_ids()), 3), np.nan)
    rf_int_flux = np.full((len(shp.list_data_ids()), 3), np.nan)
    fx = np.full((3, 3), np.nan)

    if dataScale is None:
        # Estimate average flux of all detectors, with no errors
        for i, sid in enumerate(ids):
            # Observed flux at 0.2-12 keV obs. frame (no abs. corr.)
            flux[i, 0] = shp.calc_energy_flux(id=sid, lo=0.2, hi=12.0)

            # Observed flux at 2-10 keV rest frame (only Gal. abs. corr.)
            shp.set_par('absgal.nH', val=0)
            rf_flux[i, 0] = shp.calc_energy_flux(id=sid, lo=2.0/(1+z), hi=10.0/(1+z))

            # Observed flux at 2-10 keV rest frame (abs. corr.)
            shp.set_par('abs1.nH', val=0)
            rf_int_flux[i, 0] = shp.calc_energy_flux(id=sid, lo=2.0/(1+z), hi=10.0/(1+z))
        fx[0, 0] = np.mean(flux[:, 0])
        fx[1, 0] = np.mean(rf_flux[:, 0])
        fx[2, 0] = np.mean(rf_int_flux[:, 0])

    else:
        # Estimate average flux of all detectors sampling num times the
        # distribution of parameters assuming a multigaussian
        int_component = po1
        obs_component = abs1*po1

        for i, sid in enumerate(ids):
            # Observed flux at 0.2-12 keV obs. frame (no abs. corr.)
            F = shp.sample_flux(lo=0.2, hi=12, id=sid, num=num,
                                scales=dataScale, Xrays=True)
            flux[i, :] = F[0]

            # Observed flux at 2-10 keV rest frame (only Gal. abs. corr.)
            F = shp.sample_flux(obs_component, lo=2/(1+z), hi=10/(1+z),
                                id=sid, num=num, scales=dataScale, Xrays=True)
            rf_flux[i, :] = F[1]

            # Observed flux at 2-10 keV rest frame (abs. corr.)
            F = shp.sample_flux(int_component, lo=2/(1+z), hi=10/(1+z),
                                id=sid, num=num, scales=dataScale, Xrays=True)
            rf_int_flux[i, :] = F[1]

        fx[0, 0] = np.average(flux[:, 0], weights=flux[:, 1]**-1)
        fx[1, 0] = np.average(rf_flux[:, 0], weights=rf_flux[:, 1]**-1)
        fx[2, 0] = np.average(rf_int_flux[:, 0], weights=rf_int_flux[:, 1]**-1)

        k = np.sqrt(len(ids))
        fx[0, 1:] = k/np.sum(flux[:, 2]**-1), k/np.sum(flux[:, 1]**-1)
        fx[1, 1:] = k/np.sum(rf_flux[:, 2]**-1), k/np.sum(rf_flux[:, 1]**-1)
        fx[2, 1:] = k/np.sum(rf_int_flux[:, 2]**-1), k/np.sum(rf_int_flux[:, 1]**-1)

    return fx


def main(args):
    logger = logging.getLogger("sherpa")
    logger.setLevel(logging.ERROR)

    ## Define prefix and results folder
    results_folder = os.path.join(args.folder, args.obsid)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    prefix = '{}/{}'.format(results_folder, args.detid)

    ## Load data stack and ignore bad data
    dsmod.clean()
    ds = dsmod.DataStack()
    ds.load_pha('@spec_{}.lis'.format(args.detid), use_errors=True)
    #dsmod.ignore_bad()
    dsmod.ignore(":0.2,12.0:")

    ## Dictionaries for storing best-fit parameters and fluxes
    params = {"nHgal": args.galnH,
              "nH": None, "nH_ErrMin": None, "nH_ErrMax": None,
              "PhoIndex": 1.9, "PhoIndex_ErrMin": None, "PhoIndex_ErrMax": None}
    fluxes = {"fx": None, "fx_ErrMin": None, "fx_ErrMax": None,
              "fx_obs": None, "fx_obs_ErrMin": None, "fx_obs_ErrMax": None,
              "fx_int": None, "fx_int_ErrMin": None, "fx_int_ErrMax": None,
              "Lx": None, "Lx_ErrMin": None, "Lx_ErrMax": None}
    z = args.z

    ## Set model
    ds.set_source('xsphabs.absgal * xszphabs.abs1 * xszpowerlw.po1')

    shp.set_par('absgal.nH', val=params['nHgal']/1e22, frozen=True)
    shp.set_par('abs1.redshift', val=z, frozen=True)
    shp.set_par('po1.redshift', val=z, frozen=True)

    if len(shp.list_data_ids()) == 1 or args.fixgamma:
        shp.set_par('po1.PhoIndex', val=params['PhoIndex'], frozen=True)

    properFit()
    fitstats = shp.get_fit_results()
    nH = fitstats.parvals[0]

    # Calc errors for parameters and fluxes if we found a reasonable fit,
    # otherwise just store a flux estimate and best fit parameters with
    # no errors (nan)
    if fitstats.rstat <= 3.0:
        dsmod.conf(abs1.nH)
        confstats = shp.get_conf_results()
        nHmin = confstats.parmins[0]
        nHmax = confstats.parmaxes[0]

        # Estimate nH upper-limit if nH=0, or nH-nHmin<0, or nHmin = nan
        if (nH == 0) or (nHmin is None) or (nH - nHmin <= 0):
            params['nH'] = nH_uplimit()

            # Restore original fit (NH=0)
            if len(shp.list_data_ids()) > 1:
                dsmod.thaw('po1.PhoIndex')
            shp.set_par('abs1.nH', val=0)
            properFit()

        else:
            params['nH'] = nH
            params['nH_ErrMin'] = nHmin
            params['nH_ErrMax'] = nHmax

        # Get photon index errors
        if len(shp.list_data_ids()) > 1 and not args.fixgamma:
            dsmod.conf(po1.PhoIndex)
            confstats = shp.get_conf_results()
            params['PhoIndex'] = confstats.parvals[0]
            params['PhoIndex_ErrMin'] = confstats.parmins[0]
            params['PhoIndex_ErrMax'] = confstats.parmaxes[0]

        fitstats = shp.get_fit_results()
        save_goodness(fitstats, prefix)
        my_plot_fit(shp.list_data_ids(), results_folder, emin=0.2, emax=12.0,
                    save=True, label_model=args.detid, z=args.z)

        # Estimate fluxes and flux errors
        shp.covar()
        dataScale = shp.get_covar_results().parmaxes
        if all(d is None for d in dataScale):
            fx = calc_fluxes(shp.list_data_ids(), z=z)
            fluxes['fx'], fluxes['fx_obs'], fluxes['fx_int'] = fx[:, 0]

        else:        
            fx = calc_fluxes(shp.list_data_ids(), z=z, dataScale=dataScale)            

            fluxes['fx'], fluxes['fx_ErrMin'], fluxes['fx_ErrMax'] = fx[0, :]

            fluxes['fx_obs'], fluxes['fx_obs_ErrMin'], fluxes['fx_obs_ErrMax'] = fx[1, :]

            fluxes['fx_int'], fluxes['fx_int_ErrMin'], fluxes['fx_int_ErrMax'] = fx[2, :]

    else:
        fitstats = shp.get_fit_results()
        save_goodness(fitstats, prefix)
        my_plot_fit(shp.list_data_ids(), results_folder, emin=0.2, emax=12.0,
                    save=True, label_model=args.detid, z=args.z)

        # Estimate fluxes
        fx = calc_fluxes(shp.list_data_ids(), z=z)
        fluxes['fx'], fluxes['fx_obs'], fluxes['fx_int'] = fx[:, 0]

    # Luminosity
    Dl = cosmo.luminosity_distance(z)
    Dl = Dl.to(u.cm)
    K = 4*np.pi*Dl.value**2

    fluxes['Lx'] = K*fluxes['fx_int']
    if fluxes['fx_int_ErrMin'] is not None:
        fluxes['Lx_ErrMin'] = K*fluxes['fx_int_ErrMin']
        fluxes['Lx_ErrMax'] = K*fluxes['fx_int_ErrMax']

    save_params(params, prefix)
    save_fluxes(fluxes, prefix)


if __name__ == '__main__':
    # Parser for shell parameters
    parser = argparse.ArgumentParser(description='Spectral fitting of X-ray pseudospectra')

    parser.add_argument('-obsid', dest='obsid', action='store',
                        default=None, help='OBSID of the source')

    parser.add_argument('-detid', dest='detid', action='store',
                        default=None, help='Detection ID of the source')

    parser.add_argument('-z', dest='z', action='store', type=float,
                        default=None, help='Redshift of the source')

    parser.add_argument('-nh', dest='galnH', action='store', type=float,
                        default=0.01, help='Galactic nH')

    parser.add_argument('-folder', dest='folder', action='store',
                        default='fit_results', help='Folder for saving fits results')

    parser.add_argument('--fixGamma', dest='fixgamma',
                        action='store_true', default=False,
                        help='Fit with a fixed photon index (1.9).')

    main(parser.parse_args())
