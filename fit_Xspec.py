#!/usr/bin/env python
"""
Created on Fri Jun 19 10:46:32 2015

@author: ruizca
"""
import argparse
import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import sherpa.astro.ui as shp
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from sherpa.astro import datastack as dsmod

import plots


def set_sherpa_env(stat=None):
    shp.clean()
    dsmod.clean()

    shp.set_conf_opt("sigma", 1.6)
    shp.set_conf_opt("numcores", 4)
    shp.set_proj_opt("sigma", 1.6)
    shp.set_proj_opt("numcores", 4)

    if stat:
        shp.set_stat(stat)


def parseargs(args):
    return (
        args.obsid,
        args.detid,
        args.nhgal / 1e22,
        args.z,
        Path(args.output_folder),
        Path(args.spectra_folder),
    )


def set_results_path(output_path, obsid, detid):
    results_path = output_path.joinpath(obsid)

    if not results_path.exists():
        results_path.mkdir()

    prefix = results_path.joinpath(detid).as_posix()

    return results_path, prefix


def _write_spec_files(fp, obsid, detid, spectra_path):
    obs_path = spectra_path.joinpath(obsid)

    for spec in obs_path.glob(f"{detid}_SRSPEC_*.pha"):
        fp.write(spec.resolve().as_posix() + "\n")

    fp.seek(0)

    return Path(fp.name).name


def load_stack_data(obsid, detid, spectra_path):
    ds = dsmod.DataStack()

    # Create stack file for existing spectra in the observation
    with NamedTemporaryFile("w", dir=".") as temp:
        stack_tempfile = _write_spec_files(temp, obsid, detid, spectra_path)
        ds.load_pha(f"@{stack_tempfile}", use_errors=True)

    ids = shp.list_data_ids()
    for id in ids:
        dsmod.ignore_bad(id=id)

    return ds, ids


def set_fixgamma(ids, fixgamma, default_value=1.95):
    if default_value == 0:
        raise ValueError("Photon Index with value 0 is not allowed!")

    if len(ids) == 1 or fixgamma:
        return default_value
    else:
        return False


def set_default_params_dict(nhgal):
    return {
        "nHgal": nhgal,
        "nH": None,
        "nH_ErrMin": None,
        "nH_ErrMax": None,
        "PhoIndex": None,
        "PhoIndex_ErrMin": None,
        "PhoIndex_ErrMax": None,
    }


def set_default_fluxes_dict():
    return {
        "fx": None,
        "fx_ErrMin": None,
        "fx_ErrMax": None,
        "fx_obs": None,
        "fx_obs_ErrMin": None,
        "fx_obs_ErrMax": None,
        "fx_int": None,
        "fx_int_ErrMin": None,
        "fx_int_ErrMax": None,
        "Lx": None,
        "Lx_ErrMin": None,
        "Lx_ErrMax": None,
    }


def set_model_powerlaw(ds, nhgal, z, fixgamma):
    ds.set_source("xsphabs.absgal * xszphabs.abs1 * xszpowerlw.po1")

    shp.set_par("absgal.nH", val=nhgal, frozen=True)
    shp.set_par("abs1.redshift", val=z, frozen=True)
    shp.set_par("po1.redshift", val=z, frozen=True)

    if fixgamma:
        shp.set_par("po1.PhoIndex", val=fixgamma, frozen=True)


def properFit(tolerance=1e-1):
    while True:
        dsmod.fit()
        fitresult = shp.get_fit_results()

        if fitresult.dstatval <= tolerance:  # 2e-2:
            break

    return fitresult


def _nH_uplimit(fixgamma, sigma_level=0.955):
    # Fix nH to 1e19 cm-2 and find the best fit
    shp.set_par("abs1.nH", val=0.001, frozen=True)
    properFit()
    dsmod.freeze("po1.PhoIndex")

    fitresult = properFit()
    chi2_min = fitresult.statval

    # Estimate the probability distribution of NH using chi2
    NHtest = np.logspace(-3, 3, num=120)
    PNHtest = np.full(len(NHtest), np.nan)
    cumNHtest = np.full(len(NHtest), np.nan)

    for i, nh in enumerate(NHtest):
        shp.set_par("abs1.nH", val=nh)
        fitresult = properFit()

        Dchi2 = fitresult.statval - chi2_min
        PNHtest[i] = np.exp(-Dchi2 / 2)
        cumNHtest[i] = np.trapz(PNHtest[: i + 1], NHtest[: i + 1])

    # Normalize
    Pnorm = 1 / np.trapz(PNHtest, NHtest)
    C = Pnorm * cumNHtest
    nH = 10 ** np.interp(
        np.log10(sigma_level), np.log10(C), np.log10(NHtest), left=0, right=1
    )

    #    fig = plt.figure()
    #    ax = fig.add_subplot(111)
    #    ax.loglog(NHtest, P, linewidth=0, marker='o')
    #    ax.loglog(NHtest, C, linewidth=0, marker='o')
    #    ax.axvline(nH)
    #    plt.show()

    # Restore original fit (NH=0)
    if not fixgamma:
        dsmod.thaw("po1.PhoIndex")

    shp.set_par("abs1.nH", val=0)
    properFit()

    return nH


def get_nh_param(params, fitstats, fixgamma):
    nH = fitstats.parvals[0]

    if fitstats.rstat <= 3.0:
        dsmod.conf(abs1.nH)
        confstats = shp.get_conf_results()
        nHmin = confstats.parmins[0]
        nHmax = confstats.parmaxes[0]

        # Estimate nH upper-limit if nH=0, or nH-nHmin<0, or nHmin = nan
        if (nH == 0) or (nHmin is None) or (nH - nHmin <= 0):
            params["nH"] = _nH_uplimit(fixgamma)
        else:
            params["nH"] = nH
            params["nH_ErrMin"] = nHmin
            params["nH_ErrMax"] = nHmax
    else:
        params["nH"] = nH

    return params


def get_gamma_param(params, fitstats, fixgamma):
    if fixgamma:
        params["PhoIndex"] = fixgamma
    else:
        if fitstats.rstat <= 3:
            dsmod.conf(po1.PhoIndex)
            confstats = shp.get_conf_results()
            params["PhoIndex"] = confstats.parvals[0]
            params["PhoIndex_ErrMin"] = confstats.parmins[0]
            params["PhoIndex_ErrMax"] = confstats.parmaxes[0]
        else:
            params["PhoIndex"] = fitstats.parvals[1]

    return params


def _calc_fluxes(ids, z=0, dataScale=None, nsims=10):
    flux = np.full((len(ids), 3), np.nan)
    rf_flux = np.full((len(ids), 3), np.nan)
    rf_int_flux = np.full((len(ids), 3), np.nan)
    fx = np.full((3, 3), np.nan)

    if dataScale is None:
        # Estimate average flux of all detectors, with no errors
        for i, sid in enumerate(ids):
            # Observed flux at 0.2-12 keV obs. frame (no abs. corr.)
            flux[i, 0] = shp.calc_energy_flux(id=sid, lo=0.2, hi=12.0)

            # Observed flux at 2-10 keV rest frame (only Gal. abs. corr.)
            shp.set_par("absgal.nH", val=0)
            rf_flux[i, 0] = shp.calc_energy_flux(
                id=sid, lo=2.0 / (1 + z), hi=10.0 / (1 + z)
            )

            # Observed flux at 2-10 keV rest frame (abs. corr.)
            shp.set_par("abs1.nH", val=0)
            rf_int_flux[i, 0] = shp.calc_energy_flux(
                id=sid, lo=2.0 / (1 + z), hi=10.0 / (1 + z)
            )
        fx[0, 0] = np.mean(flux[:, 0])
        fx[1, 0] = np.mean(rf_flux[:, 0])
        fx[2, 0] = np.mean(rf_int_flux[:, 0])

    else:
        # Estimate average flux of all detectors sampling num times the
        # distribution of parameters assuming a multigaussian
        int_component = po1
        obs_component = abs1 * po1

        for i, sid in enumerate(ids):
            # Observed flux at 0.2-12 keV obs. frame (no abs. corr.)
            F = shp.sample_flux(
                lo=0.2, hi=12, id=sid, num=nsims, scales=dataScale, Xrays=True
            )
            flux[i, :] = F[0]

            # Observed flux at 2-10 keV rest frame (only Gal. abs. corr.)
            F = shp.sample_flux(
                obs_component,
                lo=2 / (1 + z),
                hi=10 / (1 + z),
                id=sid,
                num=nsims,
                scales=dataScale,
                Xrays=True,
            )
            rf_flux[i, :] = F[1]

            # Observed flux at 2-10 keV rest frame (abs. corr.)
            F = shp.sample_flux(
                int_component,
                lo=2 / (1 + z),
                hi=10 / (1 + z),
                id=sid,
                num=nsims,
                scales=dataScale,
                Xrays=True,
            )
            rf_int_flux[i, :] = F[1]

        fx[0, 0] = np.average(flux[:, 0], weights=flux[:, 1] ** -1)
        fx[1, 0] = np.average(rf_flux[:, 0], weights=rf_flux[:, 1] ** -1)
        fx[2, 0] = np.average(rf_int_flux[:, 0], weights=rf_int_flux[:, 1] ** -1)

        k = np.sqrt(len(ids))
        fx[0, 1:] = k / np.sum(flux[:, 2] ** -1), k / np.sum(flux[:, 1] ** -1)
        fx[1, 1:] = k / np.sum(rf_flux[:, 2] ** -1), k / np.sum(rf_flux[:, 1] ** -1)
        fx[2, 1:] = (
            k / np.sum(rf_int_flux[:, 2] ** -1),
            k / np.sum(rf_int_flux[:, 1] ** -1),
        )

    return fx


def get_fluxes(ids, z, fluxes, fitstats, nsims=10):
    shp.covar()
    dataScale = shp.get_covar_results().parmaxes

    if fitstats.rstat <= 3.0:
        if all(d is None for d in dataScale):
            fx = _calc_fluxes(ids, z=z, nsims=nsims)
            fluxes["fx"], fluxes["fx_obs"], fluxes["fx_int"] = fx[:, 0]

        else:
            fx = _calc_fluxes(ids, z=z, dataScale=dataScale, nsims=nsims)
            fluxes["fx"], fluxes["fx_ErrMin"], fluxes["fx_ErrMax"] = fx[0, :]
            fluxes["fx_obs"], fluxes["fx_obs_ErrMin"], fluxes["fx_obs_ErrMax"] = fx[1, :]
            fluxes["fx_int"], fluxes["fx_int_ErrMin"], fluxes["fx_int_ErrMax"] = fx[2, :]
    else:
        fx = _calc_fluxes(ids, z=z, nsims=nsims)
        fluxes["fx"], fluxes["fx_obs"], fluxes["fx_int"] = fx[:, 0]

    return fluxes


def get_luminosities(z, fluxes):
    Dl = cosmo.luminosity_distance(z)
    Dl = Dl.to(u.cm)
    K = 4 * np.pi * Dl.value ** 2

    fluxes["Lx"] = K * fluxes["fx_int"]
    if fluxes["fx_int_ErrMin"] is not None:
        fluxes["Lx_ErrMin"] = K * fluxes["fx_int_ErrMin"]
        fluxes["Lx_ErrMax"] = K * fluxes["fx_int_ErrMax"]

    return fluxes


def save_goodness(prefix):
    """
    Save goodness of fit stats into a json file.
    """
    stats = shp.get_fit_results()

    dict_goodness = {
        "rchi2": stats.rstat,
        "dof": stats.dof,
        "npar": stats.numpoints - stats.dof,
        "chi2": stats.statval,
    }

    json_filename = f"{prefix}_bestfit_goodness.json"
    with open(json_filename, "w") as json_file:
        json.dump(dict_goodness, json_file, indent=2)


def save_params(dict_params, prefix):
    """
    Save best-fit parameters into a json file.
    """
    json_filename = f"{prefix}_bestfit_params.json"
    with open(json_filename, "w") as json_file:
        json.dump(dict_params, json_file, indent=2)


def save_fluxes(dict_fluxes, prefix):
    """
    Save best-fit parameters into a json file.
    """
    json_filename = f"{prefix}_bestfit_fluxes.json"
    with open(json_filename, "w") as json_file:
        json.dump(dict_fluxes, json_file, indent=2)


def main(args):
    logger = logging.getLogger("sherpa")
    logger.setLevel(logging.ERROR)

    set_sherpa_env()

    obsid, detid, nhgal, z, output_path, spectra_path = parseargs(args)
    results_path, prefix = set_results_path(output_path, obsid, detid)

    ds, ids = load_stack_data(obsid, detid, spectra_path)
    fixgamma = set_fixgamma(ids, args.fixgamma, default_value=1.95)
    params = set_default_params_dict(nhgal)
    fluxes = set_default_fluxes_dict()

    set_model_powerlaw(ds, nhgal, z, fixgamma)
    fitstats = properFit()

    params = get_nh_param(params, fitstats, fixgamma)
    params = get_gamma_param(params, fitstats, fixgamma)
    save_params(params, prefix)
    save_goodness(prefix)

    plots.spectra_bestfit(ids, emin=0.2, emax=12.0, prefix=prefix)

    fluxes = get_fluxes(ids, z, fluxes, fitstats, nsims=10)
    fluxes = get_luminosities(z, fluxes)
    save_fluxes(fluxes, prefix)


if __name__ == "__main__":
    # Parser for shell parameters
    parser = argparse.ArgumentParser(
        description="Spectral fitting of X-ray pseudospectra"
    )

    parser.add_argument(
        "--obsid", dest="obsid", action="store", default=None, help="OBSID of the source"
    )
    parser.add_argument(
        "--detid",
        dest="detid",
        action="store",
        default=None,
        help="Detection ID of the source",
    )
    parser.add_argument(
        "--redshift",
        dest="z",
        action="store",
        type=float,
        default=None,
        help="Redshift of the source",
    )
    parser.add_argument(
        "--nh",
        dest="nhgal",
        action="store",
        type=float,
        default=3.0e20,
        help="Galactic nH",
    )
    parser.add_argument(
        "--spectra_folder",
        dest="spectra_folder",
        action="store",
        default="data/spectra",
        help="Folder containing the spectra",
    )
    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        action="store",
        default="fit_results",
        help="Folder for saving fits results",
    )
    parser.add_argument(
        "--fixGamma",
        dest="fixgamma",
        action="store_true",
        default=False,
        help="Fit with a fixed photon index (1.9)",
    )

    main(parser.parse_args())
