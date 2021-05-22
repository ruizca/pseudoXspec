#!/usr/bin/env python

"""
Created on Fri Jun 19 10:46:32 2015

@author: ruizca
"""
import argparse
import logging
import subprocess
from itertools import count
from pathlib import Path

from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.table import Table
from astropy.units import UnitTypeError
from gdpyc import GasMap
from tqdm.contrib import tzip

logging.basicConfig(level=logging.INFO)


def get_last_source_fit(last_source_file):
    try:
        with last_source_file.open("r") as fp:
            first_source = int(fp.readline())

    except FileNotFoundError:
        first_source = 0

    return first_source


def update_last_source_fit(last_source, last_source_file):
    with last_source_file.open("w") as fp:
        fp.write(str(last_source))


def check_results_folder(results_folder):
    if results_folder.exists():
        raise FileExistsError(f"results_folder '{results_folder}' already exists!")
    else:
        results_folder.mkdir()


def _get_redshift(sample, zcol):
    if not zcol:
        zcol = "DEFAULT_REDSHIFT"
        sample[zcol] = 0.0

    return sample[zcol]


def _set_coords(ra, dec, unit):
    try:
        coords = SkyCoord(ra, dec)
    except UnitTypeError:
        coords = SkyCoord(ra, dec, unit=unit)

    return coords


def _get_nhgal(sample, nhcol, racol, deccol, unit="deg"):
    if not nhcol:
        nhcol = "NHGAL"
        coords = _set_coords(sample[racol], sample[deccol], unit)
        sample[nhcol] = GasMap.nh(coords, nhmap="LAB")

    return sample[nhcol]


def get_sources_data(sample_file, racol, deccol, zcol=None, nhcol=None, first_source=0):
    sample = Table.read(sample_file)
    sample = sample[first_source:]

    obsid = sample["OBS_ID"]
    detid = sample["DETID"]

    z = _get_redshift(sample, zcol)
    nhgal = _get_nhgal(sample, nhcol, racol, deccol)

    return obsid, detid, z, nhgal


def stack_spectra(obsid, detid, spec_folder):
    # Find spectra of interest for this detection
    obs_path = spec_folder.joinpath(obsid)
    spec_files = obs_path.glob(f"{detid}_SRSPEC_*.pha")

    # Create stack file for existing spectra in the observation
    stack_file = Path(f"spec_{detid}.lis")
    with stack_file.open("w") as fp:
        for spec in spec_files:
            fp.write(spec.resolve().as_posix() + "\n")

    return stack_file


def remove_stack_spectra(stack_file):
    try:
        stack_file.unlink()

    except FileNotFoundError:
        logging.warning("No stack file!")


def fit_detection(z, nh, obsid, detid, results_folder, spectra_folder, fixgamma=True):
    task = "./fit_Xspec.py"
    args = [
        "--redshift",
        f"{z:f}",
        "--nh",
        str(nh),
        "--obsid",
        obsid,
        "--detid",
        str(detid),
        "--output_folder",
        results_folder.as_posix(),
        "--spectra_folder",
        spectra_folder.as_posix(),
    ]
    if fixgamma:
        args += ["--fixGamma"]

    logging.debug(" ".join([task] + args))
    subprocess.check_output([task] + args, stderr=subprocess.STDOUT)


def main(args):
    spec_folder = Path(args.spec_folder)
    results_folder = Path(args.results_folder)
    lastsource_file = Path(args.file_lastsource)

    first_source = get_last_source_fit(lastsource_file)
    if first_source == 0:
        check_results_folder(results_folder)

    obsids, detids, redshifts, nhgals = get_sources_data(
        args.sources_table, args.racol, args.deccol, args.zcol, args.nhcol, first_source
    )

    for obsid, detid, z, nh, current_source in tzip(
        obsids, detids, redshifts, nhgals, count(first_source)
    ):
        try:
            fit_detection(z, nh, obsid, detid, results_folder, spec_folder, args.fixgamma)
            update_last_source_fit(current_source + 1, lastsource_file)
        except Exception as e:
            logging.error(e)
            logging.error(f"Something went wrong fitting detection {detid}")


if __name__ == "__main__":
    # Parser for shell parameters
    parser = argparse.ArgumentParser(description="Fitting X-ray pseudospectra")

    parser.add_argument(
        "--catalogue",
        dest="sources_table",
        action="store",
        default=None,
        help="Full route to the detections catalogue.",
    )
    parser.add_argument(
        "--spec_folder",
        dest="spec_folder",
        action="store",
        default="./data/spectra/",
        help="Folder of the pseudospectra.",
    )
    parser.add_argument(
        "--results_folder",
        dest="results_folder",
        action="store",
        default="./fit_results/",
        help="Folder for saving the fit results.",
    )
    parser.add_argument(
        "--racol",
        dest="racol",
        action="store",
        default="XMM_RA",
        help="Name of the RA column in the catalogue.",
    )
    parser.add_argument(
        "--deccol",
        dest="deccol",
        action="store",
        default="XMM_DEC",
        help="Name of the Dec column in the catalogue.",
    )
    parser.add_argument(
        "--zcol",
        dest="zcol",
        action="store",
        default=None,
        help="Name of the redshift column in the catalogue.",
    )
    parser.add_argument(
        "--nhcol",
        dest="nhcol",
        action="store",
        default=None,
        help="Name of the Galactic NH column in the catalogue.",
    )
    parser.add_argument(
        "--lsf",
        dest="file_lastsource",
        action="store",
        default="last_source.dat",
        help="File to store the last fitted source.",
    )
    parser.add_argument(
        "--fixGamma",
        dest="fixgamma",
        action="store_true",
        default=False,
        help="Fit with a fixed photon index (1.9).",
    )

    main(parser.parse_args())
