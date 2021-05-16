#!/usr/bin/env python
"""
Created on Fri Jun 12 12:40:55 2015

@author: ruizca
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from tqdm.contrib import tzip

from enums import Detector, Ebands, Submodes

#### Define limits of observation epochs for each camera
# https://www.cosmos.esa.int/web/xmm-newton/epic-response-files
EPOCH_LIMITS = {
    "pn": ["2007-01-01", "2014-01-01"],
    "mos": [
        "2001-01-01",
        "2001-10-01",
        "2002-01-01",
        "2002-10-01",
        "2003-01-01",
        "2004-10-01",
        "2005-10-01",
        "2006-10-01",
        "2007-10-01",
        "2008-10-01",
        "2009-10-01",
        "2010-10-01",
        "2011-10-01",
    ],
}


def set_output_path(dest_folder):
    spec_path = Path(dest_folder, "spectra")
    if not spec_path.exists():
        spec_path.mkdir()

    return spec_path


def set_detector(detector):
    try:
        return Detector[detector]
    except KeyError:
        raise ValueError(f"Unknown detector: {detector}")


def get_epoch_limits(detector):
    return Time(EPOCH_LIMITS[detector.type], format="iso")


def _load_table(sample_file):
    try:
        return Table.read(sample_file, memmap=True)

    except FileNotFoundError:
        raise ValueError(f"Error: table file not found: {sample_file}!")


def _load_column(sample_table, colname, detector=None):
    if detector:
        colname = f"{detector.name}_{colname}"

    return sample_table[colname]


def _get_obs_date(obs_date_column):
    return Time(obs_date_column, format="mjd")


def _get_obs_mode(obs_mode_column, detector):
    obs_mode = np.zeros(len(obs_mode_column), dtype="|U3")

    if detector.type == "pn":
        for submode in Submodes:
            mask = np.char.find(obs_mode_column.data, submode.name.encode("UTF-8")) >= 0
            obs_mode[mask] = submode.value

    return obs_mode


def _fix_count_rate_upper_limits(count_rates, count_rates_err):
    # Identify count rates consistent with negative values
    # and treat them as upper limits
    mask = count_rates - count_rates_err < 0
    upl_vals = (count_rates[mask] + count_rates_err[mask]) / 2

    count_rates[mask] = upl_vals
    count_rates_err[mask] = upl_vals

    return count_rates, count_rates_err


def _get_count_rates(sample_table, detector):
    count_rates = np.zeros((len(sample_table), len(Ebands)))
    count_rates_err = np.zeros((len(sample_table), len(Ebands)))

    for j, eband in enumerate(Ebands):
        count_rates[:, j] = sample_table[f"{detector.name}_{eband.tag}_RATE"]
        count_rates_err[:, j] = sample_table[f"{detector.name}_{eband.tag}_RATE_ERR"]

    count_rates, count_rates_err = _fix_count_rate_upper_limits(
        count_rates, count_rates_err
    )

    return count_rates, count_rates_err


def load_data(sample_file, detector):
    sample_table = _load_table(sample_file)

    detid = _load_column(sample_table, "DETID")
    obsid = _load_column(sample_table, "OBS_ID")
    obs_date = _get_obs_date(_load_column(sample_table, "MJD_START"))

    det_obs_mode = _get_obs_mode(
        _load_column(sample_table, "SUBMODE", detector), detector
    )
    det_filter = _load_column(sample_table, "FILTER", detector)
    det_rate, det_rate_err = _get_count_rates(sample_table, detector)

    return detid, obsid, obs_date, det_obs_mode, det_rate, det_rate_err, det_filter


def _get_epoch(obsdate, limits):
    """
    Returns the epoch of the observation with date obsdate 
    given the epochs dates defined in limits.
    """
    epoch = f"e{len(limits) + 1}"
    for i, d in enumerate(obsdate < limits):
        if d:
            epoch = f"e{i + 1}"
            break

    return epoch


def set_rmf_file(
    obsdate,
    obsmode,
    detector,
    limits,
    calib_folder="calib/rmfcanned/",
    rmfversion="19.0",
):
    epoch = _get_epoch(obsdate, limits)
    if detector.type == "pn":
        filename = f"epn_{epoch}_{obsmode}20_sdY9_v{rmfversion}.rmf"
    else:
        filename = f"{detector.name.lower()}_{epoch}_im_pall_o.rmf"

    return Path(calib_folder, filename)


def set_arf_file(detector, filter):
    return Path("calib", "ARF", f"{detector.long[1:]}_{filter}.arf")


def _read_rmf(rmf_file):
    rmf = Table.read(rmf_file, hdu=2)

    return rmf["CHANNEL"].data, rmf["E_MIN"].data


def _get_bad_channels(energies):
    return np.logical_or(energies < Ebands.E1.min, energies >= Ebands.E5.max)


def _get_spec_quality(bad_channels):
    spec_quality = np.zeros(len(bad_channels), dtype="int")
    spec_quality[bad_channels] = 1

    return spec_quality


def _get_spec_grouping(energies, bad_channels):
    spec_grouping = -1 * np.ones(len(energies), dtype="int")
    spec_grouping[bad_channels] = 1

    for eband in Ebands:
        mask = np.where(energies >= eband.min)[0]
        spec_grouping[mask[0]] = 1

    return spec_grouping


def set_default_spectrum(default_rmf_file):
    channels, energies = _read_rmf(default_rmf_file)
    bad_channels = _get_bad_channels(energies)

    quality = _get_spec_quality(bad_channels)
    grouping = _get_spec_grouping(energies, bad_channels)

    return channels, energies, quality, grouping


def not_detected(count_rate):
    # Undetected sources have NaN count rates in all bands
    return np.all(np.isnan(count_rate))


def set_spec_count_rates(energies, det_rate, det_rate_err):
    spec_rate = np.zeros(len(energies))
    spec_rate_err = np.zeros(len(energies))

    for i, eband in enumerate(Ebands):
        idx_chband = np.nonzero(
            np.logical_and(energies >= eband.min, energies < eband.max)
        )
        idx_rate = np.random.choice(idx_chband[0])
        spec_rate[idx_rate] = det_rate[i]
        spec_rate_err[idx_rate] = det_rate_err[i]

    return spec_rate, spec_rate_err


def set_spec_fits(channels, rate, rate_err, quality, grouping):
    cols = fits.ColDefs(
        [
            fits.Column(name="CHANNEL", format="I", array=channels),
            fits.Column(name="RATE", format="E", unit="counts/s", array=rate),
            fits.Column(name="STAT_ERR", format="E", array=rate_err),
            fits.Column(name="QUALITY", format="I", array=quality),
            fits.Column(name="GROUPING", format="I", array=grouping),
        ]
    )
    return fits.BinTableHDU.from_columns(cols)


def update_spec_fits_header(spec, detector, rsp_file, arf_file, nchannels):
    spec.header.set("EXTNAME", "SPECTRUM")
    spec.header.set("TELESCOP", "XMM")
    spec.header.set("INSTRUME", detector.long)
    spec.header.set("FILTER", "NONE")
    spec.header.set("EXPOSURE", 1.0)
    spec.header.set("BACKFILE", "NONE")
    spec.header.set("CORRFILE", "NONE")
    spec.header.set("CORRSCAL", 1.0)
    spec.header.set("RESPFILE", rsp_file.resolve().as_posix())
    spec.header.set("ANCRFILE", arf_file.resolve().as_posix())
    spec.header.set("HDUCLASS", "OGIP")
    spec.header.set("HDUCLAS1", "SPECTRUM")
    spec.header.set("HDUVERS", "1.3.0")
    spec.header.set("POISSERR", False)
    spec.header.set("CHANTYPE", "PI")
    spec.header.set("DETCHANS", nchannels)
    spec.header.set("AREASCAL", 1.0)
    spec.header.set("BACKSCAL", 1.0)

    return spec


def save_spec(spec, obsid, detid, detector, output_path):
    obs_path = output_path.joinpath(obsid)
    if not obs_path.exists():
        obs_path.mkdir()

    spec_path = obs_path.joinpath(f"{detid}_SRSPEC_{detector.long}.pha")
    spec.writeto(spec_path, overwrite=True)


def main(args):
    output_path = set_output_path(args.output_folder)
    detector = set_detector(args.detector)
    epoch_limits = get_epoch_limits(detector)

    #### Load counts and ancillary data from the catalogue
    (
        detids,
        obsids,
        obs_dates,
        det_obs_modes,
        det_rates,
        det_rates_err,
        det_filters,
    ) = load_data(args.sources_table, detector)

    #### Create pseudospectra
    default_rmf_file = set_rmf_file(
        obs_dates[0], det_obs_modes[0], detector, epoch_limits
    )
    spec_channels, spec_energies, spec_quality, spec_grouping = set_default_spectrum(
        default_rmf_file
    )

    for (
        detid,
        obsid,
        obs_date,
        det_obs_mode,
        det_rate,
        det_rate_err,
        det_filter,
    ) in tzip(
        detids, obsids, obs_dates, det_obs_modes, det_rates, det_rates_err, det_filters
    ):
        if not_detected(det_rate):
            continue

        rsp_file = set_rmf_file(obs_date, det_obs_mode, detector, epoch_limits)
        arf_file = set_arf_file(detector, det_filter)
        spec_rate, spec_rate_err = set_spec_count_rates(
            spec_energies, det_rate, det_rate_err
        )

        spec = set_spec_fits(
            spec_channels, spec_rate, spec_rate_err, spec_quality, spec_grouping
        )
        spec = update_spec_fits_header(
            spec, detector, rsp_file, arf_file, len(spec_channels)
        )

        save_spec(spec, obsid, detid, detector, output_path)


if __name__ == "__main__":
    # Parser for shell parameters
    parser = argparse.ArgumentParser(description="Create X-ray pseudospectra.")

    parser.add_argument(
        "--catalogue",
        dest="sources_table",
        action="store",
        default=None,
        help="Full path for the detections catalogue.",
    )
    parser.add_argument(
        "--camera",
        dest="detector",
        action="store",
        default="PN",
        help="EPIC detector for generating the spectra (PN/M1/M2).",
    )
    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        action="store",
        default="./data/",
        help="Path for saving the generated spectra.",
    )

    args = parser.parse_args()
    main(args)

