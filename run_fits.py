#!/usr/bin/env python

"""
Created on Fri Jun 19 10:46:32 2015

@author: ruizca
"""
import os
import glob
import argparse
import subprocess
import logging

from tqdm import trange
from astropy.table import Table

logging.basicConfig(level=logging.INFO)


def nh(ra, dec, equinox=2000.0):
    command = 'nh {} {} {}'.format(equinox, ra, dec)
    log = subprocess.check_output(command, shell=True)

    for line in log.splitlines():
        if 'LAB >> Weighted' in line.decode('utf-8'):
            nhval = line.split()[-1]

    return float(nhval)


def get_last_source_fit(last_source_file):
    try:
        with open(last_source_file, "r") as fp:
            first_source = int(fp.readline())
    except FileNotFoundError:
        first_source = 0

    return first_source


def update_last_source_fit(last_source, last_source_file):
    with open(last_source_file, "w") as fp:
        fp.write(str(last_source))


def check_results_folder(results_folder):
    if os.path.exists(results_folder):
        message = 'results_folder "{}" already exists!'
        raise FileExistsError(message.format(results_folder))
    else:
        os.makedirs(results_folder)


def get_sources_data(table_name, racol, deccol, zcol, first_source=0):
    table = Table.read(table_name)
    table = table[first_source:]

    ra = table[racol]
    dec = table[deccol]
    obsid = table['OBS_ID']
    detid = table['DETID']
    z = table[zcol]

    return ra, dec, obsid, detid, z


def stack_spectra(obsid, detid, spec_folder):
    ## Find spectra of interest for this detection
    obs_folder = os.path.join(spec_folder, obsid)
    spec_files = glob.glob("{}/{}_SRSPEC_*.pha".format(obs_folder, detid))

    # Create stack file for existing spectra in the observation
    stack_file = 'spec_{}.lis'.format(detid)
    with open(stack_file, "w") as fp:
        for spec in spec_files:
            fp.write(spec + '\n')

    return stack_file


def remove_stack_spectra(stack_file):
    try:
        os.remove(stack_file)
        #pass
    except FileNotFoundError:
        logging.warning("No stack file!")


def fit_command(z, nh, obsid, detid, folder, fixgamma=True):
    command = './fit_Xspec.py -z {:f} -nh {} -obsid {} -detid {} -folder {} '
    command = command.format(z, nh, obsid, detid, folder)

    if fixgamma:
        command = ' '.join([command, '--fixGamma'])

    return command


def main(args):
    spec_folder = args.dest_folder
    results_folder = args.results_folder
    lastsource_file = args.file_lastsource

    ## Find last fitted source
    first_source = get_last_source_fit(lastsource_file)

    ## Check if results_folder exists
    if first_source == 0:
        check_results_folder(results_folder)

    ra, dec, obsid, detid, z = get_sources_data(args.sources_table, args.racol,
                                                args.deccol, args.zcol)

    for i in trange(detid.size):
        stack_file = stack_spectra(obsid[i], detid[i], spec_folder)

        ## Fit detection
        cmd = fit_command(z[i], nh(ra[i], dec[i]), obsid[i], detid[i],
                          results_folder, args.fixgamma)
        logging.debug(cmd)

        try:
            subprocess.call(cmd, shell=True)
            update_last_source_fit(first_source + i, lastsource_file)

        except:
            logging.error("Something went wrong fitting det. {}".format(detid[i]))

        remove_stack_spectra(stack_file)


if __name__ == '__main__':
    # Parser for shell parameters
    parser = argparse.ArgumentParser(description='Fitting X-ray pseudospectra')

    parser.add_argument('--catalogue', dest='sources_table', action='store',
                        default=None,
                        help='Full route to the detections catalogue.')

    parser.add_argument('--spec_folder', dest='dest_folder', action='store',
                        default='./data/spectra/',
                        help='Folder of the pseudospectra.')

    parser.add_argument('--results_folder', dest='results_folder', action='store',
                        default='./fit_results/',
                        help='Folder for saving the fit results.')

    parser.add_argument('--racol', dest='racol', action='store', default='XMM_RA',
                        help='Name of the RA column in the catalogue.')

    parser.add_argument('--deccol', dest='deccol', action='store', default='XMM_DEC',
                        help='Name of the Dec column in the catalogue.')

    parser.add_argument('--zcol', dest='zcol', action='store', default='z',
                        help='Name of the redshift column in the catalogue.')

    parser.add_argument('--lsf', dest='file_lastsource', action='store',
                        default='last_source.dat',
                        help='File to store the last fitted source.')

    parser.add_argument('--fixGamma', dest='fixgamma', action='store_true',
                        default=False, help='Fit with a fixed photon index (1.9).')

    main(parser.parse_args())
