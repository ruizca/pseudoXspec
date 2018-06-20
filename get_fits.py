#!/usr/bin/env python
"""
Created on Fri Jun 19 10:46:32 2015

@author: ruizca
"""
import os
import json
import argparse

from tqdm import tqdm
from astropy.table import Table

def main(args):

    sample = Table.read(args.sources_table)
    nrows = sample['DETID'].size

    results = Table()
    results['SRCID'] = sample['SRCID']
    results['DETID'] = sample['DETID']    
    results['chi2'] = [None]*nrows
    results['dof'] = [None]*nrows
    results['nparams'] = [None]*nrows
    results['PhoIndex'] = [None]*nrows
    results['PhoIndexErrLo'] = [None]*nrows
    results['PhoIndexErrHi'] = [None]*nrows
    results['NH'] = [None]*nrows
    results['NHErrLo'] = [None]*nrows
    results['NHErrHi'] = [None]*nrows
    results['LHX'] = [None]*nrows
    results['LHXErrLo'] = [None]*nrows
    results['LHXErrHi'] = [None]*nrows

    for i, src in enumerate(tqdm(sample)):
        obs_folder = os.path.join(args.dest_folder, src['OBS_ID'])

        fitstats = os.path.join(obs_folder, 
                                '{}_bestfit_goodness.json'.format(src['DETID']))
        try:
            with open(fitstats) as json_data:
                d = json.load(json_data)
            results['chi2'][i] = d['chi2']
            results['dof'][i] = d['dof']
            results['nparams'][i] = d['npar']

        except FileNotFoundError as error:
            print(error)
            print(i, src['DETID'])


        fitpars = os.path.join(obs_folder, 
                               '{}_bestfit_params.json'.format(src['DETID']))
        try:
            with open(fitpars) as json_data:
                d = json.load(json_data)
            results['PhoIndex'][i] = d['PhoIndex']
            results['PhoIndexErrLo'][i] = d['PhoIndex_ErrMin']
            results['PhoIndexErrHi'][i] = d['PhoIndex_ErrMax']
            results['NH'][i] = d['nH']
            results['NHErrLo'][i] = d['nH_ErrMin']
            results['NHErrHi'][i] = d['nH_ErrMax']

        except FileNotFoundError as error:
            print(error)
            print(i, src['DETID'])

        
        fitflux = os.path.join(obs_folder, 
                               '{}_bestfit_fluxes.json'.format(src['DETID']))
        try:
            with open(fitflux) as json_data:
                d = json.load(json_data)
            results['LHX'][i] = d['Lx']
            results['LHXErrLo'][i] = d['Lx_ErrMin']
            results['LHXErrHi'][i] = d['Lx_ErrMax']

        except FileNotFoundError as error:
            print(error)
            print(i, src['DETID'])

    results_file = os.path.join(args.dest_folder, 'results.csv')
    results.write(results_file, overwrite=True)


if __name__ == '__main__' :
    # Parser for shell parameters
    parser = argparse.ArgumentParser(description='Fitting X-ray pseudospectra')
                                             
    parser.add_argument('--catalogue', dest='sources_table', action='store',
                        default=None, 
                        help='Full route to the detections catalogue.')

    parser.add_argument('--fits_folder', dest='dest_folder', action='store',
                        default='./fit_results/', 
                        help='Folder for saving the generated spectra.')

    main(parser.parse_args())