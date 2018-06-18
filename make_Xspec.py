#!/usr/bin/env python
"""
Created on Fri Jun 12 12:40:55 2015

@author: ruizca
"""

import os
import argparse

from tqdm import trange
from astropy.io import fits as pyfits
from astropy.time import Time
import numpy as np


def get_epoch(obsdate, limits):
    """
    Returns the epoch of the observation with date obsdate 
    given the epochs dates defined in limits.
    """    
    epoch = 'e{:d}'.format(len(limits) + 1)
    for i,d in enumerate(obsdate<limits):
        if d:
            epoch = 'e{:d}'.format(i + 1)
            break
        
    return epoch

def get_rmf(obsdate, obsmode, detector, limits, 
            calib_folder='calib/rmfcanned/', rmfversion='16.0'):
    
    epoch = get_epoch(obsdate, limits)
    if detector == 'PN':        
        rmf = 'epn_{}_{}20_sdY9_v{}.rmf'.format(epoch, obsmode.decode('UTF-8'), 
                                                rmfversion)
                           
    elif detector == 'MOS1': 
        rmf = 'm1_{}_im_pall_o.rmf'.format(epoch)

    elif detector == 'MOS2': 
        rmf = 'm2_{}_im_pall_o.rmf'.format(epoch)  
            
    return os.path.join(calib_folder, rmf)

def main(args):
    dest_folder = args.dest_folder
    sample_file = args.sources_table
    detector = args.detector

    spec_folder= os.path.join(dest_folder, 'spectra')
    if not os.path.exists(spec_folder) :
        os.makedirs(spec_folder)
    
    #### Define limits of observation epochs for each camera 
    # https://www.cosmos.esa.int/web/xmm-newton/epic-response-files
    if detector == 'PN':
        epoch_limits = ['2007-01-01', '2014-01-01']
        
    elif detector.startswith('MOS'):
        epoch_limits = ['2001-01-01', '2001-10-01', '2002-01-01', '2002-10-01',
                        '2003-01-01', '2004-10-01', '2005-10-01', '2006-10-01',
                        '2007-10-01', '2008-10-01', '2009-10-01', '2010-10-01',
                        '2011-10-01']     
    else :
        raise ValueError('Unknownk detector!!!')
       
    for i,e in enumerate(epoch_limits):
        epoch_limits[i] = Time(e, format='iso')
     
    #### Load counts from the catalogue
    try :
        hdulist = pyfits.open(sample_file)
    except :
        raise ValueError("Error: fits table not found!")
        
    table = hdulist[1].data
    
    # SDSS ID, det. ID and obs. date
    detid = table.field('DETID')
    obsid = table.field('OBS_ID')
    obs_date = Time(table.field('MJD_START'), format='mjd')
    
    # Detector count rates
    colpref = detector[0] + detector[-1]
    det_obs_mode = table.field(colpref + '_SUBMODE')
    
    det_rate = np.array([table.field(colpref + '_1_RATE'),
               table.field(colpref + '_2_RATE'),
               table.field(colpref + '_3_RATE'), 
               table.field(colpref + '_4_RATE'),
               table.field(colpref + '_5_RATE')])
    
    det_rateErr = np.array([table.field(colpref + '_1_RATE_ERR'),
                  table.field(colpref + '_2_RATE_ERR'),
                  table.field(colpref + '_3_RATE_ERR'),
                  table.field(colpref + '_4_RATE_ERR'),
                  table.field(colpref + '_5_RATE_ERR')])
    
    det_filter = table.field(colpref + '_FILTER')
    
    hdulist.close()
    
    # Define obs. mode
    obs_mode = np.zeros(len(obs_date), dtype='a3')
    
    if detector == 'PN' :
        idx_ff = det_obs_mode == 'PrimeFullWindow'
        idx_ef = det_obs_mode == 'PrimeFullWindowExtended'
        idx_lw = det_obs_mode == 'PrimeLargeWindow'
        idx_un = det_obs_mode == 'UNDEFINED'
    
        obs_mode[idx_ff] = 'ff'
        obs_mode[idx_ef] = 'ef'
        obs_mode[idx_lw] = 'lw'
        obs_mode[idx_un] = 'UND'

    #### Create pseudospectra   
    # Open response matrix
    try :
        rmf_file = get_rmf(obs_date[0], obs_mode[0], detector, epoch_limits)
        hdulist = pyfits.open(rmf_file)
    except :
        raise ValueError("Error: rmf file not found!")
    
    table = hdulist[2].data    
    channels = table.field('CHANNEL')
    E_min    = table.field('E_MIN')
    
    hdulist.close()
    
    # Define columns for the pseudospectrum
    spec_channels = channels
    spec_quality  = np.zeros(len(channels), dtype='int')
    spec_grouping = -1 * np.ones(len(channels), dtype='int')
    
    # Define bad channels 
    idx_badch = np.logical_or(E_min < 0.2, E_min >= 12.0)
    spec_quality[idx_badch] = 1
    spec_grouping[idx_badch] = 1
    
    # Define grouping
    band_limits = [0.2, 0.5, 1.0, 2.0, 4.5, 12.0]
    
    for band_min in band_limits :
        if band_min == 12.0 and detector.startswith('MOS'):
            continue
        msk = np.where(E_min >= band_min)[0]
        spec_grouping[msk[0]] = 1


    arf_str = 'calib/ARF/{}_{}.arf'
    spec_filename = '{}_SRSPEC_E{}.pha'

    for i in trange(len(detid)) :
        # Check that the source was detected in this camera
        if np.all(np.isnan(det_rate[:,i])):
            continue
    
        # Define response and ARF name
        arf_name = arf_str.format(detector, det_filter[i])
        rsp_name = get_rmf(obs_date[i], obs_mode[i], detector, epoch_limits)
              
        # Define count rates
        spec_rate     = np.zeros(len(channels))
        spec_rateErr  = np.zeros(len(channels))
    
        j = 0
        for Emin, Emax in zip(band_limits[:-1],band_limits[1:]) :
            idx_chband = np.nonzero(np.logical_and(E_min>=Emin, E_min<Emax))
    
            idx_rate = np.random.choice(idx_chband[0])
            spec_rate[idx_rate] = det_rate[j,i]
            spec_rateErr[idx_rate] = det_rateErr[j,i]
            j+=1
    
        # Define new columns
        spec_cols = pyfits.ColDefs([
            pyfits.Column(name='CHANNEL', format='I', array=spec_channels),
            pyfits.Column(name='RATE', format='E', unit='counts/s', array=spec_rate),
            pyfits.Column(name='STAT_ERR', format='E', array=spec_rateErr),
            pyfits.Column(name='QUALITY', format='I', array=spec_quality),
            pyfits.Column(name='GROUPING', format='I', array=spec_grouping)])
              
        # Create new spectrum
        newspec = pyfits.BinTableHDU.from_columns(spec_cols)
        
        # Update header
        newspechdr = newspec.header
        newspechdr.set('EXTNAME', 'SPECTRUM')
        newspechdr.set('TELESCOP', 'XMM')
        newspechdr.set('INSTRUME', 'E' + detector)
        newspechdr.set('FILTER', 'NONE')
        newspechdr.set('EXPOSURE', 1.)
        newspechdr.set('BACKFILE', 'NONE')
        newspechdr.set('CORRFILE', 'NONE')
        newspechdr.set('CORRSCAL', 1.)
        newspechdr.set('RESPFILE', rsp_name)
        newspechdr.set('ANCRFILE', arf_name)
        newspechdr.set('HDUCLASS', 'OGIP')
        newspechdr.set('HDUCLAS1', 'SPECTRUM')
        newspechdr.set('HDUVERS', '1.3.0')
        newspechdr.set('POISSERR', False)
        newspechdr.set('CHANTYPE', 'PI')
        newspechdr.set('DETCHANS', len(spec_channels))
        newspechdr.set('AREASCAL', 1.)
        newspechdr.set('BACKSCAL', 1.)
    
        # Save pseudospec
        obs_folder = os.path.join(spec_folder, obsid[i])
        if not os.path.exists(obs_folder) :
            os.makedirs(obs_folder)
        
        spec_file = spec_filename.format(detid[i], detector)        
        spec_file = os.path.join(obs_folder, spec_file)
        newspec.writeto(spec_file, overwrite=True)


if __name__ == '__main__' :
    # Parser for shell parameters
    parser = argparse.ArgumentParser(description='Create X-ray pseudospectra.')
                                             
    parser.add_argument('--catalogue', dest='sources_table', 
                        action='store', default=None, 
                        help='Full route to the detections catalogue.')
    
    parser.add_argument('--camera', dest='detector', 
                        action='store', default='PN',
                        help='EPIC detector for generating the spectra.')

    parser.add_argument('--spec_folder', dest='dest_folder', 
                        action='store', default='./data/', 
                        help='Folder for saving the generated spectra.')
    
    args = parser.parse_args()
    main(args)
    