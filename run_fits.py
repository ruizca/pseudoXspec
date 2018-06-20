#!/usr/bin/env python

"""
Created on Fri Jun 19 10:46:32 2015

@author: ruizca
"""
#from __future__ import print_function

import os
import glob
import argparse
import subprocess
import logging

#import numpy as np
from tqdm import trange
from astropy.table import Table

def nh(ra, dec, equinox=2000.0) :
    command = "nh " + str(equinox) + " " + str(ra) + " " + str(dec) + "> tmp"
    os.system(command)

    nhtemp = open("tmp", "r")
    for line in nhtemp :
        if line.startswith("  LAB >> Weighted") :
            nhval = line.split()[-1]
    os.remove("tmp")
        
    return float(nhval)

def main(args):
    spec_folder = args.dest_folder

    ### Find last fitted source
    try :
        fp = open(args.file_lastsource, "r")
        first_source = int(fp.readline())
        fp.close()
    except :
        first_source = 0
        
    ### Load data of detections
    table = Table.read(args.sources_table)
    table = table[first_source:]
    ra = table['XMM_RA']
    dec = table['XMM_DEC']
    obsid = table['OBS_ID']
    detid = table['DETID']
    z = table[args.zcol]
    
    args_str = "-z {:f} -nh {} -obsid {} -detid {} "    
    for i in trange(detid.size):
        ## Find spectra of interest for this detection
        obs_folder = os.path.join(spec_folder, obsid[i])
        spec_files = glob.glob("{}/{}_SRSPEC_*.pha".format(obs_folder, detid[i]))    

        # Create stack file for existing spectra in the observation
        stack_file = 'spec_{}.lis'.format(detid[i])
        fp = open(stack_file, "w")
        for spec in spec_files:
            fp.write(spec + '\n')
        fp.close()     
        
        ## Fit detection
        fit_command = "./fit_Xspec.py "
        fit_args = args_str.format(z[i], nh(ra[i], dec[i]), 
                                   obsid[i], detid[i])
        logging.debug(fit_command + fit_args)
        
        try:
            subprocess.call(fit_command + fit_args, shell=True)

            fp = open(args.file_lastsource, "w")
            fp.write(str(i+first_source))
            fp.close()            
        except:
            logging.error("Something went wrong fitting det. {}".format(detid[i]))

        try:
            os.remove(stack_file)
            #pass
        except:
            logging.warning("No stack file!")

        break

if __name__ == '__main__' :
    # Parser for shell parameters
    parser = argparse.ArgumentParser(description='Fitting X-ray pseudospectra')
                                             
    parser.add_argument('--catalogue', dest='sources_table', action='store',
                        default=None, 
                        help='Full route to the detections catalogue.')

    parser.add_argument('--spec_folder', dest='dest_folder', action='store',
                        default='./data/spectra/', 
                        help='Folder for saving the generated spectra.')

    parser.add_argument('--zcol', dest='zcol', action='store',
                        default='z', 
                        help='Name of the redshift column in the catalogue.')
                        
    parser.add_argument('--lsf', dest='file_lastsource', action='store',
                        default='last_source.dat',
                        help='File to store the last fitted source.')

    main(parser.parse_args())