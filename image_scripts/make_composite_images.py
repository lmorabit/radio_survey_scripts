import aplpy
import montage_wrapper as montage
import numpy as np
from astropy.io import fits
import glob
import argparse
import os
import matplotlib.pyplot as pyplot
import re
from astropy.wcs import WCS
from astropy.stats import median_absolute_deviation as apy_mad
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

def angdist( ra, de, ra_arr, de_arr ):
    ## get the angular distance between one position and a list of possitions
    ## convert to radians
    ra_rad = ra * np.pi / 180.
    de_rad = de * np.pi / 180.
    ra_arr_rad = ra_arr * np.pi / 180.
    de_arr_rad = de_arr * np.pi / 180.
    sra = np.sin( 0.5 * ( ra_arr_rad - ra_rad ) )
    sde = np.sin( 0.5 * ( de_arr_rad - de_rad ) )
    diff_rad = 2 * np.sin( np.sqrt( sde*sde + sra*sra*np.cos(de_arr_rad)*np.cos(de_rad)))
    diff_deg = diff_rad * 180. / np.pi
    return diff_deg

def deg2hms( hmsval ):
    ## convert degrees to sexagesimal
    hour = np.int( np.floor( hmsval/15 ) )
    minute = np.int( np.floor( ( hmsval/15 - np.floor( hmsval/15 ) ) * 60 ) )
    second = ( ( ( hmsval/15 - np.floor( hmsval/15 ) ) * 60 ) - np.floor( ( ( hmsval/15 - np.floor( hmsval/15 ) )* 60 ) ) ) * 60 
    return hour, minute, second

def deg2dms( dmsval ):
    ## convert hour-degrees (i.e., RA) to sexagesimal
    hour = np.int( np.floor( dmsval ) )
    minute = np.int( np.floor( ( dmsval - np.floor( dmsval ) ) * 60 ) )
    second = ( ( ( dmsval - np.floor( dmsval ) ) * 60 ) - np.floor( ( ( dmsval - np.floor( dmsval ) )* 60 ) ) ) * 60 
    return hour, minute, second

def make_zscale( myfits, contrast=0.5 ):
    ## use the IRAF algorithm to create a zscale
    ## this is based on the median of the data
    hdu = fits.open( myfits )
    imdat = hdu[0].data
    values = np.squeeze(np.reshape(imdat, imdat.shape[0]*imdat.shape[1]) )
    ## sort the values
    values.sort()
    ## get rid of zeros
    fin_index = np.where( values > 0 )[0]
    sorted_vals = values[fin_index]
    abscissa = range(0,len(sorted_vals))
    midpoint = np.median( abscissa )

    x = abscissa - midpoint
    y = sorted_vals
    result = np.polyfit( x, y, 1 )
    
    midpoint_value = sorted_vals[np.where(np.abs(abscissa-midpoint) == np.min(np.abs(abscissa-midpoint)))[0]]
    z1 = midpoint_value + ( result[0] / contrast ) * (1-midpoint)
    z2 = midpoint_value + ( result[0] / contrast ) * (len(abscissa)-midpoint)
    
    return z1[0], z2[0]

def make_im_limits( myfits, minsig=-1.0, maxsig=20 ):
    ## a nicer image scale than zscale
    hdu = fits.open( myfits )
    imdat = hdu[0].data 
    mad = apy_mad( imdat ) * 1.48 ## to convert from mad to standard devation
    z1 = minsig * mad + np.median( imdat )
    z2 = maxsig * mad + np.median( imdat )
    return z1, z2

def define_contours( imdata, nlevels=3, maxval=0.8 ):
    ## get the MAD
    im_mad = apy_mad( imdata )
    ## always show 3 and 5 sigma
    mylevels = np.array([3,5])*im_mad
    ## find the data maximum
    dat_max = np.max( imdata ) * maxval
    ## if it's big enough, show some more contours:
    if dat_max > im_mad * 5:
        stepsize = ( dat_max - mylevels[-1] )/nlevels
        for rr in range(1,nlevels+1):
	    mylevels = np.append( mylevels, stepsize+mylevels[-1] )
    return mylevels

def get_list_for_cutouts( fitslist ):
    ## make a file for all the SDSS cutouts that need to be downloaded
    with open( 'positions_for_cutouts.dat', 'w' ) as f:
        for fitsfile in fitslist:
            rad_header = fits.getheader(fitsfile)
            rad_wcs = WCS( rad_header )
            cpix = rad_header['NAXIS1']/2.
            naxis = rad_header['NAXIS']
            if naxis == 4:
                tmp = [cpix, cpix, 0, 0]
            if naxis == 2:
                tmp = [cpix, cpix]
            pixdat = np.reshape( tmp, (1,naxis) )
            radec = rad_wcs.all_pix2world( pixdat, 1 )
            ra = radec[0][0]
            dec = radec[0][1]

            f.write( str(ra) + ' ' + str(dec) + '\n' )
    f.close()
    ## then this has to be uploaded to the website: https://dr12.sdss.org/bulkFields/raDecFile and the files can be downloaded using wget 
    ## unzip them using bzip2 -d file.bz2

def download_first_cutout( fitsfile, imsize_arcmin=1.8 ):
    ## download an image cutout from FIRST
    rad_header = fits.getheader(fitsfile)
    rad_wcs = WCS( rad_header )
    cpix = rad_header['NAXIS1']/2.
    naxis = rad_header['NAXIS']
    if naxis == 4:
        tmp = [cpix, cpix, 0, 0]
    if naxis == 2:
        tmp = [cpix, cpix]
    pixdat = np.reshape( tmp, (1,naxis) )
    radec = rad_wcs.all_pix2world( pixdat, 1 )
    ra = radec[0][0]
    dec = radec[0][1]
    rah, ram, ras = deg2hms( ra )
    ded, dem, des = deg2dms( dec )

    fitsname = fitsfile.replace( '.fits', '_first.fits' )

    ss = 'wget -O '+fitsname + ' "https://third.ucllnl.org/cgi-bin/firstimage?RA=' + str(rah) + '%20' + str(ram) + '%20' + str(ras) + '%20%2B' + str(ded) + '%20' + str(dem) + '%20' + str(des) + '&Dec=&Equinox=J2000&ImageSize=' + str(imsize_arcmin) + '&FITS=1"'
    os.system( ss )
    return fitsname

#####################################################
########## MAIN

## get a list of the sdss image files, find their pointing centres, and make a list
sdss_pattern = '/home/morabito/data/balqsos/lotss/lotss_cutouts/bulk_sdss/frame-g*fits'
sdss_files = glob.glob( sdss_pattern )

with open( 'sdss_cutout_centres.dat', 'w' ) as f:
    sdss_ra = []
    sdss_dec = []
    for sdss_file in sdss_files:
        sdss_header = fits.getheader( sdss_file )
        sdss_wcs = WCS( sdss_header )
        cpix1 = sdss_header['NAXIS1']/2
        cpix2 = sdss_header['NAXIS2']/2
        radec = sdss_wcs.all_pix2world( cpix1, cpix2, 1 )
        sdss_ra.append(np.float(radec[0]))
        sdss_dec.append(np.float(radec[1]))
        f.write( sdss_file + ' ' + str(np.float(radec[0])) + ' ' + str(np.float(radec[1])) + '\n' )

f.close()

sdss_files = np.array( sdss_files )
sdss_ra = np.array( sdss_ra )
sdss_dec = np.array( sdss_dec )


######### set up some plotting parameters for the radio contours
base_sigmas = np.array([3,5])
mysigmas = np.array([3,5,10,20,40])

do_inset = True
myfigsize = 2
titlefontsize = myfigsize*3.75
mylw = myfigsize * 0.125

## get a list of the lofar cutouts
imlist = glob.glob('*.fits')
#imlist = imlist[0:5]

os.system('mkdir FIRST_cutouts')

## loop through them to make rgb plots of sdss with lofar + FIRST contours overlaid
for myim in imlist:

    filestem = myim.replace('.fits','')
    outfile = filestem + '.pdf'
    ## first check if the outfile already exists
    file_check = os.path.isfile( outfile )
    if file_check:
        print 'File '+outfile+' already exists, skipping!'
    else:
        ## open lofar file to get some information 
        with fits.open( myim ) as myim_hdu:
            rad_header = myim_hdu[0].header
            rad_image = myim_hdu[0].data
            ## get central RA, DEC
            rad_wcs = WCS( rad_header )
            cpix = rad_header['NAXIS1']/2.
            naxis = rad_header['NAXIS']
            if naxis == 4:
                tmp = [cpix, cpix, 0, 0]
            if naxis == 2:
                tmp = [cpix, cpix]
            pixdat = np.reshape( tmp, (1,naxis) )
            radec = rad_wcs.all_pix2world( pixdat, 1 )
            ra = radec[0][0]
            dec = radec[0][1]
            ## get MAD for lofar image
            lofar_mad = apy_mad( rad_image )
        #print lofar_mad
        #lofar_levels = lofar_mad * mysigmas
        lofar_levels = define_contours( rad_image, nlevels=3, maxval=0.8 )


	## find the right sdss pointing
	distances = angdist( ra, dec, sdss_ra, sdss_dec )
        red_file = sdss_files[np.where( distances == min(distances) )[0] ][0]
        green_file = red_file.replace('frame-g','frame-r')
        blue_file = red_file.replace('frame-g','frame-i')

        ## check if first file exists first
        first_check = os.path.isfile( 'FIRST_cutouts/' + filestem + '_first.fits' )
        if first_check:
	    first_file = 'FIRST_cutouts/' + filestem + '_first.fits'
            hdu = fits.open( first_file )
            first_image = hdu[0].data
            first_mad = apy_mad( first_image )
            hdu.close()
	    first_levels = define_contours( first_image, nlevels=3, maxval=0.8 )
        else: 
            ## download first cutout
            first_file = download_first_cutout( myim, imsize_arcmin=1.8 )
            with fits.open( first_file, mode='update' ) as hdu:
  		first_header = hdu[0].header
	        first_image = hdu[0].data
        	first_mad = apy_mad( first_image )
                first_naxis = first_header['NAXIS']
	        first_wcs = WCS( first_header )
        	if first_wcs.naxis > first_naxis:
                    ## remove extra keywords for CR*3 and CR*4
	            first_header.remove('CTYPE3')
        	    first_header.remove('CRVAL3')
                    first_header.remove('CDELT3')
	            first_header.remove('CRPIX3')
        	    first_header.remove('CROTA3')

                    first_header.remove('CTYPE4')
	            first_header.remove('CRVAL4')
        	    first_header.remove('CDELT4')
                    first_header.remove('CRPIX4')
		    first_header.remove('CROTA4')
            hdu.flush()
            #    print first_mad
            #first_levels = first_mad * mysigmas
            first_levels = define_contours( first_image, nlevels=3, maxval=0.8 )
        
            ## check that all the files exist
            a = os.path.exists( red_file )
            b = os.path.exists( green_file )
            c = os.path.exists( blue_file )
            if (a+b+c) == 3:    
                print 'creating image for '+filestem
                ## sdss rgb cube
                aplpy.make_rgb_cube([red_file,green_file,blue_file],'rgb.fits')
    
                ## zscaling
                #rz1, rz2 = make_zscale( red_file, contrast=0.5 )
                #gz1, gz2 = make_zscale( green_file, contrast=0.5 )
                #bz1, bz2 = make_zscale( blue_file, contrast=0.5 )

                ## corentin scale
                rz1, rz2 = make_im_limits( red_file, minsig=-5, maxsig=20 )
                bz1, bz2 = make_im_limits( blue_file, minsig=-5, maxsig=20 )
                gz1, gz2 = make_im_limits( green_file, minsig=-5, maxsig=20 )

                aplpy.make_rgb_image('rgb.fits',filestem+'.png', vmin_r=rz1, vmax_r=rz2, vmin_g=gz1, vmax_g=gz2, vmin_b=bz1, vmax_b=bz2)

                fig = plt.figure(figsize=(myfigsize,myfigsize))

                ## start an image object
                img = aplpy.FITSFigure('rgb_2d.fits',figure=fig,subplot=[0.075,0.05,0.85,0.85])  ## try an inset subplot
                img.show_rgb(filestem+'.png')
                img.show_contour(first_file, levels=first_levels, colors='red', smooth=1, kernel='gauss', linewidths=mylw)
                img.show_contour(myim, levels=lofar_levels, colors='white', smooth=1, kernel='gauss', linewidths=mylw )
                img.recenter( ra, dec, width=0.02777, height=0.02777 )
                img.hide_axis_labels()
                img.hide_tick_labels()
                img.ticks.hide()


                img2 = aplpy.FITSFigure('rgb_2d.fits', figure=fig, subplot=[0.075,0.7,0.2,0.2])
                img2.show_rgb(filestem+'.png')
                img2.show_contour(first_file, levels=first_levels, colors='red', smooth=1, kernel='gauss', linewidths=0.9*mylw)
                img2.show_contour(myim, levels=lofar_levels, colors='white', smooth=1, kernel='gauss', linewidths=0.9*mylw)
                img2.recenter( ra-0.0006, dec+0.0004, width=0.002777, height=0.002777)
                #img2.recenter( ra, dec, width=0.002777, height=0.002777)
                img2.hide_axis_labels()
                img2.hide_tick_labels()
                img2.ticks.hide()
                img2.frame.set_linewidth(2)
                img2.frame.set_color('white')
        
                fig.suptitle( filestem, fontsize=titlefontsize )

                fig.savefig(outfile)

                os.system('rm rgb*fits')
                os.system('rm *png')
            if not first_check:
                os.system('mv *_first.fits FIRST_cutouts/')

