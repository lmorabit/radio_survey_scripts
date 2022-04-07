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

def define_contours_3( imdata, minsig=5, max_n=10, maxval=0.8, noise=0. ):
    ## set the minimum level
    minlevel = noise * minsig
    ## find the data maximum
    dat_max = np.max( imdata ) * maxval
    ## make an array
    mylevels = np.array([minlevel])
    while np.max(mylevels) < dat_max:
        mylevels = np.append( mylevels, mylevels[-1]*np.sqrt(2.) )
    if len(mylevels) > max_n:
        mylevels = mylevels[np.arange(0,len(mylevels),2)]
    return mylevels

def get_rms(fitsimagename,boxsize=1000,niter=20,eps=1e-6,verbose=False):
    hdu = fits.open(fitsimagename)
    data=hdu[0].data
    boxsize = int(abs(hdu[0].header['CRPIX2']))-10
    print boxsize,'boxsize'
    if len(data.shape)==4:
        _,_,ys,xs=data.shape
        subim=data[0,0,ys/2-boxsize/2:ys/2+boxsize/2,xs/2-boxsize/2:xs/2+boxsize/2].flatten()
    else:
        ys,xs=data.shape
        subim=data[ys/2-boxsize/2:ys/2+boxsize/2,xs/2-boxsize/2:xs/2+boxsize/2].flatten()
    oldrms=1
    for i in range(niter):
        rms=np.std(subim)
        if verbose: print len(subim),rms
        if np.abs(oldrms-rms)/rms < eps:
            return rms
        subim=subim[np.abs(subim)<5*rms]
        oldrms=rms
    hdu.close()
    return rms


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
            if ra < 0:
                ra = ra + 360

            f.write( str(ra) + ' ' + str(dec) + '\n' )
    f.close()
    ## then this has to be uploaded to the website: https://dr12.sdss.org/bulkFields/raDecFile and the files can be downloaded using wget 
    ## unzip them using bzip2 -d file.bz2

def download_first_cutout( fitsfile, imsize_arcmin=1.8 ):

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

def main( lofar_pattern='*.fits', sdss_dir='', sdss_filters='g,r,i', cutout_size=0.02777, do_first=True, do_inset=False, do_beam=False, do_axes=False, clobber=False, nup='4x6' ):

    if sdss_dir != '':
	red_filter = 'frame-'+sdss_filters.split(',')[0]
	green_filter = 'frame-'+sdss_filters.split(',')[1]
	blue_filter = 'frame-'+sdss_filters.split(',')[2]

	## get a list of the sdss files
	sdss_files = glob.glob( sdss_dir + '/frame-*' )
	## check if they've been unzipped
	tmp = sdss_files[0].split('.')
	if tmp[1] == 'bz2':
	    for sdss_file in sdss_files:
		os.system( 'bzip2 -d '+sdss_file )
	## get a list of the centres
	sdss_files = glob.glob( sdss_dir+'/frame-'+sdss_filters.split(',')[0]+'*fits' )
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

    ## some imaging parameters
    myfigsize = 2
    titlefontsize = myfigsize*3.75
    mylw = myfigsize * 0.125

    ## get a list of the lofar cutouts
    imlist = glob.glob(lofar_pattern)

    if do_first:
	dir_check = os.path.isdir( 'FIRST_cutouts' )
	if not dir_check:
            os.system('mkdir FIRST_cutouts')

    ## loop through them to make rgb plots of sdss with lofar + FIRST contours overlaid
    for myim in imlist:

        filestem = myim.replace('.fits','')
        outfile = filestem + '.pdf'
        ## first check if the outfile already exists
        file_check = os.path.isfile( outfile )
        if file_check and not clobber:
            print 'File '+outfile+' already exists and clobber not set, skipping!'
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
		if do_beam:
		    ## get beam information
		    rad_bmaj = rad_header['BMAJ']
		    rad_bmin = rad_header['BMIN']
		    rad_bpa = rad_header['BPA']
            #print lofar_mad
            #lofar_levels = lofar_mad * mysigmas
            #lofar_levels = define_contours( rad_image, nlevels=3, maxval=0.8 )
            noise = get_rms( myim )
            print( noise )
            lofar_levels = define_contours_3( rad_image, minsig=3, max_n=9, maxval=1, noise=noise )
	    #print(lofar_levels)
	    #lofar_levels = [ 7.27495628e-04, 1.45495348e-03, 2.18241134e-03]

	    if sdss_dir != '':
		distances = angdist( ra, dec, sdss_ra, sdss_dec )
		red_file = sdss_files[np.where( distances == min(distances) )[0] ][0]
		green_file = red_file.replace( red_filter, green_filter )
		blue_file = red_file.replace( red_filter, blue_filter )
		## check that all the files exist
                a = os.path.exists( red_file )
                b = os.path.exists( green_file )
                c = os.path.exists( blue_file )
		sdss_exists = a + b + c
		if sdss_exists == 3:
                    print 'creating image for '+filestem
                    ## sdss rgb cube
                    aplpy.make_rgb_cube([red_file,green_file,blue_file],'rgb.fits')
		    ## some nice scaling
		    rz1, rz2 = make_im_limits( red_file, minsig=-5, maxsig=20 )
                    bz1, bz2 = make_im_limits( blue_file, minsig=-5, maxsig=20 )
                    gz1, gz2 = make_im_limits( green_file, minsig=-5, maxsig=20 )
		    aplpy.make_rgb_image('rgb.fits',filestem+'.png', vmin_r=rz1, vmax_r=rz2, vmin_g=gz1, vmax_g=gz2, vmin_b=bz1, vmax_b=bz2)

	    if do_first:
	        ## check if first file exists first
	        first_check = os.path.isfile( 'FIRST_cutouts/' + filestem + '_first.fits' )
        	if first_check:
		    first_file = 'FIRST_cutouts/' + filestem + '_first.fits'
        	    hdu = fits.open( first_file )
	            first_image = hdu[0].data
        	    first_mad = apy_mad( first_image )
		    if do_beam:
			first_header = hdu[0].header
			f_bmaj = first_header['BMAJ']
			f_bmin = first_header['BMIN']
			f_bpa = first_header['BPA']
	            hdu.close()
                    noise = get_rms( first_file )
                    first_levels = define_contours_3( first_image, minsig=3, max_n=9, maxval=1, noise=noise )
		    #first_levels = define_contours( first_image, nlevels=3, maxval=0.8 )
	        else: 
        	    ## download first cutout
	            first_file = download_first_cutout( myim, imsize_arcmin=1.8 )
        	    with fits.open( first_file, mode='update' ) as hdu:
	  		first_header = hdu[0].header
		        first_image = hdu[0].data
        		first_mad = apy_mad( first_image )
			if do_beam:
			    f_bmaj = first_header['BMAJ']
			    f_bmin = first_header['BMIN']
			    f_bpa = first_header['BPA']
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
        	    #first_levels = define_contours( first_image, nlevels=3, maxval=0.8 )
                    noise = get_rms( first_file )
                    first_levels = define_contours_3( first_image, minsig=3, max_n=9, maxval=1, noise=noise )


            fig = plt.figure(figsize=(myfigsize,myfigsize))

            ## start an image object
	    if sdss_dir !='':
		## use sdss as background
		## add beam information to rgb fits file
		if do_beam:
		    data, header = fits.getdata('rgb_2d.fits', header=True )
		    header['BMAJ'] = rad_bmaj
		    header['BMIN'] = rad_bmin
		    header['BPA'] = rad_bpa
		    fits.writeto('rgb_2d.fits', data, header, overwrite=True )
		if do_axes:
		    img = aplpy.FITSFigure('rgb_2d.fits',figure=fig,subplot=[0.2,0.1,0.75,0.8]) 
		else:
		    img = aplpy.FITSFigure('rgb_2d.fits',figure=fig,subplot=[0.075,0.05,0.85,0.85]) 
		img.show_rgb(filestem+'.png')
		if do_first:
		    img.show_contour(first_file, levels=first_levels, colors='red', smooth=1, kernel='gauss', linewidths=mylw)
		img.show_contour(myim, levels=lofar_levels, colors='white', smooth=1, kernel='gauss', linewidths=mylw )
	    else:
		## use lofar as background
                img = aplpy.FITSFigure(myim,figure=fig,subplot=[0.075,0.05,0.85,0.85])  ## try an inset subplot
                img.show_grayscale()
	        if do_first:
                    img.show_contour(first_file, levels=first_levels, colors='green', smooth=1, kernel='gauss', linewidths=mylw)
                img.show_contour(myim, levels=lofar_levels, colors='cyan', smooth=1, kernel='gauss', linewidths=mylw )
		
            img.recenter( ra, dec, width=cutout_size, height=cutout_size )
            #img.recenter( ra-0.0006, dec+0.0004, width=cutout_size, height=cutout_size )
	    if do_beam:
		img.add_beam()
		do_first = not do_first
		if do_first:
		    img.add_beam()
                    img.beam[0].set_color('white')
                    img.beam[0].set_corner('bottom right')
		    img.beam[1].set_major(f_bmaj)
		    img.beam[1].set_minor(f_bmin)
		    img.beam[1].set_angle(f_bpa)
		    img.beam[1].set_color('red')
		    img.beam[1].set_corner('bottom right')
		else:
		    img.beam.set_edgecolor('black')
		    img.beam.set_linewidth(0.05)
		    img.beam.set_facecolor('white')
		    img.beam.set_corner('bottom right')
		do_first = not do_first
	    if do_axes:
		img.tick_labels.set_xformat('mm:ss')
		img.tick_labels.set_yformat('mm:ss')
		img.tick_labels.set_font(size=6 )
	    else:
                img.hide_axis_labels()
                img.hide_tick_labels()
                img.ticks.hide()

	    if do_inset:
		## really this should only be done if you have sdss
		if do_axes:                
		    img2 = aplpy.FITSFigure('rgb_2d.fits', figure=fig, subplot=[0.2,0.7,0.2,0.2])
		else:
                    img2 = aplpy.FITSFigure('rgb_2d.fits', figure=fig, subplot=[0.075,0.7,0.2,0.2])
                img2.show_rgb(filestem+'.png')
		if do_first:
                    img2.show_contour(first_file, levels=first_levels, colors='red', smooth=1, kernel='gauss', linewidths=0.9*mylw)
                img2.show_contour(myim, levels=lofar_levels, colors='white', smooth=1, kernel='gauss', linewidths=0.9*mylw)
                img2.recenter( ra-0.0006, dec+0.0004, width=0.002777, height=0.002777)
                #img2.recenter( ra, dec, width=0.002777, height=0.002777)
                img2.hide_axis_labels()
                img2.hide_tick_labels()
                img2.ticks.hide()
                img2.frame.set_linewidth(2)
                img2.frame.set_color('white')

	    if do_axes:
                fig.suptitle( filestem, fontsize=titlefontsize, x=0.58, y=0.98 )
	    else:
                fig.suptitle( filestem, fontsize=titlefontsize )
            fig.savefig(outfile)

	    os.system('rm rgb*fits')
	    os.system('rm *png')
	    if do_first:
                if not first_check:
                    os.system('mv *_first.fits FIRST_cutouts/')

    # make a montage
    os.system( 'pdfnup -o Montage.pdf --no-landscape --nup '+nup+' I*pdf' )
#    nup_file = glob.glob( '*nup.pdf' )
#    os.system( 'mv '+nup_file[0]+' montage.pdf' )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument( dest='lofar_pattern', type=str, help='search pattern for lofar files', default='*.fits' )
    parser.add_argument( '--sdss_dir', dest='sdss_dir', type=str, help='directory of sdss cutouts', default='' )
    parser.add_argument( '--sdss_filters', dest='sdss_filters', type=str, help='the filters to use for the RGB channels', default='g,r,i' )
    parser.add_argument( '--size', dest='cutout_size', type=float, help='The size (degrees) of the cutout)', default=0.02777 )
    parser.add_argument( '--first', action="store_true" )
    parser.add_argument( '--inset', action="store_true" )
    parser.add_argument( '--beam', action="store_true" )
    parser.add_argument( '--axes', action="store_true" )
    parser.add_argument( '--clobber', action="store_true" )
    parser.add_argument( '--nup', dest='nup', type=str, default='4x6', help='pdfnup ncol x nrow' )
    parser.add_argument( '--get_sdss', action="store_true", help="do not run main, just get sdss file for download" )

    args = parser.parse_args()
    lofar_pattern = args.lofar_pattern
    sdss_dir = args.sdss_dir
    sdss_filters = args.sdss_filters
    cutout_size = args.cutout_size
    do_first = args.first
    do_inset = args.inset
    do_beam = args.beam
    do_axes = args.axes
    clobber = args.clobber
    nup = args.nup
    sdss_only = args.get_sdss

    if sdss_only:
        filelist = glob.glob( lofar_pattern )
        get_list_for_cutouts( filelist )
        print( 'Please upload positions_for_cutouts.dat to: https://dr12.sdss.org/bulkFields and then download the results (using wget)' )
        print( 'Then unsip using bzip2 -d file.bz2' )
    else:
        main( lofar_pattern=lofar_pattern, sdss_dir=sdss_dir, sdss_filters=sdss_filters, cutout_size=cutout_size, do_first=do_first, do_inset=do_inset, do_beam=do_beam, do_axes=do_axes, clobber=clobber, nup=nup )
