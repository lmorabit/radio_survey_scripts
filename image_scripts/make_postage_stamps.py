import pylab
import numpy as np
import pywcs
import os
import pyfits
import argparse
import astropy

def postage( fitsim, postfits, ra, dec, s, s2 ):

    ## get the image header and keywords
    head = pyfits.getheader( fitsim )
    hdulist = pyfits.open( fitsim )
    # Parse the WCS keywords in the primary HDU
#    wcs = astropy.wcs.WCS( hdulist[0].header )
    wcs = pywcs.WCS( hdulist[0].header )

    # Some pixel coordinates of interest.
    skycrd = np.array([ra,dec])
    skycrd = np.array([[ra,dec,0,0]], np.float_)

    # Convert pixel coordinates to world coordinates
    # The second argument is "origin" -- in this case we're declaring we
    # have 1-based (Fortran-like) coordinates.
    pixel = wcs.wcs_sky2pix(skycrd, 1)

    # Some pixel coordinates of interest.
    x = pixel[0][0]
    y = pixel[0][1]

    pixsize = abs(wcs.wcs.cdelt[0])
    #pixsize = 7.164120324887e-05
    if pylab.isnan(s):
        s = 25.
    N = (s/pixsize)
    if s2 != 0:
        N2 = (s2/pixsize)
    else:
	N2 = N

    print 'x=%.5f, y=%.5f, N=%i' %(x,y,N)

    ximgsize = head.get('NAXIS1')
    yimgsize = head.get('NAXIS2')

    if x ==0:
        x = ximgsize/2
    if y ==0:
        y = yimgsize/2

    offcentre = False
    # subimage limits: check if runs over edges
    xlim1 =  x - (N/2)
    if(xlim1<1):
        xlim1=1
        offcentre=True
    xlim2 =  x + (N/2)
    if(xlim2>ximgsize):
        xlim2=ximgsize
        offcentre=True
    ylim1 =  y - (N2/2)
    if(ylim1<1):
        ylim1=1
        offcentre=True
    ylim2 =  y + (N2/2)
    if(ylim2>yimgsize):
        offcentre=True
        ylim2=yimgsize

    xl = int(xlim1)
    yl = int(ylim1)
    xu = int(xlim2)
    yu = int(ylim2)
    print 'postage stamp is %i x %i pixels' %(xu-xl,yu-yl)

    # make fits cutout
    inps = fitsim + '[%0.0f:%0.0f,%0.0f:%0.0f]' %(xl,xu,yl,yu)

    if os.path.isfile(postfits): os.system('rm '+postfits)
    os.system( 'fitscopy %s %s' %(inps,postfits) )
    print  'fitscopy %s %s' %(inps,postfits) 

    return postfits

###################################################
## MAIN SCRIPT

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='fitsimage', type=str, help="fits image", required=True)
    parser.add_argument('-n', dest='ps_name', type=str, help="Name of object", required=True)
    parser.add_argument('--ra', dest='ps_ra', type=float, help="RA (degrees)", required=True)
    parser.add_argument('--dec', dest='ps_dec', type=float, help="Dec (degrees)", required=True)
    parser.add_argument('--size', dest='ps_size', type=float, default=0.3, help="Size of postage stamp (degrees)" )
    parser.add_argument('--size2', dest='ps_size2', type=float, default=0., help="Size of postage stamp in Dec if not square (degrees)" ) 

    args = parser.parse_args()
    fitsimage = args.fitsimage
    ps_name = args.ps_name
    ps_ra = args.ps_ra
    ps_dec = args.ps_dec
    ps_size = args.ps_size
    ps_size2 = args.ps_size2

    ## make a cutout fits image
    ps_fitsimage = ps_name + '.fits'
    postage( fitsimage, ps_fitsimage, ps_ra, ps_dec, ps_size, ps_size2 ) ## what is this 1.0???
    
    ## make a plot of the fits image
    #size =0.3
    #tmpimage = '%s.fits'%clusname
    #imagenoise = 5000E-6
    #aplpy_plot(clusra,clusdec,tmpimage,'%s.png'%clusname,imagenoise,size)
    #os.system('rm tmp.fits')
    

if __name__ == "__main__":
    main()

