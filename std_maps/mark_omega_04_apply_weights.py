import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lsssys
import os
import healpy as hp
import pdb
import healpix_util as hu
import scipy.special
import scipy.optimize
import sys
import time
import multiprocessing as mp
import random


random.seed(66)


basedir = '/pool/cosmo01_data1/des/y6_lss/maglim/y6_maglim_final_selection/'

nside = 4096
num_maps = 19

catdir = basedir
maskdir = '/pool/cosmo01_data1/des/y6_sp_maps/outliers_analysis/maglim_mask/y6_fiducial_high_res_masks/y6_joint_mask/'
wdir = os.path.join(basedir,'mark_omega/std_maps/wmaps_std{0}/'.format(num_maps))

catlabel = 'y6_maglim_v3d2_em_JMv4'
catfile = catlabel+'.fits.gz'
maskfile = 'maglim_joint_lss-shear_mask_nside{0}_{1}_v4.fits.gz'
maskfile2 = 'maglim_joint_lss-shear_mask_nside{0}_{1}_v4.1.fits.gz'
wfile = 'w_map_bin{0}'+'_nside{0}_mark_omega_std{1}.fits.gz'.format(nside,num_maps)


outdir = os.path.join(basedir,'mark_omega/std_maps/weighted_sample_mark_omega_std{0}/'.format(num_maps))
if os.path.exists(outdir)==False:
	os.mkdir(outdir)

catfile_out = catlabel+'_weighted_mark_omega_std_{0}.fits'.format(num_maps)

##############################################
maskpixname = 'HPIX'
fracpixname = 'FRAC_DET_GRIZ'
input_order = 'RING'


##############################################
#make sure these are the same as your input bin edges
binedges = [0.2, 0.4, 0.55, 0.7, 0.85, 0.95, 1.05]
zbins = np.arange(6)

##############################################
cat = lsssys.Y3Cat(os.path.join(catdir,catfile), sample='maglimy6',ext=1, zlum=None,extracols=None, model_mag=False, load_z_mc=True, v = True, zcol='Z_MEAN', errzcol='Z_SIGMA', zmccol='Z_MC', magcol='BDF_MAG_CORRDERED_I',idcol='COADD_OBJECT_ID')
mask = lsssys.mask(os.path.join(maskdir,maskfile.format(nside,input_order)), ZMAXcol = None, maskpixname=maskpixname,fracpixname=fracpixname,input_order=input_order,nside=nside)
mask2 = lsssys.mask(os.path.join(maskdir,maskfile2.format(nside,input_order)), ZMAXcol = None, maskpixname=maskpixname,fracpixname=fracpixname,input_order=input_order,nside=nside)


##########################################################
clobber = True

def main():
	
	hpix = hu.HealPix('ring',nside)
	cat_pix1 = hpix.eq2pix(cat.ra,cat.dec)
	if (mask.mask[cat_pix1]==False).all()==False:
		print('WARNING: The catalog does not match the mask. Removing coordinates that are not contained in the mask')
		goodpixels = (mask.mask[cat_pix1]==False)
		cat.cut(goodpixels)
	
	
	for ibin in zbins:
		print(ibin)
		
		w_sys = np.ones(len(cat.ra))
		
		ra,dec,amask = cat.eqinbin(binedges[ibin],binedges[ibin+1],returnmask=True)
		cat_pix_ = hpix.eq2pix(ra,dec)
		
		wmap_ = lsssys.SysMap(os.path.join(wdir,wfile.format(ibin)),systnside=nside)
		assert wmap_.nside==nside
		assert (wmap_.mask==mask.mask).all()
		
		w_ = wmap_.data[cat_pix_]
		w_sys[amask] = w_sys[amask]*w_
		print('Mean before = ', np.mean(w_sys[amask]))
		w_sys[amask] = w_sys[amask]/np.mean(w_sys[amask])
		print('Mean after = ', np.mean(w_sys[amask]))
		cat.weight = cat.weight*w_sys
	
	cat_pix2 = hpix.eq2pix(cat.ra,cat.dec)
	if (mask2.mask[cat_pix2]==False).all()==False:
		print('WARNING: The catalog does not match the second mask. Removing coordinates that are not contained in the mask')
		goodpixels2 = (mask2.mask[cat_pix2]==False)
		cat.cut(goodpixels2)
	
	cat.savecat(os.path.join(outdir,catfile_out), clobber=clobber)

if __name__=='__main__':
	main()

