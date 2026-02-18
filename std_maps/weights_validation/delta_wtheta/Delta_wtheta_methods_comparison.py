import numpy as np 
import matplotlib
matplotlib.use('Agg')
import lsssys 
import time
import os
import healpix_util as hu
import scipy.special
import sys
#import misc
import multiprocessing as mp
import errno
import astropy.io.fits as pf
import matplotlib.pyplot as plt

basedir = '/pool/cosmo01_data1/des/y6_lss/maglim/y6_maglim_final_selection/'

sample_version = 'maglim-v3d2'

nside = 4096
equalarea_bins = False
nbins1d = 10
threshold = 2
sigma1d = 'cubic'
weights_version = '1.1'

if equalarea_bins:
        label_1dbins = 'equalarea'
else:
        label_1dbins = 'equalwidth'

#################################
num_maps = 19

#################################
maskpixname = 'HPIX'
fracpixname = 'FRAC_DET_GRIZ'
mask_version = 4.1


#################################
zcol = 'Z_MEAN'
errzcol = 'Z_SIGMA'
magcol = 'BDF_MAG_CORRDERED_I'
idcol = 'COADD_OBJECT_ID'


##########################
maskdir = '/pool/cosmo01_data1/des/y6_sp_maps/outliers_analysis/maglim_mask/y6_fiducial_high_res_masks/y6_joint_mask/'
maskfile = 'maglim_joint_lss-shear_mask_nside{0}_RING_v{1}.fits.gz'.format(nside,mask_version)

configs = ['no_weights','isd_{0:.1f}{1}_sig'.format(threshold,sigma1d),'mark_omega_std{0}'.format(num_maps)]
catpath_dict = {
	'no_weights':os.path.join(basedir,'isd/weighted_sample_{0}_nbins1d_{1}_{2:.1f}{3}_sig_v{4}'.format(label_1dbins,nbins1d,threshold,sigma1d,weights_version)),
	'isd_{0:.1f}{1}_sig'.format(threshold,sigma1d):os.path.join(basedir,'isd/weighted_sample_{0}_nbins1d_{1}_{2:.1f}{3}_sig_v{4}'.format(label_1dbins,nbins1d,threshold,sigma1d,weights_version)),
	'mark_omega_std{0}'.format(num_maps):os.path.join(basedir,'mark_omega/std_maps/weighted_sample_mark_omega_std{0}/'.format(num_maps)),
	}


catfile_dict = {
	'no_weights':'y6_maglim_v3d2_em_JMv4_isd_weighted_{0}_nbins1d_{1}_{2:.1f}{3}_sig_v{4}.fits.gz'.format(label_1dbins,nbins1d,threshold,sigma1d,weights_version),
	'isd_{0:.1f}{1}_sig'.format(threshold,sigma1d):'y6_maglim_v3d2_em_JMv4_isd_weighted_{0}_nbins1d_{1}_{2:.1f}{3}_sig_v{4}.fits.gz'.format(label_1dbins,nbins1d,threshold,sigma1d,weights_version),
	'mark_omega_std{0}'.format(num_maps):'y6_maglim_v3d2_em_JMv4_weighted_mark_omega_std_{0}.fits.gz'.format(num_maps),
	}


##########################
binedges = np.array([0.2, 0.4, 0.55, 0.7, 0.85, 0.95, 1.05])
zbins = np.arange(6)

#################################
nthetabins = 20
thetamin = 2.499999998622972/60.
thetamax = 250./60.
bin_slop = 0.01
num_threads = 20


############################
outdir = os.path.join(basedir,'mark_omega/std_maps/weights_validation/')
if os.path.exists(outdir)==False:
	os.mkdir(outdir)

mask = lsssys.mask(os.path.join(maskdir,maskfile),ZMAXcol=None,maskpixname=maskpixname,fracpixname=fracpixname,input_order='RING',nside=nside)

wtheta0_dict = {}
wthetaw_dict = {}
for config in configs:
	print(config)
	
	cat = lsssys.Y3Cat(os.path.join(catpath_dict[config],catfile_dict[config]),sample='maglimy6',ext=1, zlum=None,extracols=None, model_mag=False, load_z_mc=False, v = True, zcol=zcol, errzcol=errzcol, magcol=magcol,idcol=idcol)
	
	for ibin in range(len(binedges)-1):
		print(ibin)
		
		zmin = binedges[ibin]
		zmax = binedges[ibin+1]
		ra,dec,amask = cat.eqinbin(zmin,zmax,returnmask=True)
		
		if config=='no_weights':
			weight = None
		else:
			weight = cat.weight[amask]
		galmap = lsssys.cat2galmap(ra, dec, mask, weight=weight, nside=nside, minfrac=0.0, rescale=False)
		
		################
		theta, wtheta_pix = lsssys.corr2pt(galmap,galmap,w1=None,w2=None,nthetabins=nthetabins,thetamax=thetamax,thetamin=thetamin, scale1=1./galmap.fracdet, weights=None, scale2=1./galmap.fracdet, weights2=None, bin_slop=bin_slop, fracweights=True,fracweights2=True, num_threads = num_threads)
		
		if config=='no_weights':
			wtheta0_dict['{0}_theta'.format(ibin)] = theta
			wtheta0_dict['{0}_wtheta'.format(ibin)] = wtheta_pix
			continue
		else:
			assert (theta==wtheta0_dict['{0}_theta'.format(ibin)]).all()
			wthetaw_dict['{0}_{1}_wtheta'.format(ibin,config)] = wtheta_pix
	
Delta_dict = {}
for ibin in zbins:
	Delta_dict['{0}_theta'.format(ibin)] = wtheta0_dict['{0}_theta'.format(ibin)]
	for config in configs[1:]:
		Delta_dict['{0}_{1}_wtheta'.format(ibin,config)] = wtheta0_dict['{0}_wtheta'.format(ibin)]-wthetaw_dict['{0}_{1}_wtheta'.format(ibin,config)]

	
np.save(os.path.join(outdir,'Delta_wtheta_dict_{0}_binslop{1}_nside{2}_methods_comparison.npy'.format(sample_version,bin_slop,nside)),Delta_dict)


