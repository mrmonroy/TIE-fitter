"""
script to calculate w(theta) using treecorr, 
can do this using a pixel estimator, or a randoms points estimator (used in Y1 papers)
make sure you are using an up to date version of treecorr, some old versions won't use the SP weights.
"""
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import lsssys
import os
import healpy as hp 
import healpix_util as hu
import sys
import time

mocknumber = sys.argv[1]

def main():
	basedir = '/pool/cosmo01_data1/des/y6_lss/maglim/y6_maglim_final_selection/'
	
	nside = 512 #if using the pixel estimator
	
	
	maskdir = '/pool/cosmo01_data1/des/y6_sp_maps/outliers_analysis/maglim_mask/y6_fiducial_high_res_masks/y6_joint_mask/'
	maskfile = 'maglim_joint_lss-shear_mask_nside{0}_RING_v4.1.fits.gz'
	
	#################################
	input_order = 'RING'
	maskpixname = 'HPIX'
	fracpixname = 'FRAC_DET_GRIZ'
	
	#switch weights on/off. 
	do_weight = True
	do_noweight = False #Set to False if this was already calculated
	
	nthetabins = 20
	thetamin = 2.499999998622972/60.
	thetamax = 250./60.
	bin_slop = 0.01
	autoonly = True
	thin = None
	num_threads = 2 #set to the maximum number of cores available.
	
	if autoonly:
		corrlabel = 'autoonly'
	else:
		corrlabel = 'cross'
	
	num_maps = 19

	
	#############################################
	binedges = np.array([0.2, 0.4, 0.55, 0.7, 0.85, 0.95, 1.05])
	skip = []
	bins = list(np.arange(len(binedges)-1))
	finalbins = [zb for zb in bins if zb not in skip]
	if (finalbins==bins)==True:
		print('Not skipping any z-bin')
	else:
		print('Skipping z-bins = {0}'.format(skip))
	print('finalbins = ',finalbins)
	
	
	mockdir = os.path.join(basedir,'lognormal_mocks/enet_contaminated_mocks/v0.1/')
	mockfile = 'y6_maglim_conenetv0.1_nside1024_maskv4_mid_nside{NSIDE}_mock{MOCK_NUMBER}.fits.gz'.format(NSIDE=nside,MOCK_NUMBER=mocknumber)
	
	wmapdir = os.path.join(basedir,'mark_omega/std_maps/weights_validation/under-correction_bias/mock_wmaps_std{0}/'.format(num_maps))
	wmapfile = 'w_map_cmock{0}_nside{1}_mark_omega_std{2}.fits.gz'.format(mocknumber,nside,num_maps)
	
	
	outdir = os.path.join(basedir,'mark_omega/std_maps/weights_validation/under-correction_bias/mock_wtheta_std{0}/'.format(num_maps))
	if os.path.exists(outdir) == False:
		os.mkdir(outdir)
	print(outdir)
	
	
	##############################################################
	print('Loading degraded mask')
	mask = lsssys.mask(os.path.join(maskdir,maskfile.format(nside)),ZMAXcol=None,maskpixname=maskpixname,fracpixname=fracpixname,input_order=input_order,nside=nside)
	
	mock = lsssys.Mock(None, nside, empty=True)
	mock.load(os.path.join(mockdir,mockfile),nside=nside, loadfracdet=True)
	try:
		wmap = lsssys.Mock(None, nside, empty=True)
		wmap.load(os.path.join(wmapdir,wmapfile),nside=nside, loadfracdet=True)
	except IOError:
		print('Weight map not found: {0}'.format(wmapfile))
	
	if (mock.ngal[finalbins[0]].mask==mask.mask).all()==False:
		print('WARNING: The mock catalog does not match the mask. Masking it')
		for zb in bins:
			mock.ngal[zb].addmask(mask.mask,mask.fracdet)
	if (wmap.ngal[finalbins[0]].mask==mask.mask).all()==False:
		print('WARNING: The mock weight map does not match the mask. Masking it')
		for zb in bins:
			wmap.ngal[zb].addmask(mask.mask,mask.fracdet)
	
	#assert (mock.ngal[0].fracdet==wmap.ngal[0].fracdet).all()
	assert (mock.ngal[0].mask==wmap.ngal[0].mask).all()
	
	if do_weight == True:
		print('Calculating w(theta) for weighted data')
		w_pix_dict = {}
		for ibin in finalbins:
			
			mock_galmap1 = mock.ngal[ibin]#############################
			w_sys1 = wmap.ngal[ibin]
			
			mockgal1 = lsssys.Map()
			galval1 = np.ones(hp.nside2npix(nside))*hp.UNSEEN
			galval1[~mock_galmap1.mask] = mock_galmap1.data[~mock_galmap1.mask]*w_sys1.data[~mock_galmap1.mask]
			mockgal1.adddata(galval1,mock_galmap1.mask,mock_galmap1.fracdet)
			
			for jbin in finalbins:
				if jbin < ibin:
					continue
				if autoonly == True and ibin != jbin:
					continue
				print(ibin,jbin)
				
				mock_galmap2 = mock.ngal[jbin]##################################
				w_sys2 = wmap.ngal[jbin]
				
				mockgal2 = lsssys.Map()
				galval2 = np.ones(hp.nside2npix(nside))*hp.UNSEEN
				galval2[~mock_galmap2.mask] = mock_galmap2.data[~mock_galmap2.mask]*w_sys2.data[~mock_galmap2.mask]
				mockgal2.adddata(galval2,mock_galmap2.mask,mock_galmap2.fracdet)
				
				theta1, w_pix = lsssys.corr2pt(mockgal1,mockgal2, w1=None, w2=None, nthetabins=nthetabins,thetamax=thetamax,thetamin=thetamin, weights=1./mockgal1.fracdet, weights2=1./mockgal2.fracdet, bin_slop=bin_slop, num_threads = num_threads,fracweights=True,fracweights2=True)
				w_pix_dict['theta'] = theta1
				w_pix_dict[finalbins[ibin],finalbins[jbin]] = w_pix
		np.save(os.path.join(outdir,'w_pix_decontaminated_nside{nside}_mock{MOCK_NUMBER}.npy'.format(nside=nside,MOCK_NUMBER=mocknumber)), w_pix_dict)

	if do_noweight == True:
		print('Calculating w(theta) for non-weighted data')
		w_pix_noweight_dict = {}
		for ibin in finalbins:
			
			mock_galmap1 = mock.ngal[ibin]
			
			for jbin in finalbins:
				if jbin < ibin:
					continue
				if autoonly == True and ibin != jbin:
					continue
				
				print(ibin,jbin)
				
				mock_galmap2 = mock.ngal[jbin]
				
				theta1, w_pix_noweight = lsssys.corr2pt(mock_galmap1,mock_galmap2, w1=None, w2=None,nthetabins=nthetabins,thetamax=thetamax,thetamin=thetamin, weights=1./mock_galmap1.fracdet, weights2=1./mock_galmap2.fracdet, bin_slop=bin_slop, num_threads = num_threads,fracweights=True,fracweights2=True)
				w_pix_noweight_dict['theta'] = theta1
				w_pix_noweight_dict[ibin,jbin] = w_pix_noweight
		np.save(os.path.join(outdir,'w_pix_contaminated_nside{nside}_mock{MOCK_NUMBER}.npy'.format(nside=nside,MOCK_NUMBER=mocknumber)), w_pix_noweight_dict)

if __name__ == '__main__':
	t1 = time.time()
	main()
	'''
	for mocknum in mocks:
		main(mocknum)
	'''
	t2 = time.time()
	exetime = t2-t1
	print('Execution time (minutes): ',exetime/60.)


