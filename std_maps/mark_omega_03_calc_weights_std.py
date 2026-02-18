import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import lsssys
import os
import fitsio as fio
import time
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import multiprocessing as mp
import random

random.seed(66)

test = False
save_plots = True

def std(x,weights=None):
	if weights is None:
		weights = np.ones(len(x))
	
	var_ = np.sum(weights*(x-np.average(x,weights=weights))**2.)/(np.sum(weights)-1.)
	std_ = np.sqrt(var_)
	return std_

def plane(xx,yy,coeffs):
	a,b,c = coeffs
	return a*xx+b*yy+c

def standardise_spdata(sysmap,fracmask=None):
	if fracmask is None:
		sp_data = sysmap.data[~sysmap.mask]
		sp_fracdet = sysmap.fracdet[~sysmap.mask]
	else:
		sp_data = sysmap.data[fracmask]
		sp_fracdet = sysmap.fracdet[fracmask]
		
	mean_data = np.average(sp_data,weights=sp_fracdet)
	std_data = std(sp_data,weights=sp_fracdet)
	sp_data_std = (sp_data-mean_data)/std_data
	print(np.average(sp_data_std),np.std(sp_data_std))
	return sp_data_std


#################
basedir = '/pool/cosmo01_data1/des/y6_lss/maglim/y6_maglim_final_selection/'

####################
nside = 512
upnside = 4096
order = 'RING'
sample_version = 'v3d2_em_JMv4'
mask_version = 'v4'

nmocks = 1000

####################
maskdir = '/pool/cosmo01_data1/des/y6_sp_maps/outliers_analysis/maglim_mask/y6_fiducial_high_res_masks/y6_joint_mask/'
mapdir = '/pool/cosmo01_data1/des/y6_sp_maps/official_v4/degraded_joint_mask/main_sp_maps/'
coeffdir = os.path.join(basedir,'mark_omega/std_maps/')


maskfile = 'maglim_joint_lss-shear_mask_nside{0}'+'_{0}_{1}.fits.gz'.format(order,mask_version)


maskpixname = 'HPIX'
fracpixname = 'FRAC_DET_GRIZ'
print('Mask = ',maskfile)

if order=='RING':
	do_ring = False
else:
	do_ring = True

####################
maplist = [m for m in os.listdir(mapdir) if '{0}'.format(nside) in m]
maplist.sort()
num_maps = len(maplist)
#print(maplist)
print('Number of SP maps used = ', len(maplist))

coefffile_data = 'data_coeff_dict_nside{0}_std{1}.npy'.format(nside,num_maps)
coefffile_mock = 'mock_coeff_dict_nside{0}_std{1}.npy'.format(nside,num_maps)

outdir = os.path.join(basedir,'mark_omega/std_maps/wmaps_std{0}/'.format(num_maps))
if os.path.exists(outdir)==False:
	os.mkdir(outdir)

if save_plots:
	outdir_plots = os.path.join(outdir,'plots/')
	if os.path.exists(outdir_plots)==False:
		os.mkdir(outdir_plots)


####################
binedges = np.array([0.2, 0.4, 0.55, 0.7, 0.85, 0.95, 1.05])

zbins = np.arange(6)

#####################################################
mask = lsssys.Mask(os.path.join(maskdir,maskfile.format(nside)),ZMAXcol=None,input_order=order,do_ring=do_ring,nside=nside,maskpixname=maskpixname,fracpixname=fracpixname)
maskup = lsssys.Mask(os.path.join(maskdir,maskfile.format(upnside)),ZMAXcol=None,input_order=order,do_ring=do_ring,nside=upnside,maskpixname=maskpixname,fracpixname=fracpixname)

assert mask.nside==nside
assert (mask.fracdet[~mask.mask]!=0.).all()
assert (mask.fracdet[mask.mask]==0.).all()

assert maskup.nside==upnside
assert (maskup.fracdet[~maskup.mask]!=0.).all()
assert (maskup.fracdet[maskup.mask]==0.).all()


fracdet = mask.fracdet
fracmask = fracdet>=0.5

data_coeff_dict = np.load(os.path.join(coeffdir,coefffile_data),allow_pickle=True).ravel()[0]
mock_coeff_dict = np.load(os.path.join(coeffdir,coefffile_mock),allow_pickle=True).ravel()[0]


nbins = 100

pvalue_dict = {}
for ibin in zbins:
	print(ibin)
	
	data_coeffs = np.array(data_coeff_dict[ibin][0])
	
	mock_coeffs = []
	for imock in range(nmocks):
		mock_coeffs.append(mock_coeff_dict[(imock,ibin)][0])
	mock_coeffs = np.array(mock_coeffs)
	
	
	fs_ibin = np.ones(hp.nside2npix(upnside))*hp.UNSEEN
	fs_ibin[~maskup.mask] = 0.
	
	for i,imap in enumerate(maplist):
		data_coeff_ = data_coeffs[i]
		
		mock_coeffs_ = mock_coeffs[:,i]
		
		ns_,bins_,_ = plt.hist(mock_coeffs_,bins=nbins,density=True,color='cyan',alpha=0.7)
		
		bin_centers_ = (bins_[1:]+bins_[:-1])/2.
		#print(bin_centers_)
		dbin = np.average(np.diff(bin_centers_))
		
		pmask = (bin_centers_>-1.*np.abs(data_coeff_))*(bin_centers_<np.abs(data_coeff_))
		assert np.round((np.sum(ns_[pmask])+np.sum(ns_[~pmask]))*dbin,6)==1.0
		
		pval_ = np.sum(ns_[~pmask])*dbin
		pvalue_dict[(ibin,imap)] = pval_
		
		if save_plots:
			plt.axvline(x=data_coeff_,ls='--',color='b',label='a data')
			plt.axvline(x=-data_coeff_,ls='--',color='r',label='-a data')
			plt.plot([],[],label='p-value = {0:.4f}'.format(pval_))
			plt.grid()
			plt.xlabel(r'$a_{{{0}}}$'.format(i))
			plt.ylabel('Density')
			plt.title(imap.replace('.fits.gz',''))
			plt.legend(loc="best")
			plt.savefig(os.path.join(outdir_plots,'dist_a{0}_{1}_zbin{2}.png'.format(i,imap.replace('.fits.gz',''),ibin)))
			#plt.show()
			plt.close()
		
		sysmapup_ = lsssys.SysMap(os.path.join(mapdir,imap.replace(str(nside),str(upnside))),systnside=upnside)
		assert sysmapup_.nside==upnside
		
		sysmapup_.addmask(maskup.mask,maskup.fracdet)
		lsssys.mask_checks(sysmapup_,maskup)
		std_data_up_ = standardise_spdata(sysmapup_)
		
		#fs_ibin[~maskup.mask] = fs_ibin[~maskup.mask]+data_coeff_*(1.-pval_)*sysmapup_.data[~maskup.mask]
		fs_ibin[~maskup.mask] = fs_ibin[~maskup.mask]+data_coeff_*(1.-pval_)*std_data_up_
		print(imap,data_coeff_,(1.-pval_))
		print('Coeff. a_{0} = {1}'.format(i,data_coeff_))
		'''
		plt.hist(data_coeff_*(1.-pval_)*std_data_up_,bins=100)
		plt.yscale("log")
		plt.xlabel(imap)
		plt.show()
		plt.close()
		'''
	
	data_coeff0 = data_coeff_dict[ibin][1]
	print(data_coeff0)
	mock_coeffs0 = []
	for imock in range(nmocks):
		mock_coeffs0.append(mock_coeff_dict[(imock,ibin)][1])
	mock_coeffs0 = np.array(mock_coeffs0)
	#print(mock_coeffs0)
	
	if save_plots:
		ns_,bins_,_ = plt.hist(mock_coeffs0,bins=nbins,density=True,color='cyan',alpha=0.7)
		
		plt.axvline(x=data_coeff0,ls='--',color='b',label='b data')
		#plt.axvline(x=-data_coeff0,ls='--',color='r')
		plt.grid()
		plt.xlabel('b')
		plt.ylabel('Density')
		plt.title('Intercept')
		plt.legend(loc="best")
		plt.savefig(os.path.join(outdir_plots,'dist_b_zbin{0}.png'.format(ibin)))
		#plt.show()
		plt.close()
	
	print('Coeff b = ',data_coeff0)
	fs_ibin[~maskup.mask] = fs_ibin[~maskup.mask]+data_coeff0
	
	w_ibin = np.ones(hp.nside2npix(upnside))*hp.UNSEEN
	
	w_ibin[~maskup.mask] = 1./fs_ibin[~maskup.mask]
	w_mean = np.average(w_ibin[~maskup.mask])
	w_ibin[~maskup.mask] = w_ibin[~maskup.mask]/w_mean
	
	wsysmap = lsssys.Map()
	wsysmap.adddata(w_ibin,maskup.mask,(~maskup.mask).astype('float'))
	wsysmap.save(os.path.join(outdir,'w_map_bin{0}_nside{1}_mark_omega_std{2}.fits'.format(ibin,upnside,num_maps)),clobber=True)
	
	np.save(os.path.join(outdir,'pvalues_bin{0}_mark_omega_std{1}.npy'.format(ibin,num_maps)),pvalue_dict)
	
	if save_plots:
		_ = plt.hist(w_ibin[~maskup.mask],bins=100)
		plt.yscale("log")
		plt.grid()
		plt.xlabel('w')
		plt.ylabel('Number of pixels (nside = {0})'.format(upnside))
		plt.title('z-bin = {0}'.format(ibin+1))
		plt.savefig(os.path.join(outdir_plots,'w_map_bin{0}_nside{1}_mark_omega_std{2}.png'.format(ibin,upnside,num_maps)))
		#plt.show()
		plt.close()
	print(pvalue_dict)
	print('##########################################################################')


np.save(os.path.join(outdir,'pvalues_mark_omega_std{0}.npy'.format(num_maps)),pvalue_dict)

