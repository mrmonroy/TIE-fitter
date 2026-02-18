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
order = 'RING'
maskpixname = 'HPIX'
fracpixname = 'FRAC_DET_GRIZ'
sample_version = 'v3d2_em_JMv4'
mask_version = 'v4'

if order=='RING':
	do_ring = False
else:
	do_ring = True

#################average ngal_mean from mocks 
mockdir = os.path.join(basedir,'lognormal_mocks/enet_contaminated_mocks/v0.1/')
maskdir = '/pool/cosmo01_data1/des/y6_sp_maps/outliers_analysis/maglim_mask/y6_fiducial_high_res_masks/y6_joint_mask/'
mapdir = '/pool/cosmo01_data1/des/y6_sp_maps/official_v4/degraded_joint_mask/main_sp_maps/'


maskfile = 'maglim_joint_lss-shear_mask_nside{0}'+'_{0}_{1}.fits.gz'.format(order,mask_version)

####################
mocklist = [mock for mock in os.listdir(mockdir) if '.fits.gz' in mock and '_nosys_' not in mock and str(nside) in mock]

ngal_mean_mocks = {0:6.024978712384133,
	1:4.342514640893101,
	2:4.602832005072836,
	3:5.80304698265032,
	4:4.523950153625241,
	5:4.582774299022881}

nmocks = len(mocklist)
print('Number of mocks = ',nmocks)

nbins = 6
#ibin = 5

###########################This checks the z-bins numbering (starting from 0 or from 1)
mock0 = fio.FITS(os.path.join(mockdir,mocklist[0]))[1]
z_num = []
for mockcol in mock0.get_colnames():
	if 'bin' in mockcol:
		z_num.append(int(mockcol[3:]))
lowest_bin = np.min(z_num)

####################
maplist = [m for m in os.listdir(mapdir) if '{0}'.format(nside) in m]
maplist.sort()
num_maps = len(maplist)
print('Number of SP maps = ', len(maplist))


outdir = os.path.join(basedir,'mark_omega/std_maps/weights_validation/under-correction_bias/')
if os.path.exists(outdir)==False:
	os.mkdir(outdir)

#####################################################
mask = lsssys.Mask(os.path.join(maskdir,maskfile.format(nside)),ZMAXcol=None,input_order=order,do_ring=do_ring,nside=nside,maskpixname=maskpixname,fracpixname=fracpixname)
assert mask.nside==nside
assert (mask.fracdet[~mask.mask]!=0.).all()
assert (mask.fracdet[mask.mask]==0.).all()

fracdet = mask.fracdet
fracmask = fracdet>=0.5

#####################################################
sp_data_std = []
for imap in maplist:
	sysmap_ = lsssys.SysMap(os.path.join(mapdir,imap),systnside=nside)
	assert sysmap_.nside==mask.nside
	sysmap_.addmask(mask.mask,mask.fracdet)
	lsssys.mask_checks(sysmap_,mask)
	
	std_data_ = standardise_spdata(sysmap_,fracmask)
	sp_data_std.append(std_data_)

sp_data_std = np.transpose(np.array(sp_data_std))
assert sp_data_std.shape==(len(fracdet[fracmask]),len(maplist))


#####################################################
def worker(imock):
	global fracmask
	
	mock1 = fio.read(os.path.join(mockdir,mocklist[imock]))
	output_dict = {}
	for ibin in range(nbins):
		Ngal_mock = np.ones(hp.nside2npix(nside))*hp.UNSEEN
		Ngal_mock[mock1['HPIX']] = mock1['bin{0}'.format(ibin+lowest_bin)]
		Ngal_mock[~fracmask] = hp.UNSEEN
		'''
		fracdet_mock = np.zeros(hp.nside2npix(nside))
		fracdet_mock[mock1['HPIX']] = mock1['FRACGOOD']
		fracdet_mock[~fracmask] = 0.0
		
		galmap0 = lsssys.Map()
		galmap0.adddata(Ngal_mock, mask=Ngal_mock==hp.UNSEEN, fracdet=fracdet_mock)
		assert galmap0.nside==nside
		#lsssys.mask_checks(galmap0,mask)
		'''
		assert (Ngal_mock[fracmask]!=hp.UNSEEN).all()
		assert (Ngal_mock[~fracmask]==hp.UNSEEN).all()
		
		#Ngal = galmap0.data[fracmask]
		
		ngal = Ngal_mock[fracmask]/fracdet[fracmask]
		ngal_mean = np.average(ngal,weights=fracdet[fracmask])
		
		ngal_ratio = ngal/ngal_mean
		
	
		linreg = LinearRegression()
		linreg.fit(sp_data_std,ngal_ratio)
		
		output_dict[(imock,ibin)] = [linreg.coef_,linreg.intercept_]
		
	return output_dict

###############################################

if __name__=='__main__':
	coeff_dict = {}
	
	if test==False:
		blocks = np.arange(0,nmocks+1,10)
		mocks_list = []
		for i in range(len(blocks)-1):
			mocks_list.append(np.arange(blocks[i],blocks[i+1]))
		
		print('##################################################### ',mocks_list)

		for mocknum in mocks_list:
			print('Mock numbers = ',mocknum)
			pool = mp.Pool(len(mocknum))
			results = pool.map(worker,mocknum)
			print(results)
			print('Closing pool of workers')
			pool.close()
			print('Joining pool of workers')
			pool.join()
			t_wait=1
			print('Waiting ', t_wait,' seconds')
			time.sleep(t_wait)
			
			for elem_ in results:
				for key_ in elem_.keys():
					print(key_)
					assert (len(elem_[key_][0])==sp_data_std.shape[1])
					coeff_dict[key_] = elem_[key_]
	
	else:
		results = worker(0)
		
		for key_ in results.keys():
			#print(key_)
			assert (len(results[key_][0])==sp_data_std.shape[1])
			coeff_dict[key_] = results[key_]
		
	if test==False:
		np.save(os.path.join(outdir,'contaminated_mock_coeff_dict_nside{0}_std{1}.npy'.format(nside,len(maplist))),coeff_dict)
	else:
		np.save(os.path.join(outdir,'contaminated_mock_coeff_dict_nside{0}_std{1}_TEST.npy'.format(nside,len(maplist))),coeff_dict)
	





