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
sample_version = 'v3d2_em_JMv4'
mask_version = 'v4'

####################
ngaldir = os.path.join(basedir,'ngal_maps/')
maskdir = '/pool/cosmo01_data1/des/y6_sp_maps/outliers_analysis/maglim_mask/y6_fiducial_high_res_masks/y6_joint_mask/'
mapdir = '/pool/cosmo01_data1/des/y6_sp_maps/official_v4/degraded_joint_mask/main_sp_maps/'
outdir = os.path.join(basedir,'mark_omega/std_maps/')
if os.path.exists(outdir)==False:
	os.mkdir(outdir)

ngalfile = 'y6_maglim_{0}'.format(sample_version)+'_ngal_map_nside{0}_zbin{1}.fits.gz'
maskfile = 'maglim_joint_lss-shear_mask_nside{0}'+'_{0}_{1}.fits.gz'.format(order,mask_version)

maskpixname = 'HPIX'
fracpixname = 'FRAC_DET_GRIZ'
print('Mask = ',maskfile)

if order=='RING':
	do_ring = False
else:
	do_ring = True


maplist = [m for m in os.listdir(mapdir) if '{0}'.format(nside) in m]
maplist.sort()
print(maplist)
print('Number of SP maps = ', len(maplist))

binedges = np.array([0.2, 0.4, 0.55, 0.7, 0.85, 0.95, 1.05])

zbins = np.arange(6)

#####################################################
mask = lsssys.Mask(os.path.join(maskdir,maskfile.format(nside)),ZMAXcol=None,input_order=order,do_ring=do_ring,nside=nside,maskpixname=maskpixname,fracpixname=fracpixname)
assert mask.nside==nside
assert (mask.fracdet[~mask.mask]!=0.).all()
assert (mask.fracdet[mask.mask]==0.).all()

fracdet = mask.fracdet
fracmask = fracdet>=0.5


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


def worker(ibin):
	global fracmask
	galmap0 = lsssys.GalMap(os.path.join(ngaldir,ngalfile.format(nside,ibin)),nside=nside)
	assert galmap0.nside==mask.nside
	lsssys.mask_checks(galmap0,mask)
	
	Ngal = galmap0.data[fracmask]
	ngal = Ngal/fracdet[fracmask]
	ngal_mean = np.average(ngal,weights=fracdet[fracmask])
	
	ngal_ratio = ngal/ngal_mean
	
	output_dict = {}
	###############################################
	linreg = LinearRegression()
	linreg.fit(sp_data_std,ngal_ratio)
	
	print('Linear regression coefficients: ', linreg.coef_)
	print('Linear regression intercept: ', linreg.intercept_)
	
	output_dict[ibin] = [linreg.coef_,linreg.intercept_]
	
	return output_dict

###############################################

if __name__=='__main__':
	coeff_dict = {}
	
	if test==False:
		pool = mp.Pool(len(zbins))
		results = pool.map(worker,zbins)
		print(results)
		print('Closing pool of workers')
		pool.close()
		print('Joining pool of workers')
		pool.join()
		t_wait=1
		print('Waiting ', t_wait,' seconds')
		time.sleep(t_wait)
		
	else:
		results = worker(0)
		
	for elem_ in results:
		for key_ in elem_.keys():
			print(key_)
			assert (len(elem_[key_][0])==sp_data_std.shape[1])
			coeff_dict[key_] = elem_[key_]
	
	if test==False:
		np.save(os.path.join(outdir,'data_coeff_dict_nside{0}_std{1}.npy'.format(nside,len(maplist))),coeff_dict)
	else:
		np.save(os.path.join(outdir,'data_coeff_dict_nside{0}_std{1}_TEST.npy'.format(nside,len(maplist))),coeff_dict)
		



