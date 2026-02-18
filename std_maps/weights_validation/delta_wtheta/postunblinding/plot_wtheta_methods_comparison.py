import numpy as np
import matplotlib.pyplot as plt
import os


basedir = '/pool/cosmo01_data1/des/y6_lss/maglim/y6_maglim_final_selection/'

sample_version = 'maglim-v3d2'

nside = 4096

threshold = 2
sigma1d = 'cubic'

num_maps = 19


##########################
binedges = np.array([0.2, 0.4, 0.55, 0.7, 0.85, 0.95, 1.05])
zbins = np.arange(6)

#################################
nthetabins = 20
thetamin = 2.499999998622972/60.
thetamax = 250./60.
bin_slop = 0.01


deltadir = os.path.join(basedir,'mark_omega/std_maps/weights_validation/postunblinding/')
deltafile = 'wtheta_dict_{0}_binslop{1}_nside{2}_methods_comparison_postunblinding.npy'.format(sample_version,bin_slop,nside)

Delta_wtheta = np.load(os.path.join(deltadir,deltafile),allow_pickle=True).ravel()[0] 


errorsdir = basedir
errorsfile = 'wtheta_covmat_v6.txt'

covmatrix = np.loadtxt(os.path.join(errorsdir,errorsfile))
error = np.sqrt(np.diagonal(covmatrix))

scale_cuts = {0:36.4020,
	1:26.5960,
	2:19.4850,
	3:16.2870,
	4:14.5150,
	5:13.3220,
	}

outdir = deltadir

configs = ['no_weights','isd_{0:.1f}{1}_sig'.format(threshold,sigma1d),'mark_omega_std{0}'.format(num_maps)]
color_dict = {'no_weights':'r','isd_{0:.1f}{1}_sig'.format(threshold,sigma1d):'orange','mark_omega_std{0}'.format(num_maps):'b'}

for ibin in zbins:
	print(ibin)
	
	theta_ = Delta_wtheta['{0}_theta'.format(ibin)]
	errorbar = error[ibin*nthetabins:(ibin+1)*nthetabins]
	
	fig = plt.figure(figsize=(10,7))
	ax = fig.add_subplot(111)
	for config in configs:
		wtheta_ = Delta_wtheta['{0}_{1}_wtheta'.format(ibin,config)]
		ax.plot(60.*theta_,60.*theta_*wtheta_,color=color_dict[config],label=config)
		if 'isd' in config:
			ax.errorbar(60.*theta_,60.*theta_*wtheta_,yerr=60.*theta_*errorbar,fmt='',color=color_dict[config],ecolor=color_dict[config])
	
	ax.axvspan(60.*theta_.min()*0.95, scale_cuts[ibin], color='grey', alpha=0.5)
	ax.set_xscale("log")
	ax.grid()
	ax.set_xlabel(r'$\theta \, \rm [arcmin]$',fontsize=16)
	ax.set_ylabel(r'\theta \cdot w(\theta)',fontsize=16)
	ax.legend(loc="best",fontsize=16)
	plt.savefig(os.path.join(outdir,'wtheta_methods_comparison_nside{0}_zbin{1}.png'.format(nside,ibin)))
	plt.close()





