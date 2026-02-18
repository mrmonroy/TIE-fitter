import numpy as np
import matplotlib.pylab as plt
import pylab
import lsssys
import os

basedir = '/pool/cosmo01_data1/des/y6_lss/maglim/y6_maglim_final_selection/'

sample = 'maglim_v3d2'
nside = 512
autoonly = True
bin_slop = 0.01

if autoonly:
	corrlabel = 'autoonly'
else:
	corrlabel = 'cross'
num_maps = 19

wthetadir0 = os.path.join(basedir,'isd/weights_validation/wtheta_binslop{0}_nside{1}_treecorr336_{2}_pixel_uncontaminated/'.format(bin_slop,nside,corrlabel))
wthetadir = os.path.join(basedir,'mark_omega/std_maps/weights_validation/over-correction_bias/mock_wtheta_std{0}/'.format(num_maps))
outdir = os.path.join(basedir,'mark_omega/std_maps/weights_validation/over-correction_bias/plots_wtheta_std{0}/'.format(num_maps))
if os.path.exists(outdir)==False:
	os.mkdir(outdir)

wlabel_mock0 = 'w_pix_uncontaminated_nside{nside}_mock{MOCK_NUMBER}.npy'
wlabel_wmock = 'w_pix_weights_uncontaminated_nside{nside}_mock{MOCK_NUMBER}.npy'

zbins = np.arange(6)
#zbins = [0,1,2,3]

nmocks = 1000

w_ref = np.load(os.path.join(wthetadir0,wlabel_mock0.format(nside=nside,MOCK_NUMBER=0)),allow_pickle=True).ravel()[0]

scale_cuts = {0:0.,1:0.,2:0.,3:0.,4:0.,5:0.}

for ibin in zbins:
	wtheta_mock0_list = []
	wtheta_wmock_list = []
	
	for i in range(1000):
		w_mock0 = np.load(os.path.join(wthetadir0,wlabel_mock0.format(nside=nside,MOCK_NUMBER=i)),allow_pickle=True).ravel()[0]
		assert (w_mock0['theta']==w_ref['theta']).all()
		wtheta_mock0_list.append(w_mock0[ibin,ibin])
	
	for i in range(nmocks):
		#w_mock0 = np.load(path0+wlabel_mock0.format(MOCK_NUMBER=i)).ravel()[0]
		w_wmock = np.load(os.path.join(wthetadir,wlabel_wmock.format(nside=nside,MOCK_NUMBER=i)),allow_pickle=True).ravel()[0]
		#assert (w_mock0['theta']==w_ref['theta']).all()
		assert (w_wmock['theta']==w_ref['theta']).all()
		#wtheta_mock0.append(w_mock0[ibin,ibin])
		wtheta_wmock_list.append(w_wmock[ibin,ibin])
	
	wmean_mock0 = np.mean(wtheta_mock0_list,axis=0)
	#print wmean_mock0
	wmean_wmock = np.mean(wtheta_wmock_list,axis=0)
	
	w_covmat = lsssys.covariance(wtheta_mock0_list,v=True)
	errbar = np.sqrt(np.diagonal(w_covmat))
	#print errbar
	
	theta = w_ref['theta']
	
	######
	plt.figure(figsize=(11.5,8))
	plt.tight_layout()
	
	for i in range(nmocks):
		plt.plot(theta,theta*wtheta_wmock_list[i],'b-',alpha=10./nmocks)
	
	plt.plot(theta,theta*wmean_wmock,'-',color='b',label='Mark Omega Decont. mock, std{0}'.format(num_maps))
	plt.errorbar(theta,theta*wmean_mock0,yerr=theta*errbar,color='k',ecolor='k',label='Uncont. Mock')
	plt.axvline(x=scale_cuts[ibin]/60.)
	
	plt.xlabel(r'$\theta \, [deg]$',fontsize=15)
	plt.xticks(fontsize=15)
	plt.ylabel(r'$\theta \cdot w(\theta)$',fontsize=15)
	plt.yticks(fontsize=15)
	plt.title('{0}, z-bin {1}'.format(sample,ibin+1),fontsize=15)
	plt.grid()
	plt.legend(loc="best",fontsize=12)
	figname = 'wtheta_test3_zbin{0}_mark_omega_std{1}_nmocks{2}.png'.format(ibin,num_maps,nmocks)
	plt.savefig(os.path.join(outdir,figname))
	#plt.show()
	plt.close()

