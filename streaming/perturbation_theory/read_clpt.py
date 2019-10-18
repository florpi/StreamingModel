import numpy as np

def perturbation_prediction(simulation, F1, F2, f):

	datadir = '/cosma/home/dp004/dc-cues1/CLPT_GSRSD/data/'

	xifile = 'xi_' + str(simulation) + '.txt'
	v12file = 'v12_' + str(simulation) + '.txt'
	s12file = 's12_' + str(simulation) + '.txt'

	xiCLPT = np.loadtxt(datadir + xifile)
	v12CLPT = np.loadtxt(datadir + v12file)
	s12CLPT = np.loadtxt(datadir + s12file)

	r = xiCLPT[:,0]
	xi_CLPT = xiCLPT[:,2] + F1 * xiCLPT[:,3] + F2 * xiCLPT[:,4] + (F1**2)*xiCLPT[:,5] + F1*F2 * xiCLPT[:,6] + (F2**2)*xiCLPT[:,7]

	v12_CLPT = f*( v12CLPT[:,2] + F1*v12CLPT[:,3] + F2*v12CLPT[:,4] + (F1**2)*v12CLPT[:,5] + F1*F2*v12CLPT[:,6] )

	s12_par_CLPT  = f**2*( s12CLPT[:,1] + F1*s12CLPT[:,2] + F2*s12CLPT[:,3] + (F1**2)*s12CLPT[:,4] )

	s12_perp_CLPT = f**2*0.5*( s12CLPT[:,5] + F1*s12CLPT[:,6] + F2*s12CLPT[:,7] + (F1**2)*s12CLPT[:,8] )

	return r, xi_CLPT, v12_CLPT, s12_par_CLPT, s12_perp_CLPT

def r_binning():

	datadir = '/cosma/home/dp004/dc-cues1/CLPT_GSRSD/data/'

	xifile = 'xi.txt'

	xiCLPT = np.loadtxt(datadir + xifile)

	r = xiCLPT[:,0]
	return r 



def tpcf_prediction(F1, F2, f):

	datadir = '/cosma/home/dp004/dc-cues1/CLPT_GSRSD/data/'

	xifile = 'xi.txt'
	v12file = 'v12.txt'
	s12file = 's12.txt'

	xiCLPT = np.loadtxt(datadir + xifile)

	xi_CLPT = xiCLPT[:,2] + F1 * xiCLPT[:,3] + F2 * xiCLPT[:,4] + (F1**2)*xiCLPT[:,5] + F1*F2 * xiCLPT[:,6] + (F2**2)*xiCLPT[:,7]

	return xi_CLPT 


def weighted_perturbation_prediction(F1, F2, f):

	datadir = '/cosma/home/dp004/dc-cues1/CLPT_GSRSD/data/'

	xifile = 'xi.txt'
	v12file = 'v12.txt'
	s12file = 's12.txt'

	xiCLPT = np.loadtxt(datadir + xifile)
	v12CLPT = np.loadtxt(datadir + v12file)
	s12CLPT = np.loadtxt(datadir + s12file)

	r = xiCLPT[:,0]
	xi_CLPT = xiCLPT[:,2] + F1 * xiCLPT[:,3] + F2 * xiCLPT[:,4] + (F1**2)*xiCLPT[:,5] + F1*F2 * xiCLPT[:,6] + (F2**2)*xiCLPT[:,7]

	v12_CLPT = f*( v12CLPT[:,2] + F1*v12CLPT[:,3] + F2*v12CLPT[:,4] + (F1**2)*v12CLPT[:,5] + F1*F2*v12CLPT[:,6] ) /(1.+ xi_CLPT)

	s12_par_CLPT  = f**2*( s12CLPT[:,1] + F1*s12CLPT[:,2] + F2*s12CLPT[:,3] + (F1**2)*s12CLPT[:,4] ) / (1.+xi_CLPT)

	s12_perp_CLPT = f**2*0.5*( s12CLPT[:,5] + F1*s12CLPT[:,6] + F2*s12CLPT[:,7] + (F1**2)*s12CLPT[:,8] ) / (1.+ xi_CLPT)

	return r, xi_CLPT, v12_CLPT, s12_par_CLPT, s12_perp_CLPT


def weighted_mean_prediction(F1, F2, f):
	datadir = '/cosma/home/dp004/dc-cues1/CLPT_GSRSD/data/'

	xifile = 'xi.txt'
	v12file = 'v12.txt'
	s12file = 's12.txt'

	xiCLPT = np.loadtxt(datadir + xifile)
	v12CLPT = np.loadtxt(datadir + v12file)

	r = xiCLPT[:,0]
	xi_CLPT = xiCLPT[:,2] + F1 * xiCLPT[:,3] + F2 * xiCLPT[:,4] + (F1**2)*xiCLPT[:,5] + F1*F2 * xiCLPT[:,6] + (F2**2)*xiCLPT[:,7]
	v12_CLPT = f*( v12CLPT[:,2] + F1*v12CLPT[:,3] + F2*v12CLPT[:,4] + (F1**2)*v12CLPT[:,5] + F1*F2*v12CLPT[:,6] ) /(1.+ xi_CLPT)

	return v12_CLPT


