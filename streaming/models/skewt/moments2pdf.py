import numpy as np
from scipy.special import gamma
from scipy.stats import t
from scipy.optimize import fsolve, curve_fit
from scipy.special import gamma

#TODO: Docs!!

def skewt_pdf(v, w, v_c, alpha, nu):
	rescaled_v = (v - v_c)/w

	cdf_arg = alpha * rescaled_v * ((nu + 1)/(rescaled_v**2 + nu))**0.5
	
	return 2./w * t.pdf(rescaled_v, scale = 1, df = nu) * t.cdf(cdf_arg, df = nu+1, scale = 1)

def moments2skewt(mean, std, gamma1, gamma2):

	st_parameters = moments2parameters(mean, std, gamma1, gamma2)

	def pdf_los(vlos, rperp, rparallel):

		return skewt_pdf(vlos, *st_parameters)

	return pdf_los




#********************** GIVEN MOMENTS COMPUTE ST PARAMETERS *******************************
def gamma1_constrain(alpha, dof, gamma1):
	b_dof = (dof/np.pi)**0.5 * gamma(0.5*(dof - 1))/gamma(0.5*dof)
	delta = alpha / np.sqrt(1+alpha**2)
	
	return gamma1 - delta*b_dof * ((dof * (3 - delta**2))/(dof-3) - 3*dof/(dof-2.)\
				+ 2*delta**2*b_dof**2) * (dof/(dof-2) - delta**2 * b_dof**2)**(-1.5)


def gamma2_constrain(alpha, dof, gamma2):
	b_dof = (dof/np.pi)**0.5 * gamma(0.5*(dof - 1))/gamma(0.5*dof)
	delta = alpha / np.sqrt(1+alpha**2)
	
	return gamma2 - ( 3 * dof**2 / ((dof-2) * (dof-4)) - \
			4 * delta**2*b_dof**2 *dof*(3-delta**2)/(dof-3)\
			+ 6 * delta**2 *b_dof**2*dof/(dof-2) -\
			3 * delta**4 * b_dof**4) * (dof/(dof-2.) - delta**2 * b_dof**2)**(-2.) + 3.
	

def constrains(x, gamma1, gamma2):
	alpha, nu = x
	return (gamma1_constrain(alpha, nu, gamma1), gamma2_constrain(alpha, nu, gamma2))

def moments2parameters(mean, std, gamma1, gamma2, p0 = (-0.7, 5)):
	
	alpha, nu =  fsolve(constrains, p0, args = (gamma1, gamma2))

	delta = alpha / np.sqrt(1+alpha**2)

	b = (nu/np.pi)**0.5 * gamma((nu-1)/2.)/gamma(nu/2.)
	
	w = std / np.sqrt( nu/(nu-2) - delta**2 * b**2)
	
	v_c = mean - w * delta * b
		
	return w, v_c, alpha, nu


#********************** GIVEN PARAMETERS COMPUTE ST MOMENTS *******************************
def mean(v_c, w, alpha, dof):
	b_dof = (dof/np.pi)**0.5 * gamma(0.5*(dof - 1))/gamma(0.5*dof)
	delta = alpha / np.sqrt(1+alpha**2)
	return v_c + w*delta*b_dof

def std(w, alpha, dof):
	b_dof = (dof/np.pi)**0.5 * gamma(0.5*(dof - 1))/gamma(0.5*dof)
	delta = alpha / np.sqrt(1+alpha**2)
	return np.sqrt(w**2  *(dof/(dof-2) - delta**2 * b_dof**2))

def gamma1(alpha, dof):
	b_dof = (dof/np.pi)**0.5 * gamma(0.5*(dof - 1))/gamma(0.5*dof)
	delta = alpha / np.sqrt(1+alpha**2)
	
	return delta*b_dof * ((dof * (3 - delta**2))/(dof-3) - 3*dof/(dof-2.)\
				+ 2*delta**2*b_dof**2) * (dof/(dof-2) - delta**2 * b_dof**2)**(-1.5)

def gamma2(alpha, dof):
	b_dof = (dof/np.pi)**0.5 * gamma(0.5*(dof - 1))/gamma(0.5*dof)
	delta = alpha / np.sqrt(1+alpha**2)
	
	value =  ( 3 * dof**2 / ((dof-2) * (dof-4)) - \
			4 * delta**2*b_dof**2 *dof*(3-delta**2)/(dof-3)\
			+ 6 * delta**2 *b_dof**2*dof/(dof-2) -\
			3 * delta**4 * b_dof**4) * (dof/(dof-2.) - delta**2 * b_dof**2)**(-2.) - 3.
	
	return value

def parameters2moments(w, v_c, alpha, dof):

	return mean(v_c, w, alpha, dof), std(w, alpha, dof), gamma1(alpha, dof), gamma2(alpha, dof)
