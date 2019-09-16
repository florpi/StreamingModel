import numpy as np
from scipy.special import gamma
from scipy.stats import t
from scipy.optimize import fsolve, curve_fit
from scipy.special import gamma
from scipy.interpolate import interp2d

#TODO: Docs!!

def skewt_pdf(v, w, v_c, alpha, nu):
	''' Probability Density Function of a Skewed-Student-t distribution in one dimension.

	Args: 
		v: random variable.
		w: scale parameter.
		v_c: location parameter.
		alpha: skewness parameter.
		nu: degrees of freedom.

	Returns:
		Skewt PDF evaluated at v

	'''
	rescaled_v = (v - v_c)/w

	cdf_arg = alpha * rescaled_v * ((nu + 1)/(rescaled_v**2 + nu))**0.5
	
	return 2./w * t.pdf(rescaled_v, scale = 1, df = nu) * t.cdf(cdf_arg, df = nu+1, scale = 1)

def init_parameter_grid(sim_measurement):

	st_parameters = np.zeros(
			(sim_measurement.r_perpendicular.shape[0],
			sim_measurement.r_parallel.shape[0], 
			4)
			)

	mean, std, gamma1, gamma2 = sim_measurement.get_los_moments_from_rt()

	for i, rperp in enumerate(sim_measurement.r_perpendicular):
		for j, rparallel in enumerate(sim_measurement.r_parallel):

			st_parameters[i,j,:] = moments2parameters(
										mean[i,j], std[i,j], gamma1[i,j], gamma2[i,j]
										)
	print('Found ST parameters from moments')

	return st_parameters

def interpolate_parameters(sim_measurement, st_parameters):

	interpolators = []

	for parameter in range(st_parameters.shape[-1]):

		interpolators.append(
				interp2d(sim_measurement.r_parallel, sim_measurement.r_perpendicular,
					st_parameters[..., parameter], kind = 'linear') 
				)
	print(len(interpolators))

	return interpolators
				


def moments2skewt(sim_measurement):


	# Getting the parameters at every rperp and rparallel distance when integrating is too expensive,
	# create a grid and then interpolate

	st_parameters = init_parameter_grid(sim_measurement)

	w, v_c, alpha, nu = interpolate_parameters(sim_measurement, st_parameters)

	def pdf_los(vlos, rperp, rparallel):

		#TODO: Fix interpolation

		# take all combinations of rperp and rparallel
		#rperp = rperp.flatten()
		#rparallel = rparallel.flatten()


		#return skewt_pdf(vlos, w(rparallel, rperp), v_c(rparallel, rperp), alpha(rparallel, rperp), nu(rparallel, rperp))

		r_perp_bins = np.digitize(rperp, sim_measurement.r_perpendicular - 0.5 * (sim_measurement.r_perpendicular[1] - \
				sim_measurement.r_perpendicular[0])) - 1
		r_parallel_bins = np.digitize(rparallel, sim_measurement.r_parallel - 0.5 * (sim_measurement.r_parallel[1] - \
				sim_measurement.r_parallel[0])) - 1

		w = st_parameters[r_perp_bins, r_parallel_bins, 0]
		v_c = st_parameters[r_perp_bins, r_parallel_bins, 1]
		alpha = st_parameters[r_perp_bins, r_parallel_bins, 2]
		nu = st_parameters[r_perp_bins, r_parallel_bins, 3]

		return skewt_pdf(vlos, w, v_c, alpha, nu)

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
