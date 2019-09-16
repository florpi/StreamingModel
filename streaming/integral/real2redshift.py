from scipy.interpolate import interp1d, interp2d
import numpy as np
from scipy.integrate import simps, quadrature, quad
import quadpy


def integrand_s_mu(s_c, mu_c, twopcf_function, los_pdf_function): 
	'''

	Computes the streaming model integrand ( https://arxiv.org/abs/1710.09379, Eq 22 ) at s, mu

	Args:
		s_c: np.array
			bin centers for the pair distance bins.
		mu_c: np.array
			bin centers for the cosine of the angle rescpect to the line of sight bins.
		twopcf_function: function
			function that given pair distance as an argument returns the real space two point 
			correlation function.
		los_pdf_function: function
			given the line of sight velocity, perpendicular and parallel distances to the line
			of sight, returns the value of the line of sight pairwise velocity distribution.

	Returns:
		integrand: np.ndarray
			2-D array with the value of the integrand evaluated at the given s_c and mu_c.			

	'''
	def integrand(y):

		S = s_c.reshape(-1,1)
		MU = mu_c.reshape(1,-1)

		s_parallel = S * MU


		s_perp = S * np.sqrt(1 - MU**2)

		# Use reshape to vectorize all possible combinations
		s_perp = s_perp.reshape(-1, 1)
		s_parallel = s_parallel.reshape(-1,1)
		y = y.reshape(1, -1)
		vlos = (s_parallel - y) * np.sign(y)

		r = np.sqrt(s_perp**2 + y **2)
		

		return los_pdf_function( vlos, s_perp, np.abs(y)) * (1 + twopcf_function(r))

	return integrand


def simps_integrate(s, mu, twopcf_function, los_pdf_function, limit = 70., epsilon = 0.0001, n = 300): 
	'''

	Computes the streaming model integral ( https://arxiv.org/abs/1710.09379, Eq 22 ) 

	Args:
		s: np.array
			pair distance bins.
		mu: np.array
			cosine of the angle rescpect to the line of sight bins.
		twopcf_function: function
			function that given pair distance as an argument returns the real space two point 
			correlation function.
		los_pdf_function: function
			given the line of sight velocity, perpendicular and parallel distances to the line
			of sight, returns the value of the line of sight pairwise velocity distribution.
		limit: float
			r_parallel limits of the integral.
		epsilon: float
			due to discontinuity at zero, add small offset +-epsilon to estimate integral.
		n: int
			number of points to evaluate the integrand.

	Returns:
		twopcf_s: np.ndarray
			2-D array with the resulting redshift space two point correlation function

	'''


	s_c = 0.5 * ( s[1:] + s[:-1] )
	mu_c = 0.5 * ( mu[1:] + mu[:-1] )

	streaming_integrand = integrand_s_mu(s_c, mu_c, twopcf_function, los_pdf_function)

	# split integrand in two due to discontinuity at 0

	r_integrand = np.linspace(-limit, -epsilon, n)
	integral_left = simps(streaming_integrand(r_integrand), r_integrand, axis = -1).reshape((s_c.shape[0], mu_c.shape[0]))

	r_integrand = np.linspace(epsilon, limit, n)
	integral_right = simps(streaming_integrand(r_integrand), r_integrand, axis = -1).reshape((s_c.shape[0], mu_c.shape[0]))

	twopcf_s = integral_left + integral_right - 1.

	return twopcf_s


def integrand_pi_sigma(s_perp_c, s_parallel_c, twopcf_function, los_pdf_function): 
	'''

	Computes the streaming model integrand ( https://arxiv.org/abs/1710.09379, Eq 22 ) at s_perp_c and s_parallel_c

	Args:
		s_perp_c: np.array
			bin centers for the pair distance bins perpendicular to the line of sight.
		s_parallel_c: np.array
			bin centers for the pair distance bins parallel to the line of sight.
		twopcf_function: function
			function that given pair distance as an argument returns the real space two point 
			correlation function.
		los_pdf_function: function
			given the line of sight velocity, perpendicular and parallel distances to the line
			of sight, returns the value of the line of sight pairwise velocity distribution.

	Returns:
		integrand: np.ndarray
			2-D array with the value of the integrand evaluated at the given s_perp_c and s_parallel_c.			

	'''


	s_perp_c = s_perp_c.reshape(-1, 1)
	s_parallel_c = s_parallel_c.reshape(-1, 1)

	def integrand(y):

		y = y.reshape(1, -1)

		vlos = (s_parallel_c.reshape(-1, 1) - y) * np.sign(y)
		vlos = vlos[np.newaxis, ...]

		r = np.sqrt(s_perp_c**2 + y **2)[:, np.newaxis, :]

		return los_pdf_function( vlos, s_perp_c, np.abs(y))\
				* (1 + twopcf_function(r))

	return integrand


def simps_integrate_pi_sigma(s_perp, s_parallel, twopcf_function, los_pdf_function, limit = 70., epsilon = 0.0001, n = 300): 
	'''

	Computes the streaming model integral ( https://arxiv.org/abs/1710.09379, Eq 22 )

	Args:
		s_perp: np.array
			pair distance bins perpendicular to the line of sight.
		s_parallel: np.array
			pair distance bins parallel to the line of sight.
		twopcf_function: function
			function that given pair distance as an argument returns the real space two point 
			correlation function.
		los_pdf_function: function
			given the line of sight velocity, perpendicular and parallel distances to the line
			of sight, returns the value of the line of sight pairwise velocity distribution.
		limit: float
			r_parallel limits of the integral.
		epsilon: float
			due to discontinuity at zero, add small offset +-epsilon to estimate integral.
		n: int
			number of points to evaluate the integrand.

	Returns:
		twopcf_s: np.ndarray
			2-D array with the resulting redshift space two point correlation function

	'''



	s_perp_c = 0.5 * ( s_perp[1:] + s_perp[:-1] )
	s_parallel_c = 0.5 * ( s_parallel[1:] + s_parallel[:-1] )


	streaming_integrand = integrand_pi_sigma(s_perp_c, s_parallel_c, twopcf_function, los_pdf_function)

	# split integrand in two due to discontinuity at 0

	r_integrand = np.linspace(-limit, -epsilon, n)
	integral_left = simps(streaming_integrand(r_integrand), r_integrand, axis = -1)	
	
	r_integrand = np.linspace(epsilon, limit, n)
	integral_right = simps(streaming_integrand(r_integrand), r_integrand, axis = -1)

	return integral_left + integral_right - 1.



