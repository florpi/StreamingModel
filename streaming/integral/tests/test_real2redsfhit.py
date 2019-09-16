import pytest
from streaming.integral.real2redshift import simps_integrate, simps_integrate_pi_sigma
import numpy as np

#TODO: Add more tests with los_pdf

def los_pdf(vlos, rperp, rparallel):

	return np.ones_like(vlos) 

def tpcf_function(r):

	return 4 * np.exp(-r**2) - 1.

def test_simps_integrate_tpcf():

	s = np.linspace(0., 20, 10)
	mu = np.linspace(0.,1., 5)

	result = simps_integrate(s, mu, tpcf_function, los_pdf, epsilon = 1.e-6, n = 1200)

	s_c = 0.5 * ( s[1:] + s[:-1] )
	mu_c = 0.5 * ( mu[1:] + mu[:-1] )

	S = s_c.reshape(-1,1)
	MU = mu_c.reshape(1,-1)

	s_parallel = S * MU
	s_perp = S * np.sqrt(1 - MU**2)

	s_perp = s_perp.reshape(-1, 1)

	analytical_result = (np.exp(-s_perp**2) * 4. * 1.77245 - 1.).reshape(result.shape)

	np.testing.assert_almost_equal(result, analytical_result, decimal = 4)

def test_simps_integrate_tpcf_pi_sigma():

	s = np.linspace(0., 20, 10)

	s_c = 0.5 * ( s[1:] + s[:-1] )

	result = simps_integrate_pi_sigma(s, s, tpcf_function, los_pdf, epsilon = 1.e-6, n = 1200)


	s_c= s_c.reshape(-1, 1)

	ind_result = np.array(len(s_c) * [1.77245]).reshape(1,-1)
	analytical_result = np.exp(-s_c**2) * 4. * ind_result - 1.


	np.testing.assert_almost_equal(result, analytical_result, decimal = 4)


if __name__=='__main__':

	test_simps_integrate_tpcf()

