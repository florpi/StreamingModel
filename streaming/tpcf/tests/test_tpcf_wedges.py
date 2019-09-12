from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import sympy as sp
from streaming.tpcf.tpcf_tools import tpcf_wedges
import pytest

def analytical_wedges(tpcf_s_mu, mu, n_wedges):
	'''

	Computes the analytical wedges of a given two point correlation tpcf_s_mu

	Args:
		tpcf_s_mu: function that returns the two point correlation function values at s and mu.
		mu: sympy symbolic variable.
		n_wedges: number of resulting wedges.

	Returns:
		1D array of the two point correlation function containing n_wedges for a given s value.

	'''


	mu_wedges = np.linspace( 0., 1, n_wedges + 1)

	wedge_sizes = mu_wedges[1:] - mu_wedges[:-1]


	return [1/wedge_sizes[w] * sp.integrate(tpcf_s_mu, ( mu, mu_wedges[w], mu_wedges[w+1])) for w in range(len(mu_wedges) - 1)]

'''
Examples of analytical functions to test against our integration
'''
def analytical_s_mu_1(s, mu):

	return s + mu

def analytical_s_mu_2(s, mu):

	return s*mu

def analytical_s_mu_3(s, mu):

	return s*mu**2

def analytical_s_mu_4(s, mu):

	return s*mu**4


@pytest.mark.parametrize("n_wedges, s_mu", [(5, analytical_s_mu_1),
						(4, analytical_s_mu_2), (3, analytical_s_mu_3),
						(3, analytical_s_mu_4)])
def test_wedges(n_wedges, s_mu):
	''' 
	Test wedges coincide with analytical prediction for different functional forms of tpcf_s_mu
	'''

	s = np.linspace(1,10,4)

	mu = sp.symbols('mu', positive = True)

	n_mu_bins_per_wedge = 50

	mu_bins = np.concatenate(
			[np.linspace(i * (1./n_wedges), (i+1) * 1./n_wedges, n_mu_bins_per_wedge) for i in range(n_wedges)]
			)



	s_c = 0.5 * (s[1:] + s[:-1])
	mu_c = 0.5 * (mu_bins[1:] + mu_bins[:-1])

	analytical_result = np.zeros((n_wedges, len(s_c) ))
	
	for i, s_value in enumerate(s_c):

		analytical_result[:, i] = analytical_wedges(s_mu(s_value, mu), mu, n_wedges)

	result = tpcf_wedges(s_mu(s_c.reshape(-1,1), mu_c.reshape(1,-1)), 
			mu_bins, n_wedges = n_wedges)

	np.testing.assert_array_almost_equal(analytical_result, result, decimal = 4)

