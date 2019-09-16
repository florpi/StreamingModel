from streaming.taylor_expansion.taylor_expansion import TaylorExpansion, AnalyticalTaylorExpansion
import numpy as np
import pandas as pd
import pytest


def m(r, a, b, c):
	
	return -a * np.exp(-b * r) + c 

def m_dot(r, a, b, c):

	return a * b * np.exp(-b * r)

def tpcf(r, a, b):
	return a*r**2 + b


def tpcf_dot(r, a, b):
	return 2. * a *r



@pytest.mark.parametrize('multipole', ['mono', 'quad', 'hexa'])
def test_first_order_mono(multipole):
	m_a = 20.
	m_b = 4.
	m_c = 6.

	tpcf_a = 10.
	tpcf_b = 3.

	moments = {
				'tpcf': {'function': tpcf , 'popt': [tpcf_a, tpcf_b]},
				'd1_tpcf': {'function': tpcf_dot , 'popt': [tpcf_a, tpcf_b]},
				'm_10': {'function': m, 'popt':[m_a, m_b, m_c] }, 
				'd1_m_10': {'function': m_dot, 'popt': [m_a, m_b, m_c]},
				}

	moments_df = pd.DataFrame(moments)


	mu =  np.sort(1 - np.geomspace(0.0001, 1., 120))
	s = np.arange(0.,50.,1)

	te = TaylorExpansion(s, mu, moments_df, 1, cutoff_tpcf = 2)

	te_analytical = AnalyticalTaylorExpansion(s, moments_df, 1)

	np.testing.assert_allclose(getattr(te, multipole), getattr(te_analytical, multipole), rtol= 1.e-2)


