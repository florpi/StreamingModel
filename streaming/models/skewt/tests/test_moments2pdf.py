import numpy as np
import pytest
from streaming.models.skewt import moments2pdf as st
from scipy.integrate import simps

def test_normalized():
	v = np.linspace(-100.,100.,100)
	w_true = 5.
	v_c_true = 2.5
	alpha_true = -1.
	nu_true = 10.2

	np.testing.assert_almost_equal(simps(st.skewt_pdf(v, w_true, v_c_true, alpha_true, nu_true), v), 1.)


def test_moments2parameters():
	v = np.linspace(-100.,100.,100)
	w_true = 5.
	v_c_true = 4.5
	alpha_true = -1.
	nu_true = 10.2

	skewt_true = st.skewt_pdf(v, w_true, v_c_true, alpha_true, nu_true)


	true_mean = simps(skewt_true * v, v)
	true_std = np.sqrt(simps(skewt_true * (v - true_mean)**2, v)) 
	true_gamma1 = simps(skewt_true * (v - true_mean)**3, v)/true_std**3
	true_gamma2 = simps(skewt_true * (v - true_mean)**4, v)/true_std**4 - 3.


	w, v_c, alpha, nu = st.moments2parameters(true_mean, true_std, true_gamma1, true_gamma2, p0 = (-0.5, 5))

	np.testing.assert_almost_equal(w_true, w, decimal = 2)
	np.testing.assert_almost_equal(v_c_true, v_c, decimal = 2)
	np.testing.assert_almost_equal(alpha_true, alpha, decimal = 2)
	np.testing.assert_almost_equal(nu_true, nu, decimal = 2)


	mean_estimated, std_estimated, gamma1_estimated, gamma2_estimated = st.parameters2moments(w_true, v_c_true, alpha_true, nu_true)

	np.testing.assert_almost_equal(mean_estimated, true_mean, decimal = 2)
	np.testing.assert_almost_equal(std_estimated, true_std, decimal = 2)
	np.testing.assert_almost_equal(gamma1_estimated, true_gamma1, decimal = 2)
	np.testing.assert_almost_equal(gamma2_estimated, true_gamma2, decimal = 2)


