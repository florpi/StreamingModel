from scipy.optimize import curve_fit
import numpy as np
from scipy.stats import norm
from streaming.utils.digitize import bin_number_from_center 
from streaming.moments.project_moments import project_moment_to_los
import streaming.moments.compute_moments as cm 

def gaussian(v, mean, std):

	return norm.pdf(v, loc = mean, scale = std)


def fit_gaussian(r_perpendicular, r_parallel, v_los, pdf_los, first_guess ):

	popt = np.zeros((len(r_perpendicular),
					len(r_parallel), 2))


	
	for i, rperp in enumerate(r_perpendicular):
		for j, rpar in enumerate(r_parallel):

			popt[i,j], pcov = curve_fit( gaussian, v_los,
										pdf_los[i,j,:],
										p0 = (first_guess[0][i,j], first_guess[1][i,j])
										)


	mean = popt[...,0]
	std = popt[...,1]

	return mean, std

def bestfit_gaussian(sim_measurement):

	mean = project_moment_to_los(sim_measurement, sim_measurement.r_parallel.reshape(1,-1), 
			sim_measurement.r_perpendicular.reshape(-1, 1), 1, mode = 'm')
	std = cm.std(project_moment_to_los(sim_measurement, sim_measurement.r_parallel.reshape(1,-1), 
		sim_measurement.r_perpendicular.reshape(-1,1), 2, mode = 'c'))



	mean, std = fit_gaussian(sim_measurement.r_perpendicular,
		sim_measurement.r_parallel, sim_measurement.v_los, 
		sim_measurement.jointpdf_los.mean, first_guess = (mean, std))

	def pdf_los(v_los, r_perp, r_parallel):

		r_perp_bins = bin_number_from_center(r_perp, sim_measurement.r_perpendicular)	
		r_par_bins = bin_number_from_center(r_parallel, sim_measurement.r_parallel)	

		return gaussian(v_los, mean[r_perp_bins, r_par_bins], std[r_perp_bins, r_par_bins])

	return pdf_los

