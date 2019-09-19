import numpy as np
from streaming.utils.digitize import bin_number_from_center, mask_valid_bins


def measured_los_pdf(sim_measurement):


	def function_los(v_los: np.array, r_perp: np.array, r_parallel: np.array):

		r_perp_bins = bin_number_from_center(r_perp, sim_measurement.r_perpendicular)	
		r_par_bins = bin_number_from_center(r_parallel, sim_measurement.r_parallel)	
		v_bins = bin_number_from_center(v_los, sim_measurement.v_los)	

		v_mask = mask_valid_bins(v_los, sim_measurement.v_los)


		output = sim_measurement.jointpdf_los.mean[r_perp_bins, r_par_bins, v_bins]

		# Filter output for bounds
		filtered = np.where(v_mask, output, 0.0)

		return filtered 

	return function_los


