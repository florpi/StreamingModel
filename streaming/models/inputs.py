from streaming.utils.mean_error import MeanError
from streaming.moments import compute_moments as cm
from streaming.moments import project_moments as pm
import h5py
import numpy as np

class SimulationMeasurements():
	'''Class containing summary statistics measured from the simulation.

	Attributes:
		r:
		v_r:
		vt:
		r_perpendicular:
		r_parallel:
		v_los:
		n_boxes:
		jointpdf_rt:
		jointpdf_los
		m_10: mean of the radial velcity pairwise distribution.
		c_xy: central moment of the radial and tangential distribution of order (radial = x, tangential = y).
			Up to x = 4, y = 4. By symmetry, some of the moments vanish and are not included.
		m_1: mean of the line of sight velocity distribution.
		c_x:
		r_tpcf:
		s:
		mu:
		tpcf:
		monopole:
		quadrupole:
		hexadecapole:

	'''
	def __init__(self, pdf_filename: str, tpcf_filename: str) -> None:
		'''
		Args:
			pdf_filename: path to hdf5 file where the pairwise velocity distributions are saved.
			tpcf_filename: path to thdf5 file where the two point statistics are saved.

		'''

		self.read_hdf5_file(pdf_filename)

		self.m_10 = MeanError(cm.moment_2D(1,0, self.v_t, self.v_r, self.jointpdf_rt), interpolate_x = self.r)

		nonzero_central_moments = ['c_20', 'c_02', 'c_30', 'c_12', 'c_22', 'c_40', 'c_04']
		for moment in nonzero_central_moments:
			setattr(self, moment, MeanError(cm.central_moment_2D(int(moment[-2]), int(moment[-1]),
											self.v_t, self.v_r, self.jointpdf_rt),
											interpolate_x = self.r))

		self.m_1= MeanError(cm.moment_1D(1, self.v_los, self.jointpdf_los))
		nonzero_central_moments_los = ['c_2', 'c_3', 'c_4']
		for moment in nonzero_central_moments_los:
			setattr(self, moment, MeanError(cm.central_moment_1D(int(moment[-1]), self.v_los, self.jointpdf_los)))


		# PDFs
		self.jointpdf_los = MeanError(self.jointpdf_los)
		self.jointpdf_rt = MeanError(self.jointpdf_rt)

		self.read_tpcf(tpcf_filename)
		self.tpcf = MeanError(self.tpcf, interpolate_x = self.r_tpcf)
		self.monopole = MeanError(self.monopole)
		self.quadrupole = MeanError(self.quadrupole)
		self.hexadecapole = MeanError(self.hexadecapole)



	def read_hdf5_file(self, filename):

		with h5py.File(filename, "r") as f:


			self.r = f['r'][:]
			self.v_r = f['vr'][:]
			self.v_t = f['vt'][:]
			self.r_perpendicular= f['r_perpendicular'][:]
			self.r_parallel= f['r_parallel'][:]
			self.v_los = f['v_los'][:]

			self.n_boxes = f['n_boxes'].value

			self.jointpdf_rt = []
			self.jointpdf_los = []

			for box in range(1, self.n_boxes):
				self.jointpdf_rt.append(f[f'box_{box}']['jointpdf_rt'][:])
				self.jointpdf_los.append(f[f'box_{box}']['jointpdf_los'][:])

		self.jointpdf_rt = np.asarray(self.jointpdf_rt)
		self.jointpdf_los = np.asarray(self.jointpdf_los)

	def read_tpcf(self, filename):

		with h5py.File(filename, 'r') as f:

			self.r_tpcf = f['r'][:]
			self.s_c = f['s'][:]
			self.mu = f['mu'][:]
			self.tpcf, self.tpcf_s_mu, self.monopole, self.quadrupole, self.hexadecapole  = [], [], [], [], []

			for box in range(1, self.n_boxes):
				self.tpcf.append(f[f'box_{box}']['tpcf'][:])
				self.tpcf_s_mu.append(f[f'box_{box}']['tpcf_s_mu'][:])
				self.monopole.append(f[f'box_{box}']['monopole'][:])
				self.quadrupole.append(f[f'box_{box}']['quadrupole'][:])
				self.hexadecapole.append(f[f'box_{box}']['hexadecapole'][:])
	

	def get_los_moments_from_rt(self):

		mean = pm.project_moment_to_los(self, self.r_parallel.reshape(-1,1),
				self.r_perpendicular.reshape(1,-1), 1, mode = 'm').T

		c2  = pm.project_moment_to_los(self, self.r_parallel.reshape(-1,1),
				self.r_perpendicular.reshape(1,-1), 2, mode = 'c')
 
		std = cm.std(c2 ).T

		gamma1 = cm.gamma1(pm.project_moment_to_los(self, self.r_parallel.reshape(-1,1),
				self.r_perpendicular.reshape(1,-1), 3, mode = 'c'), c2 ).T

		gamma2 = cm.gamma2(pm.project_moment_to_los(self, self.r_parallel.reshape(-1,1),
				self.r_perpendicular.reshape(1,-1), 4, mode = 'c'), c2 ).T

		return mean, std, gamma1, gamma2


	def get_los_moments_directly(self):

		mean = self.m_1.mean

		std = cm.std(self.c_2.mean)

		gamma1 = cm.gamma1(self.c_3.mean, self.c_2.mean)

		gamma2 = cm.gamma2(self.c_4.mean, self.c_2.mean)

		return mean, std, gamma1, gamma2


if __name__=='__main__':

	pdf_filename = "../data/pairwise_velocity_pdf.hdf5"
	tpcf_filename = "../data/tpcf.hdf5"
	measured = Measured(pdf_filename, tpcf_filename)

	print(measured.c_20.mean(measured.r))


