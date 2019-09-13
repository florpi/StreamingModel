from streaming.utils.mean_error import MeanError
import h5py
import numpy as np



class Measured():

	def __init__(self, pdf_filename, tpcf_filename):

		self.read_hdf5_file(pdf_filename)
		self.jointpdf_los = MeanError(self.jointpdf_los)
		self.jointpdf_rt = MeanError(self.jointpdf_rt)

		self.read_tpcf(tpcf_filename)
		self.tpcf = MeanError(self.tpcf)
		self.monopole = MeanError(self.monopole)
		self.quadrupole = MeanError(self.quadrupole)
		self.hexadecapole = MeanError(self.hexadecapole)



	def read_hdf5_file(self, filename):

		with h5py.File("../data/pairwise_velocity_pdf.hdf5", "r") as f:


			self.r = f['r'][:]
			self.vr = f['vr'][:]
			self.vt = f['vt'][:]
			self.r_perpendicular= f['r_perpendicular'][:]
			self.r_parallel= f['r_parallel'][:]

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

			self.r_tpcf = f['r']
			self.s = f['s']
			self.mu = f['mu']
			self.tpcf, self.tpcf_s_mu, self.monopole, self.quadrupole, self.hexadecapole  = [], [], [], [], []

			for box in range(1, self.n_boxes):
				self.tpcf.append(f[f'box_{box}']['tpcf'][:])
				self.tpcf_s_mu.append(f[f'box_{box}']['tpcf_s_mu'][:])
				self.monopole.append(f[f'box_{box}']['monopole'][:])
				self.quadrupole.append(f[f'box_{box}']['quadrupole'][:])
				self.hexadecapole.append(f[f'box_{box}']['hexadecapole'][:])



if __name__=='__main__':

	pdf_filename = "../data/pairwise_velocity_pdf.hdf5"
	tpcf_filename = "../data/tpcf.hdf5"
	measured = Measured(pdf_filename, tpcf_filename)

	print(measured.jointpdf_los.mean)
	print(measured.jointpdf_los.std)
