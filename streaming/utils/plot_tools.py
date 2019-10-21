import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, LogFormatterSciNotation
from scipy.integrate import simps
import numpy as np

import matplotlib.font_manager
import matplotlib as mpl
mpl.style.use('~/StreamingModel/streaming/utils/mplstyle')

def plot_taylor_expansion(toy_model, list_of_models, attribute, labels, legend = True):


	fig, (ax1, ax2) = plt.subplots(nrows=2,sharex = True, squeeze = True,
			       gridspec_kw = {'wspace':0, 'hspace':0.1, 'height_ratios':[4,1]})

	measured_attribute	= getattr(toy_model, attribute)
	ax1.plot(toy_model.s_c, 
				toy_model.s_c **2* measured_attribute,
				label = labels[0], color='black')


	for i, model in enumerate(list_of_models):

		ax1.plot(toy_model.s_c, toy_model.s_c**2 * getattr(model, attribute),
				label = labels[i+1], color = model.color, linestyle = model.linestyle, alpha = model.alpha)

		ax2.plot(toy_model.s_c, getattr(model, attribute) / measured_attribute,
				color = model.color, linestyle = model.linestyle, alpha = model.alpha) 

	ax1.set_ylim(np.min(toy_model.s_c**2 * measured_attribute)*(1 - np.sign(np.min(toy_model.s_c**2 * measured_attribute))* 0.2) , 
		np.max(toy_model.s_c**2 * measured_attribute)*(1 + np.sign(np.max(toy_model.s_c**2 * measured_attribute))* 0.2 )) 

	ax1.set_xlim(0.4, 49.1)

	ax2.set_ylim(0.95,1.05)
	ax2.fill_between(toy_model.s_c,0.99, 1.01, facecolor = 'yellow', alpha = 0.3)
	ax2.axhline(y = 1., linestyle='-', color='gray', alpha = 0.3)

	if legend:
		ax1.legend(frameon = False)

	if attribute == 'monopole':
		l = 0

	elif attribute == 'quadrupole':
		l = 2

	elif attribute == 'hexadecapole':
		l = 4

	ax1.set_ylabel(r'$s^2 \xi_{%d}(s)$'%l)

	ax2.set_xlabel(r'$s \mathrm{[Mpc/h]}$')
	ax2.set_ylabel(r'$\xi_%d/\xi_{%d, \mathrm{%s}}$'%(l,l,labels[0]))


def plot_residuals_std(simulation_direct_measurement, list_of_models, attribute, colors, legend = True):


	fig, (ax1, ax2) = plt.subplots(nrows=2,sharex = True, squeeze = True,
			       gridspec_kw = {'wspace':0, 'hspace':0.1, 'height_ratios':[4,1]})

	measured_attribute	= getattr(simulation_direct_measurement, attribute)
	ax1.errorbar(simulation_direct_measurement.s_c, 
				simulation_direct_measurement.s_c **2* measured_attribute .mean,
				yerr = simulation_direct_measurement.s_c**2* measured_attribute .std,
				label = 'N-body ', color='black')


	for i, model in enumerate(list_of_models):

		ax1.plot(model.s_c, model.s_c**2 * getattr(model, attribute),
				label = model.label, color = colors[i], linestyle = model.linestyle)

		ax2.plot(model.s_c, (getattr(model, attribute) - measured_attribute.mean)/measured_attribute.std,
				color = colors[i], linestyle = model.linestyle) 

	ax2.set_ylim(-5,5)
	ax1.set_xlim(0.4, 49.1)

	ax2.fill_between(simulation_direct_measurement.s_c,-1., 1., facecolor = 'yellow', alpha = 0.5)
	ax2.axhline(y = 0., linestyle='-', color='gray', alpha = 0.5)

	if legend:
		ax1.legend(frameon = False)

	if attribute == 'monopole':
		l = 0

	elif attribute == 'quadrupole':
		l = 2

	elif attribute == 'hexadecapole':
		l = 4

	ax1.set_ylabel(r'$s^2 \xi_{%d}(s)$'%l)

	ax2.set_xlabel(r'$s \mathrm{[Mpc/h]}$')
	ax2.set_ylabel(r'$\Delta{\xi}_%d/\sigma_{\xi_%d}$'%(l,l))


def jointplot(x, y, jointpdf, log=False):
	'''

	Plots the joint PDF of two random variables together with its marginals
		Args:
			x and y, random variables,
			jointpdf, their joint PDF

	'''
	threshold = (x>-20) & (x<20)
	x = x[threshold]
	y = y[threshold]
	jointpdf = jointpdf[threshold, :]
	jointpdf = jointpdf[:, threshold]
	#error_jointpdf = error_jointpdf[threshold, :]
	#error_jointpdf = error_jointpdf[:, threshold]

	nullfmt = NullFormatter()		  # no labels
	# definitions for the axes
	left, width = 0.1, 0.65
	bottom, height = 0.1, 0.65
	bottom_h = left_h = left + width + 0.02

	rect_scatter = [left, bottom, width, height]
	rect_histx = [left, bottom_h, width, 0.2]
	rect_histy = [left_h, bottom, 0.2, height]

	# start with a rectangular Figure
	plt.figure(1, figsize=(8, 8))

	axScatter = plt.axes(rect_scatter)
	axHistx = plt.axes(rect_histx)
	axHisty = plt.axes(rect_histy)

	# no labels
	axHistx.xaxis.set_major_formatter(nullfmt)
	axHisty.yaxis.set_major_formatter(nullfmt)

	# the scatter plot:
	if(log == True):
		cs = axScatter.contour(x, y, np.log10(jointpdf))
	else:
		cs = axScatter.contour(x, y, jointpdf)
	fmt = {}
	for level in cs.levels:
		fmt[level] = f'{10**level:.1e}'

	plt.clabel(cs, cs.levels, inline = True, fmt = fmt)

	#error_x = simps(error_jointpdf, y , axis =0)
	#error_y = simps(error_jointpdf, x , axis = -1)
	marginal_x = simps(jointpdf, y, axis = 0)
	marginal_y = simps(jointpdf, x, axis = -1)

	axHistx.semilogy(y, marginal_x, color = 'midnightblue')
	axHisty.semilogx(marginal_y, x, color = 'slategray')
	#axHistx.errorbar(y, marginal_x, yerr =  error_x, color = 'midnightblue')
	#axHistx.set_yscale('log')
	#axHisty.errorbar(marginal_y, x, xerr = error_y, color = 'slategray')
	#axHisty.set_xscale('log')


	axScatter.set_xlabel(r'$v_r \, \mathrm{[Mpc/h]}$')
	axScatter.set_ylabel(r'$v_t \, \mathrm{[Mpc/h]}$')

	axHistx.set_ylabel(r'$\mathcal{P}(v_r | r)$')
	axHisty.set_xlabel(r'$\mathcal{P}(v_t |r)$')


