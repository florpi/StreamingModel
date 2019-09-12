from distutils.core import setup

setup(
    name='StreamingModel',
	author="Carolina Cuesta-Lazaro",
	author_email="carolina.cuesta-lazaro@durham.ac.uk",
    version='0.1dev',
	url='https://github.com/florpi/StreamingModel',
    packages=['streaming'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.txt').read(),

	install_requires=[
					'numpy',
					'jupyter',
					'h5py',
					'pandas',
					'scipy',
					'seaborn',
					'matplotlib',
					'halotools'
					],

	zip_safe=False
)

