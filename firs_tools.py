import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, SpanSelector
from scipy.io import readsav
import scipy.interpolate as scinterp
from scipy.optimize import curve_fit
import numpy.polynomial.polynomial as npoly
import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.io.fits as fits
from sunpy.coordinates import frames
from importlib import resources
import h5py
import FTS_atlas


# NOTE: IMPORTANT
# ---------------
# This package does not constitute a substitute for the IDL FIRS pipeline.
# Rather, this should be used after the FIRS pipeline has outputted a level-1 map.
# When running the IDL pipeline, pass a FIRS flat field through it as if it were a science map.
# This flat is used for fringe corrections.
# At a later date, I will also use this map for the prefilter.
# This package will perform a reasonably-decent set of additional calibrations to level-1 FIRS data
# Currently, it will perform:
# wavelength calibration (from FTS )
# Prefilter correction (from Stokes-I and FTS atlas)
# Correction for the linear tilt of the QUV spectra (via 1d fitting)
# Correction for fringes (via Fourier filtering the flat map)
# It will then wrap the FIRS file into a nice fits HDUList, and write it as a level-1.5 product.


def _gaussian(x, a0, a1, a2, c):
	"""Function to define a Gaussian profile over range x

	Parameters
	----------
	x  : array-like
		The range over which the gaussian is calculated
	a0 : float
		The height of the Gaussian peak
	a1 : float
		The offset of the Gaussian core
	a2 : float
		The standard deviation of the Gaussian, the width
	c  : float
		Vertical offset of the gaussian continuum

	Returns
	-------
	y : array-like
		The corresponding intensities for a Gaussian defined over x
	"""
	z = (x-a1)/a2
	y = a0*np.exp(- z**2 / 2.) + c
	return y


def _fts_window(wavemin, wavemax, atlas='FTS', norm=True, lines=False):
	""" For a given wavelength range, return the solar reference spectrum within that range.

	Parameters
	----------
	wavemin : float
	The minimum desired wavelength (in angstroms)
	wavemax : float
		The maximum desired wavelength (in angstroms)
	atlas : str
		Currently accepts "Wallace" and "FTS" (as these are the only two downloaded).
		Wallace takes the 2011 Wallace updated atlas, FTS takes the base 1984 FTS atlas.
	norm : bool
		If False and atlas is set to "FTS", will return the solar irradiance between wavemin and wavemax. Do not recommend
	lines : bool
		If True, returns additional arrays denoting line centers and names between wavemin and wavemax.

	Returns
	-------
	wave : array-like
		Array of wavelengths from the FTS atlas between wavemin and wavemax
	spec : array-like
		Array of corresponding spectral values
	line_centers : array-like, optional
		Array of line centers between wavemin and wavemax
	line_names : array-like, optional
		Array of line names between wavemin and wavemax
	"""

	def read_data(path, fname) -> np.array:
		with resources.path(path, fname) as df:
			return np.load(df)

	if wavemin >= wavemax:
		print("Minimum Wavelength is greater than or equal to the Maximum Wavelength. Reverse those, bud.")
		return None
	if (wavemin <= 2960) or (wavemax >= 13000):
		print(
			"Your selected wavelengths are not in FTS atlas bounds. Come back when I bother downloading the IR/UV atlas")
		return None

	if atlas.lower() == "wallace":
		if (wavemax <= 5000.) or (wavemin <= 5000.):
			atlas_angstroms = read_data('FTS_atlas', 'Wallace2011_290-1000nm_Wavelengths.npy')
			atlas_spectrum = read_data("FTS_atlas", 'Wallace2011_290-1000nm_Observed.npy')
		else:
			atlas_angstroms = read_data('FTS_atlas', 'Wallace2011_500-1000nm_Wavelengths.npy')
			atlas_spectrum = read_data('FTS_atlas', 'Wallace2011_500-1000nm_Corrected.npy')
	else:
		atlas_angstroms = read_data('FTS_atlas', 'FTS1984_296-1300nm_Wavelengths.npy')
		if norm:
			atlas_spectrum = read_data('FTS_atlas', 'FTS1984_296-1300nm_Atlas.npy')
		else:
			print("Using full solar irradiance. I hope you know what you're doing")
			atlas_spectrum = read_data('FTS_atlas', 'FTS1984_296-1300nm_Irradiance.npy')

	idx_lo = _find_nearest(atlas_angstroms, wavemin) - 5
	idx_hi = _find_nearest(atlas_angstroms, wavemax) + 5

	wave = atlas_angstroms[idx_lo:idx_hi]
	spec = atlas_spectrum[idx_lo:idx_hi]

	if lines:
		line_centers_full = read_data("FTS_atlas", 'RevisedMultiplet_Linelist_2950-13200_CentralWavelengths.npy')
		line_names_full = read_data('FTS_atlas', 'RevisedMultiplet_Linelist_2950-13200_IonNames.npy')

		line_selection = (line_centers_full < wavemax) & (line_centers_full > wavemin)
		line_centers = line_centers_full[line_selection]
		line_names = line_names_full[line_selection]
		return wave, spec, line_centers, line_names
	else:
		return wave, spec


def _rolling_median(data, window):
	"""Simple rolling median function, rolling by the central value. By default, preserves the edges to provide
	an output array of the same shape as the input.
	Parameters:
	-----------
	data : array-like
		Array of data to smooth
	window : int
		Size of the window to median-ify
	
	Returns:
	--------
	rolled : array-like
		Rolling median of input array
	"""

	rolled = np.zeros(len(data))
	half_window = int(window/2)
	if half_window >= 4:
		for i in range(half_window):
			rolled[i] = np.nanmedian(data[i:i+1])
			rolled[-(i+1)] = np.nanmedian(data[(-(i+4)):(-(i+1))])
	else:
		rolled[:half_window] = data[:half_window]
		rolled[-(half_window + 1):] = data[-(half_window + 1):]
	for i in range(len(data) - window):
		rolled[half_window + i] = np.nanmedian(data[i:half_window + i])
	return rolled


def _find_nearest(array, value):
	""" Determines the index of the closest value in an array to a specified other value

	Parameters
	----------
	array : array-like
		An array of int/float values
	value : int,float
		A value that we will check for in the array

	Returns
	-------
	idx : int
		The index of the input array where the closest value is found
	"""
	idx = (np.abs(array-value)).argmin()
	return idx


def _correct_datetimes(datetime):
	"""Unfortunately, FIRS for some ungodly reason doesn't record its timestamps in UTC.
	Rather, it records in local time. This is irritating.
	This function returns the offset in hours between local and UTC

	Parameters:
	-----------
	datetime : numpy datetime64
		Reference datetime object

	Returns:
	--------
	offset : float
		Offset in hours between local and UTC
	"""

	year = datetime.astype('datetime64[Y]')

	dst_start = np.datetime64(year.astype(str) + "-03-12 00:00")
	dst_end = np.datetime64(year.astype(str) + "-11-05 00:00")

	if (datetime >= dst_start) & (datetime <= dst_end):
		offset = np.timedelta64(6, 'h')
	else:
		offset = np.timedelta64(7, 'h')

	return offset


def read_firs(firs_file):
	"""Simple routine to read FIRS binary file into numpy array for ease of use.
	Assumes a .sav file of the same name in the same location as firs_file
	Parameters:
	-----------
	firs_file : str
		path to FIRS *.dat binary file for readin

	Returns:
	--------
	firs_data : array-like
		Formatted nd array of FIRS data
	"""
	sav_file = firs_file + '.sav'

	firs_mets = readsav(sav_file)
	firs_nspex = int(firs_mets['dx_final'] + 1)
	firs_ysize = int(firs_mets['dy_final'] + 1)

	firs = np.fromfile(firs_file, dtype=np.float32)
	firs_xsize = int(len(firs)/4/firs_nspex/firs_ysize)

	return firs.reshape((firs_xsize, 4, firs_ysize, firs_nspex))


def select_callback(eclick, erelease):
	"""
	Callback for line selection, with eclick and erelease being the press and release events.
	"""

	x1, y1 = eclick.xdata, eclick.ydata
	x2, y2 = erelease.xdata, erelease.ydata

	return sorted([x1, x2]), sorted([y1, y2])


def select_image_region(img_data, xdata=None):
	"""Generalized function to plot an image and allow the user to select a region.
	Parameters:
	-----------
	img_data : array-like
		Image data to be plotted. Uses imshow if 2d, plot if 1d
	xdata : None or array-like, optional
		If not none, takes an array of x-values for use with plt.plot
	
	Returns:
	--------
	selections : array-like
		Array of selected indices
	"""

	fig = plt.figure()
	ax = fig.add_subplot(111)
	
	if len(img_data.shape) == 1:
		if xdata is not None:
			ax.plot(xdata, img_data, drawstyle='steps-mid')
		else:
			xdata = np.arange(len(img_data))
			ax.plot(img_data, drawstyle='steps-mid')
			ax.set_xlim(xdata[0], xdata[-1])

		def onselect(xmin, xmax):
			indmin, indmax = np.searchsorted(xdata, (xmin, xmax))
			indmax = min(len(xdata) - 1, indmax)
			region_x = xdata[indmin:indmax]
			region_y = img_data[indmin:indmax]

		ax.set_title(f"Click and drag to select a Span.\n Press t to toggle.")
		
		selector = SpanSelector(
				ax,
				onselect,
				"horizontal",
				useblit=True,
				button=[1, 3],
				interactive=True,
				drag_from_anywhere=True)
		plt.show()
	elif len(img_data.shape) == 2:
		ax.imshow(img_data, origin='lower', aspect='auto', interpolation='none')

		ax.set_title(f"Click and drag to draw a {RectangleSelector.__name__}.\n Press t to toggle")
		selector = RectangleSelector(
			ax,
			select_callback,
			useblit=True,
			button=[1, 3],
			spancoords='pixels',
			interactive=True)
		plt.show()
	
	else:
		print("Invalid image dimensions, must be 1- or 2-D array")
		return None 

	return selector.extents


def select_spec_region(spectrum, reference_spectrum):
	"""Function the plots the spectrum and reference in adjacent axes, then allows the user to alter the highlighted
	portions to select discrete corresponding regions in the spectrum and reference.

	Parameters:
	-----------
	spectrum : array-like
		Spectral data to for selection
	reference_spectrum : array-like
		Spectrum of the reference

	Returns:
	--------
	line_selections : array-like
		Array of selected indices
	"""

	xdata = np.arange(len(spectrum))
	xref = np.arange(len(reference_spectrum))

	fig = plt.figure()
	ax_dat = fig.add_subplot(211)
	ax_dat.plot(xdata, spectrum, drawstyle='steps-mid', color='k')

	ax_ref = fig.add_subplot(212)
	ax_ref.plot(xref, reference_spectrum, drawstyle='steps-mid', color='k')

	def onselect1(xmin, xmax):
		indmin, indmax = np.searchsorted(xdata, (xmin, xmax))
		indmax = min(len(xdata) - 1, indmax)

	def onselect2(xmin, xmax):
		indmin, indmax = np.searchsorted(xdata, (xmin, xmax))
		indmax = min(len(xdata) - 1, indmax)

	def onselect3(xmin, xmax):
		indmin, indmax = np.searchsorted(xref, (xmin, xmax))
		indmax = min(len(xref) - 1, indmax)

	def onselect4(xmin, xmax):
		indmin, indmax = np.searchsorted(xref, (xmin, xmax))
		indmax = min(len(xref) - 1, indmax)

	ax_dat.set_title(
		f"Click and drag to select a two regions in the top plot." +
		"\nSelect the same two below. Close the window when done."
	)

	selector1 = SpanSelector(
		ax_dat,
		onselect1,
		"horizontal",
		useblit=True,
		props=dict(alpha=0.5, facecolor='C0'),
		interactive=True,
		drag_from_anywhere=True,
		ignore_event_outside=True
	)

	selector2 = SpanSelector(
		ax_dat,
		onselect2,
		"horizontal",
		useblit=True,
		props=dict(alpha=0.5, facecolor='C1'),
		interactive=True,
		drag_from_anywhere=True,
		ignore_event_outside=True
	)

	selector3 = SpanSelector(
		ax_ref,
		onselect3,
		"horizontal",
		useblit=True,
		props=dict(alpha=0.5, facecolor='C0'),
		interactive=True,
		drag_from_anywhere=True,
		ignore_event_outside=True
	)

	selector4 = SpanSelector(
		ax_ref,
		onselect4,
		"horizontal",
		useblit=True,
		props=dict(alpha=0.5, facecolor='C1'),
		interactive=True,
		drag_from_anywhere=True,
		ignore_event_outside=True
	)

	selector1._selection_completed = True
	selector2._selection_completed = True
	selector3._selection_completed = True
	selector4._selection_completed = True

	# Defaults. Can just close the window if this is fine
	# Typically, FIRS 10830 data is ~1019 wavelength samples
	# So if there's more than 900 points, it's safe ish to use these values
	if len(xdata) > 900:
		xmin1, xmax1 = 203, 220
		xcen1 = np.where(spectrum[xmin1:xmax1+1] == np.nanmin(spectrum[xmin1:xmax1+1]))[0][0]
		selector1.extents = (xmin1 + xcen1 - 6, xmin1 + xcen1 + 6)

		xmin2, xmax2 = 788, 800
		xcen2 = np.where(spectrum[xmin2:xmax2 + 1] == np.nanmin(spectrum[xmin2:xmax2 + 1]))[0][0]
		selector2.extents = (xmin2 + xcen2 - 4, xmin2 + xcen2 + 4)

		xmin3, xmax3 = 669, 690
		selector3.extents = (xmin3, xmax3)

		xmin4, xmax4 = 2392, 2410
		selector4.extents = (xmin4, xmax4)
	else:
		xmin1, xmax1 = xdata[0], xdata[int(len(xdata)/8)]
		selector1.extents = (xmin1, xmax1)
		xmin2, xmax2 = xdata[int(7 * len(xdata) / 8)], xdata[-1]
		selector2.extents = (xmin2, xmax2)

		xmin3, xmax3 = xref[0], xref[int(len(xref) / 8)]
		selector3.extents = (xmin3, xmax3)
		xmin4, xmax4 = xref[int(7 * len(xref) / 8)], xref[-1]
		selector4.extents = (xmin4, xmax4)

	plt.show()

	line_selections = [
		selector1.extents,
		selector2.extents,
		selector3.extents,
		selector4.extents
	]

	return line_selections


# noinspection PyTupleAssignmentBalance
def firs_wavelength_cal(sample_int_spectrum, wavelims=(10818, 10858)):
	"""Calculates the FIRS wavelength array from comparison to the FTS atlas.

	2023-04-04: Rewritten ground up for simplicity and ease of use.

	Parameters:
	-----------
	sample_int_spectrum : array-like
		1-D Array of intensity for wavelength calibration. Normalize it first, please.
	wavelims : list
		Upper and lower wavelength bounds for FTS atlas selection.
	Returns:
	--------
	wavelength_array : array-like
		Array corresponding to sample_int_spectrum with the wavelength corresponding to each point.
	"""

	fts_w, fts_i = _fts_window(wavelims[0], wavelims[1])

	print("Select recognizable, gaussian-esque lines from your FIRS spectrum and the reference FTS spectrum")
	line_exts = select_spec_region(sample_int_spectrum, fts_i)

	line1_exts = line_exts[0]
	line2_exts = line_exts[1]

	line1_fts = line_exts[2]
	line2_fts = line_exts[3]

	line1_firs_i = sample_int_spectrum[int(line1_exts[0]):int(line1_exts[1])]
	line2_firs_i = sample_int_spectrum[int(line2_exts[0]):int(line2_exts[1])]

	line1_fts_i = fts_i[int(line1_fts[0]):int(line1_fts[1])]
	line2_fts_i = fts_i[int(line2_fts[0]):int(line2_fts[1])]

	line1_firs_fit, _ = curve_fit(
		_gaussian,
		np.arange(len(line1_firs_i)),
		line1_firs_i,
		p0=[
			line1_firs_i.min(),
			len(line1_firs_i) / 2,
			len(line1_firs_i) / 4,
			line1_firs_i[0]]
	)

	line2_firs_fit, _ = curve_fit(
		_gaussian,
		np.arange(len(line2_firs_i)),
		line2_firs_i,
		p0=[
			line2_firs_i.min(),
			len(line2_firs_i) / 2,
			len(line2_firs_i) / 4,
			line2_firs_i[0]]
	)

	line1_fts_fit, _ = curve_fit(
		_gaussian,
		np.arange(len(line1_fts_i)),
		line1_fts_i,
		p0=[
			line1_fts_i.min(),
			len(line1_fts_i) / 2,
			len(line1_fts_i) / 4,
			line1_fts_i[0]]
	)

	line2_fts_fit, _ = curve_fit(
		_gaussian,
		np.arange(len(line2_fts_i)),
		line2_fts_i,
		p0=[
			line2_fts_i.min(),
			len(line2_fts_i) / 2,
			len(line2_fts_i) / 4,
			line2_fts_i[0]]
	)

	line1_fts_wvl = fts_w[int(line1_fts[0]) + int(line1_fts_fit[1])]
	line2_fts_wvl = fts_w[int(line2_fts[0]) + int(line2_fts_fit[1])]

	line1_firs_center = int(line1_exts[0]) + int(line1_firs_fit[1])
	line2_firs_center = int(line2_exts[0]) + int(line2_firs_fit[1])

	angstrom_per_pixel = np.abs(line2_fts_wvl - line1_fts_wvl) / np.abs(line2_firs_center - line1_firs_center)

	zero_wvl = line1_fts_wvl - (angstrom_per_pixel * line1_firs_center)

	wavelength_array = (np.arange(0, len(sample_int_spectrum)) * angstrom_per_pixel) + zero_wvl

	return wavelength_array


# noinspection PyTupleAssignmentBalance
def firs_wavelength_cal_poly(sample_int_spectrum, wavelims=(10818, 10858), plot=True):
	"""Calculates the FIRS wavelength array from comparison to the FTS atlas.

	2023-04-04: Rewritten ground up for simplicity and ease of use.

	Parameters:
	-----------
	sample_int_spectrum : array-like
		1-D Array of intensity for wavelength calibration. Normalize it first, please.
	wavelims : list
		Upper and lower wavelength bounds for FTS atlas selection.
	plot : bool
		If true, allows user to select lines manually. Uses defaults otherwise
	Returns:
	--------
	wavelength_array : array-like
		Array corresponding to sample_int_spectrum with the wavelength corresponding to each point.
	"""

	fts_w, fts_i = _fts_window(wavelims[0], wavelims[1])

	if plot or (len(sample_int_spectrum) < 900):
		print("Select recognizable, line cores from your FIRS spectrum and the reference FTS spectrum")
		line_exts = select_spec_region(sample_int_spectrum, fts_i)
		line1_exts = line_exts[0]
		line2_exts = line_exts[1]

		line1_fts = line_exts[2]
		line2_fts = line_exts[3]
	else:
		xmin1, xmax1 = 203, 220
		xcen1 = np.where(sample_int_spectrum[xmin1:xmax1 + 1] == np.nanmin(sample_int_spectrum[xmin1:xmax1 + 1]))[0][0]
		line1_exts = (xmin1 + xcen1 - 6, xmin1 + xcen1 + 6)

		xmin2, xmax2 = 788, 800
		xcen2 = np.where(sample_int_spectrum[xmin2:xmax2 + 1] == np.nanmin(sample_int_spectrum[xmin2:xmax2 + 1]))[0][0]
		line2_exts = (xmin2 + xcen2 - 4, xmin2 + xcen2 + 4)

		xmin3, xmax3 = 669, 690
		line1_fts = (xmin3, xmax3)

		xmin4, xmax4 = 2392, 2410
		line2_fts = (xmin4, xmax4)

	line1_firs_i = sample_int_spectrum[int(line1_exts[0]):int(line1_exts[1])]
	line2_firs_i = sample_int_spectrum[int(line2_exts[0]):int(line2_exts[1])]

	line1_fts_i = fts_i[int(line1_fts[0]):int(line1_fts[1])]
	line2_fts_i = fts_i[int(line2_fts[0]):int(line2_fts[1])]

	line1_firs_coef = npoly.polyfit(
		np.arange(len(line1_firs_i)),
		line1_firs_i,
		2
	)

	# X-coord of a vertex of a parabola = -b/2a
	line1_firs_x = -line1_firs_coef[1]/(2 * line1_firs_coef[2])

	line2_firs_coef = npoly.polyfit(
		np.arange(len(line2_firs_i)),
		line2_firs_i,
		2
	)

	line2_firs_x = -line2_firs_coef[1]/(2 * line2_firs_coef[2])

	line1_fts_coef = npoly.polyfit(
		np.arange(len(line1_fts_i)),
		line1_fts_i,
		2
	)

	line1_fts_x = -line1_fts_coef[1]/(2 * line1_fts_coef[2])

	line2_fts_coef = npoly.polyfit(
		np.arange(len(line2_fts_i)),
		line2_fts_i,
		2
	)

	line2_fts_x = -line2_fts_coef[1]/(2 * line2_fts_coef[2])

	line1_fts_wvl = scinterp.interp1d(
		np.arange(len(fts_w[int(line1_fts[0]):int(line1_fts[1])])),
		fts_w[int(line1_fts[0]):int(line1_fts[1])],
		kind='linear'
	)(line1_fts_x)

	line2_fts_wvl = scinterp.interp1d(
		np.arange(len(fts_w[int(line2_fts[0]):int(line2_fts[1])])),
		fts_w[int(line2_fts[0]):int(line2_fts[1])],
		kind='linear'
	)(line2_fts_x)

	line1_firs_center = line1_exts[0] + line1_firs_x
	line2_firs_center = line2_exts[0] + line2_firs_x

	angstrom_per_pixel = np.abs(line2_fts_wvl - line1_fts_wvl) / np.abs(line2_firs_center - line1_firs_center)

	zero_wvl = line1_fts_wvl - (angstrom_per_pixel * line1_firs_center)

	wavelength_array = (np.arange(0, len(sample_int_spectrum)) * angstrom_per_pixel) + zero_wvl

	return wavelength_array


def linear_spectral_tilt_correction(wave, spec):
	"""Subtracts a 1st order polynomial from given wave and spec.

	Parameters
	----------
	wave : array-like
		Wavelength grid
	spec : array-like
		Corresponding spectrum

	Returns
	-------
	tilt_corrected : array-like
		Spectrum corrected for tilt
	"""

	coefs = npoly.Polynomial.fit(wave, spec, 1).convert().coef
	fit_line = wave * coefs[1] + coefs[0]

	tilt_corrected = spec - fit_line
	return tilt_corrected


def firs_prefilter_correction(firs_data, wavelength_array, degrade_to=50, rolling_window=8, return_pfcs=False):
	"""Applies a prefilter correction to FIRS intensity data, then correct QUV channels for I.
	Pre-filter is determined along the slit by dividing the observed spectrum by the FTS spectrum,
	degrading the dividend to n points, then taking the rolling median of the degraded spectrum.

	2023-03-29: Adding correction for spectral tilt via linear fit to QUV

	2023-04-04: Rather that creating a pfc for each x,y in a 4D cube, we now average the n_slits/8 brightest slit
		positions to create an average slit. Then we create a pfc for each spectrum along the slit. This is applied to
		each slit position.

	Parameters:
	-----------
	firs_data : array-like
		Either a 4d FIRS data cube or a 2d array containing a single IQUV spectrum
	wavelength_array : array-like
		Array of wavelengths corresponding to the FIRS spectrum
	degrade_to : int, default 50
		Number of wavelength points to degrade the spectrum to for smoothing
	rolling_window : int, default 16
		Width of window for rolling median
	
	Returns:
	--------
	firs_data_corr : array-like
		An array of the same shape as the input firs_data, containing the prefilter corrected I,
		as well as QUV divided by the pre-filter corrected I
	slit_pfc : array-like, optional
		An array of the calculated prefilter corrections. For a 4D data cube, this is for the average slit.
	"""

	fts_wave, fts_spec = _fts_window(wavelength_array[0], wavelength_array[-1])
	fts_spec_in_firs_resolution = scinterp.interp1d(fts_wave, fts_spec)(wavelength_array)
	degrade_wave = np.linspace(wavelength_array[0], wavelength_array[-1], num=degrade_to)
	firs_data_corr = np.zeros(firs_data.shape)
	if len(firs_data.shape) == 2:
		if firs_data.shape[0] == 4:
			divided = firs_data[0, :] / fts_spec_in_firs_resolution
		else:
			divided = firs_data[:, 0] / fts_spec_in_firs_resolution
		firsfts_interp = scinterp.interp1d(
				wavelength_array,
				divided)(degrade_wave)
		pfc = scinterp.interp1d(
				degrade_wave,
				_rolling_median(firsfts_interp, rolling_window))(wavelength_array)
		pfc = pfc / np.nanmax(pfc)
		if firs_data.shape[0] == 4:
			firs_data_corr[0, :] = firs_data[0, :] / pfc
			firs_data_corr[1, :] = firs_data[1, :] / firs_data_corr[0, :]
			firs_data_corr[2, :] = firs_data[2, :] / firs_data_corr[0, :]
			firs_data_corr[4, :] = firs_data[3, :] / firs_data_corr[0, :]
			slit_pfc = pfc
		else:
			firs_data_corr[:, 0] = firs_data[:, 0] / pfc
			firs_data_corr[:, 1] = firs_data[:, 1] / firs_data_corr[:, 0]
			firs_data_corr[:, 2] = firs_data[:, 2] / firs_data_corr[:, 0]
			firs_data_corr[:, 3] = firs_data[:, 3] / firs_data_corr[:, 0]
			slit_pfc = pfc
	else:
		# Creating a pfc for each slit position is quite wasteful.
		# Instead, we'll take the mean of the n_slits/8 brightest slits, and create the pfc from that.
		if firs_data.shape[0] < 16:
			mean_slit = np.nanmean(firs_data[:, 0, :, :], axis=0)
		else:
			slit_sums = np.nansum(firs_data[:, 0, :, :], axis=(1, 2))
			slit_brightness_argsort = np.argsort(slit_sums)
			n_brightest = int(firs_data.shape[0]/8)
			bright_args = slit_brightness_argsort[-n_brightest:]
			mean_slit = np.zeros((firs_data.shape[2], firs_data.shape[3]))
			for i in bright_args:
				mean_slit += firs_data[i, 0, :, :]
			mean_slit = mean_slit / len(bright_args)

		slit_pfc = np.zeros(mean_slit.shape)
		for i in range(mean_slit.shape[0]):
			divided = mean_slit[i, :] / fts_spec_in_firs_resolution
			firsfts_interp = scinterp.interp1d(
				wavelength_array,
				divided)(degrade_wave)
			pfc = scinterp.interp1d(
				degrade_wave,
				_rolling_median(firsfts_interp, rolling_window))(wavelength_array)
			pfc = pfc / np.nanmax(pfc)
			slit_pfc[i, :] = pfc

		for i in range(firs_data.shape[0]):
			for j in range(firs_data.shape[2]):
				firs_data_corr[i, 0, j, :] = firs_data[i, 0, j, :] / slit_pfc[j, :]
				qtmp = firs_data[i, 1, j, :] / firs_data_corr[i, 0, j, :]
				firs_data_corr[i, 1, j, :] = linear_spectral_tilt_correction(wavelength_array, qtmp)
				utmp = firs_data[i, 2, j, :] / firs_data_corr[i, 0, j, :]
				firs_data_corr[i, 2, j, :] = linear_spectral_tilt_correction(wavelength_array, utmp)
				vtmp = firs_data[i, 3, j, :] / firs_data_corr[i, 0, j, :]
				firs_data_corr[i, 3, j, :] = linear_spectral_tilt_correction(wavelength_array, vtmp)
	if return_pfcs:
		return firs_data_corr, slit_pfc
	else:
		return firs_data_corr


def firs_fringe_template(flat_dat_file, lopass_cutoff=0.4, plot=True):
	"""Creating templates of fringes in QUV.
	We do this with a flat file from the same day that's gone through the FIRS reduction pipeline.
	This flat is then wavelength-calibrated and prefilter-corrected using the above suite of functions.
	Once corrected for prefilter, the rolling median is taken for each position along the averaged slit.
	This gets rid of hot pixels. Then, a linear fit is performed for each position along the averaged slit to remove the
	spectral tilt. Now corrected for prefilter, hot pixels, and tilt, the slit-position averaged flat field is subjected
	to a Fourier low-pass filter with a cutoff of 1/lopass_cutoff angstroms (i.e.,periodicities longer than
	lopass_cutoff angstroms). This provides an image of the fringes. This fringe image is returned, and can be
	subtracted from QUV data that have been prefilter and tilt corrected.

	Parameters:
	-----------
	flat_dat_file : str
		Filename of the binary .dat file corresponding to a flat field that's been through the FIRS reduction pipeline.
	flat_sav_file : str
		Filename of the IDL .sav file containing the metadata of flat_dat_file
	lopass_cutoff : float, optional
		The frequency cutoff for making the low-pass Fourier fringe filter.
	plot : bool
		If true allows user to select spectral lines for wavelength calibration

	Returns:
	--------
	quv_fringe_image : array-like
		Array of shape (3,ny,nlambda) containing the fringe images for Stokes QUV. Subtract from the slit image to
		correct for fringes.
	"""

	flat_map = read_firs(flat_dat_file)
	wavelength_array = firs_wavelength_cal_poly(np.nanmean(flat_map[:, 0, 100:400, :], axis=(0, 1)), plot=plot)
	flat_map = np.nanmean(firs_prefilter_correction(flat_map, wavelength_array), axis=0)
	fftfreqs = np.fft.fftfreq(len(wavelength_array), wavelength_array[1]-wavelength_array[0])

	ft_cut1 = fftfreqs >= lopass_cutoff
	ft_cut2 = fftfreqs <= -lopass_cutoff
	quv_fringe_image = np.zeros((3, flat_map.shape[1], flat_map.shape[2]))
	for i in range(flat_map.shape[1]):
		for j in range(1, 4):
			quv_ft = np.fft.fft(_rolling_median(flat_map[j, i, :], 16))
			quv_ft[ft_cut1] = 0
			quv_ft[ft_cut2] = 0
			quv_fringe_image[j-1, i, :] = np.real(np.fft.ifft(quv_ft))
	return wavelength_array, quv_fringe_image


def firs_fringecorr(map_data, map_waves, flat_data_file, lopass=0.5, plot=True):
	"""Applies fringe correction to FIRS map by calling the fringe template function, determining any difference in the
	wavelength regimes, and any offset in the spectrum.

	Parameters:
	-----------
	map_data : array-like
		The 4d firs image cube
	map_waves : array-like
		The corresponding spectral axis
	flat_data_file : str
		Path to a flat file that has been through the FIRS pipeline
	lopass : float, optional
		The frequency cutoff to be passed to the fringe template function
	plot : bool
		If true passes kwarg to enable selection and plotting of spectral lines

	Returns:
	--------
	fringe_corrected_map : array-like
		A map that has been (hopefully) corrected for spectral fringeing.

	"""

	flat_waves, fringe_template = firs_fringe_template(flat_data_file, lopass_cutoff=lopass, plot=plot)
	if len(flat_waves) != len(map_waves):
		fringe_template = scinterp.interp1d(
			flat_waves,
			fringe_template,
			axis=-1,
			bounds_error=False,
			fill_value='extrapolate'
		)(map_waves)

	fringe_corrected_map = np.zeros(map_data.shape)

	fringe_corrected_map[:, 0, :, :] = map_data[:, 0, :, :]

	for i in range(fringe_corrected_map.shape[0]):
		for j in range(3):
			for k in range(fringe_corrected_map.shape[2]):
				map_med = np.nanmedian(map_data[i, j+1, k, :50])
				fringe_med = np.nanmedian(fringe_template[j, k, :50])

				corr_factor = fringe_med - map_med

				fringe_corr = fringe_template[j, k, :] - corr_factor

				fringe_corrected_map[i, j+1, k, :] = map_data[i, j+1, k, :] - fringe_corr

	return fringe_corrected_map


def firs_coordinate_conversion(raw_file):
	"""Converts telescope Stonyhurst to Helioprojective Coordinates.
	I should be able to do this with the Alt-Az coordinates in the sav file, but it doesn't appear to work out.
	Instead, we need a raw firs file, as this contains the DST_SLAT and DST_SLNG keywords.

	Parameters:
	-----------
	raw_file : str
		Path to a raw firs fits file

	Returns:
	--------
	helio_coord : SkyCoord object
		astropy skycoord object containing the Helioprojective coordinates of the observation series
	rotation_angle : float
		Guider angle minus the 13.3 degree offset
	date: str
		String with the date from header. Used elsewhere.
	"""

	raw_hdr = fits.open(raw_file)[0].header
	stony_lat = raw_hdr['DST_SLAT']
	stony_lon = raw_hdr['DST_SLNG']
	rotation_angle = raw_hdr['DST_GDRN'] - 13.3  # 13.3 is the offset of the DST guider head to solar north
	# There may still be 90 degree rotations, or other translations
	obstime = raw_hdr['OBS_STAR']
	date = raw_hdr['DATE_OBS'].replace('/', '-')
	stony_coord = SkyCoord(
		stony_lon*u.deg,
		stony_lat*u.deg,
		frame=frames.HeliographicStonyhurst,
		observer='earth',
		obstime=obstime
	)

	helio_coord = stony_coord.transform_to(frames.Helioprojective)
	return helio_coord, rotation_angle, date


# noinspection PyTypeChecker
def firs_construct_hdu(firs_data, firs_lambda, meta_file, coordinates, rotation, date, dx, dy, exptime, coadd):
	"""Helper function that constructs HDUList for packaging to a final level 1.5 data product
	Parameters:
	-----------
	firs_data : array-like
		The fringe, tilt, prefilter corrected FIRS data 4d object
	firs_lambda : array-like
		The array of wavelengths for FIRS
	meta_file : str
		The .sav file containing FIRS metadata
	coordinates : astropy.coordinates.SkyCoord object
		SkyCoord object with XCEN and YCEN
	rotation : float
		Rotation angle relative to solar-north
	date : str
		str containing the obs date (no time)
	dx : float
		dx element in arcsec
	dy : float
		dy element in arcsec
	exptime : float
		time for single exposure
	coadd : float
		number of coadds to determine total exptime.

	Returns:
	--------
	hdulist : astropy.io.fits.HDUList object
		Nicely formatted HDUList for writing to disk
	"""

	meta_info = readsav(meta_file)
	t0 = np.datetime64(date) + np.timedelta64(
		int(1000 * 60 * 60 * meta_info['ttime'][0]), 'ms'
	)
	t1 = np.datetime64(date) + np.timedelta64(
		int(1000 * 60 * 60 * meta_info['ttime'][-1]), 'ms'
	)

	utc_offset = _correct_datetimes(t0)

	t0 += utc_offset
	t1 += utc_offset

	ext0 = fits.PrimaryHDU()
	ext0.header['DATE'] = (np.datetime64('now').astype(str), 'File created')
	ext0.header['TELESCOP'] = 'DST'
	ext0.header['INSTRUME'] = 'FIRS'
	ext0.header['DATA_LEV'] = 1.5
	ext0.header['DATE_OBS'] = t0.astype(str)
	ext0.header['STARTOBS'] = t0.astype(str)
	ext0.header['DATE_END'] = t1.astype(str)
	ext0.header['ENDOBS'] = t1.astype(str)
	ext0.header['BTYPE'] = 'Intensity'
	ext0.header['BUNIT'] = 'Corrected DN'
	ext0.header['FOVX'] = (firs_data.shape[0] * dx, 'arcsec')
	ext0.header['FOVY'] = (firs_data.shape[2] * dy, 'arcsec')
	ext0.header['XCEN'] = (coordinates.Tx.value, 'arcsec')
	ext0.header['YCEN'] = (coordinates.Ty.value, 'arcsec')
	ext0.header['ROT'] = rotation
	ext0.header['EXPTIME'] = (exptime, 'ms per coadd')
	ext0.header['XPOSUR'] = (exptime*coadd, 'ms')
	ext0.header['NSUMEXP'] = (coadd, 'coadds')
	ext0.header['PRSTEP1'] = ('DARK-SUBTRACTION,FLATFIELDING', "FIRS Calibration Pipeline (C.Beck)")
	ext0.header['PRSTEP2'] = ('POLARIZATION-CALIBRATION', 'FIRS Calibration Pipeline (C.Beck)')
	ext0.header['PRSTEP3'] = ('PREFILTER,FRINGE-CORRECTIONS', 'firs-tools (S.Sellers)')
	ext0.header['PRSTEP4'] = ('WAVELENGTH-CALIBRATION', 'firs-tools (S.Sellers)')

	ext1 = fits.ImageHDU(np.flipud(np.rot90(firs_data[:, 0, :, :])))
	ext1.header['EXTNAME'] = 'Stokes-I'
	ext1.header['CDELT1'] = (dx, 'arcsec')
	ext1.header['CDELT2'] = (dy, 'arcsec')
	ext1.header['CDELT3'] = (firs_lambda[1] - firs_lambda[0], 'Angstom')
	ext1.header['CTYPE1'] = 'HPLT-TAN'
	ext1.header['CTYPE2'] = 'HPLT-TAN'
	ext1.header['CTYPE3'] = 'WAVE'
	ext1.header['CUNIT1'] = 'arcsec'
	ext1.header['CUNIT2'] = 'arcsec'
	ext1.header['CUNIT3'] = 'Angstrom'
	ext1.header['CRVAL1'] = coordinates.Tx.value
	ext1.header['CRVAL2'] = coordinates.Ty.value
	ext1.header['CRVAL3'] = firs_lambda[0]
	ext1.header['CRPIX1'] = firs_data.shape[0]/2
	ext1.header['CRPIX2'] = firs_data.shape[2]/2
	ext1.header['CRPIX3'] = 1
	ext1.header['CROTAN'] = rotation

	ext2 = fits.ImageHDU(np.flipud(np.rot90(firs_data[:, 1, :, :])))
	ext2.header['EXTNAME'] = 'Stokes-Q'
	ext2.header['CDELT1'] = (dx, 'arcsec')
	ext2.header['CDELT2'] = (dy, 'arcsec')
	ext2.header['CDELT3'] = (firs_lambda[1] - firs_lambda[0], 'Angstom')
	ext2.header['CTYPE1'] = 'HPLT-TAN'
	ext2.header['CTYPE2'] = 'HPLT-TAN'
	ext2.header['CTYPE3'] = 'WAVE'
	ext2.header['CUNIT1'] = 'arcsec'
	ext2.header['CUNIT2'] = 'arcsec'
	ext2.header['CUNIT3'] = 'Angstrom'
	ext2.header['CRVAL1'] = coordinates.Tx.value
	ext2.header['CRVAL2'] = coordinates.Ty.value
	ext2.header['CRVAL3'] = firs_lambda[0]
	ext2.header['CRPIX1'] = firs_data.shape[0] / 2
	ext2.header['CRPIX2'] = firs_data.shape[2] / 2
	ext2.header['CRPIX3'] = 1
	ext2.header['CROTAN'] = rotation

	ext3 = fits.ImageHDU(np.flipud(np.rot90(firs_data[:, 2, :, :])))
	ext3.header['EXTNAME'] = 'Stokes-U'
	ext3.header['CDELT1'] = (dx, 'arcsec')
	ext3.header['CDELT2'] = (dy, 'arcsec')
	ext3.header['CDELT3'] = (firs_lambda[1] - firs_lambda[0], 'Angstom')
	ext3.header['CTYPE1'] = 'HPLT-TAN'
	ext3.header['CTYPE2'] = 'HPLT-TAN'
	ext3.header['CTYPE3'] = 'WAVE'
	ext3.header['CUNIT1'] = 'arcsec'
	ext3.header['CUNIT2'] = 'arcsec'
	ext3.header['CUNIT3'] = 'Angstrom'
	ext3.header['CRVAL1'] = coordinates.Tx.value
	ext3.header['CRVAL2'] = coordinates.Ty.value
	ext3.header['CRVAL3'] = firs_lambda[0]
	ext3.header['CRPIX1'] = firs_data.shape[0] / 2
	ext3.header['CRPIX2'] = firs_data.shape[2] / 2
	ext3.header['CRPIX3'] = 1
	ext3.header['CROTAN'] = rotation

	ext4 = fits.ImageHDU(np.flipud(np.rot90(firs_data[:, 3, :, :])))
	ext4.header['EXTNAME'] = 'Stokes-V'
	ext4.header['CDELT1'] = (dx, 'arcsec')
	ext4.header['CDELT2'] = (dy, 'arcsec')
	ext4.header['CDELT3'] = (firs_lambda[1] - firs_lambda[0], 'Angstom')
	ext4.header['CTYPE1'] = 'HPLT-TAN'
	ext4.header['CTYPE2'] = 'HPLT-TAN'
	ext4.header['CTYPE3'] = 'WAVE'
	ext4.header['CUNIT1'] = 'arcsec'
	ext4.header['CUNIT2'] = 'arcsec'
	ext4.header['CUNIT3'] = 'Angstrom'
	ext4.header['CRVAL1'] = coordinates.Tx.value
	ext4.header['CRVAL2'] = coordinates.Ty.value
	ext4.header['CRVAL3'] = firs_lambda[0]
	ext4.header['CRPIX1'] = firs_data.shape[0] / 2
	ext4.header['CRPIX2'] = firs_data.shape[2] / 2
	ext4.header['CRPIX3'] = 1
	ext4.header['CROTAN'] = rotation

	ext5 = fits.ImageHDU(firs_lambda)
	ext5.header['EXTNAME'] = 'lambda-coordinate'
	ext5.header['BTYPE'] = 'lambda axis'
	ext5.header['BUNIT'] = '[AA]'

	ext6 = fits.ImageHDU(60*60*meta_info['ttime'] + utc_offset.astype(float))
	ext6.header['EXTNAME'] = 'time-coordinate'
	ext6.header['BTYPE'] = 'time axis'
	ext6.header['BUNIT'] = '[s]'

	hdulist = fits.HDUList([ext0, ext1, ext2, ext3, ext4, ext5, ext6])

	return hdulist


def firs_to_fits(firs_map_fname, flat_map_fname, raw_file, outname, dx=0.3, dy=0.15, exptime=125, coadd=10, plot=False):
	"""This function converts FIRS .dat files to level 1.5 fits files with a wavelength array, time array, and corrected
	for fringeing. You will require a map containing a flat field that has been processed as a science map by the FIRS
	IDL pipeline.

	Parameters:
	-----------
	firs_map_fname : str
		Path to a FIRS science map at level-1 (post-IDL)
	flat_map_fname : str
		Path to a FIRS flat map processed by the IDL pipeline as a science map
	raw_file : str
		Path to a raw FIRS fits file (for header information)
	outname : str
		The pattern used to save the FIRS map as a fits file.
	dx : float
		From the slit spacing, default to 0.3 arcsec/slit_pos
	dy : float
		From math, default to 0.15 arcsec/pix
	exptime : float
		Exposure time, default to 125 ms
	coadd : float
		Number of coadds, default to 10 for standard obs modes.
	plot : bool
		If true, allows the user to confirm selections of spectral lines used in wavelength cal

	Returns:
	--------
	None, but it writes a fits file.
	"""

	# L1 data
	firs_data = read_firs(firs_map_fname)

	# Wave Cal
	firs_waves = firs_wavelength_cal_poly(
		np.nanmean(firs_data[:, 0, 100:400, :], axis=(0, 1)),
		plot=plot
	)

	# Prefilter Cal
	firs_data = firs_prefilter_correction(firs_data, firs_waves)

	# Fringe Cal
	firs_data = firs_fringecorr(firs_data, firs_waves, flat_map_fname)

	coordinates, crotan, date = firs_coordinate_conversion(raw_file)

	hdulist = firs_construct_hdu(
		firs_data,
		firs_waves,
		firs_map_fname + '.sav',
		coordinates,
		crotan,
		date,
		dx,
		dy,
		exptime,
		coadd
	)

	hdulist.writeto(outname)


def repackHazel(
		h5File, initFile, fitsFile, saveName,
		nx=None, ny=None,
		ch_key='ch1', ph_key='ph1', sp_key='he',
		translation=False,
):
	"""Master function to repack h5 output from Hazel2 code to level-2 fits file for archiving and distribution.
	Currently defaults save only the final cycle of the inversion, and the 0th randomization index.

	Parameters:
	h5File : string
		Path to h5 Hazel code output.
	initFile : string
		Path to pre-Hazel h5 file.
	fitsFile : string
		Path to level-1.5 FIRS fits file. Used to construct headers.
	saveName : string
		Name for final fits file
	nx : int or None (default)
		By default uses the header in fitsFile to determine nx.
		User can provide nx if they fit a specific subfield
	ny : int or None (default)
		Same as nx
	ch_key : str or list
		Key or list of keys of chromospheres used in Hazel inversions.
		Creates a new fits extension for each key provided
	ph_key : str
		Key of photosphere (singular) used in Hazel inversion.
		Creates a fits extension for the photosphere
	sp_key : str
		Key of the spectral extension in the h5 file.
		Creates 8 spectral extensions; IQUV for each data and inversion
	translation : bool
		Legacy option from before the orientation was corrected in firs-tools level1.5 conversion.
		If true, rotates the final maps 90 degrees and flips it along the vertical axis.
	"""
	fitsFile = fits.open(fitsFile)
	dx = fitsFile[1].header['CDELT1']
	dy = fitsFile[1].header['CDELT2']

	initFile = h5py.File(initFile, "r")

	h5File = h5py.File(h5File, 'r')
	if not nx:
		nx = fitsFile[1].header['NAXIS3'] - 1
	if not ny:
		ny = fitsFile[1].header['NAXIS2'] - 1
	if type(ch_key) == str:
		ch_key = [ch_key]

	wavelength = h5File[sp_key]['wavelength'][:]

	chParams = [
		'Bx', 'Bx_err',
		'By', 'By_err',
		'Bz', 'Bz_err',
		'a', 'a_err',
		'beta', 'beta_err',
		'deltav', 'deltav_err',
		'ff', 'ff_err',
		'tau', 'tau_err',
		'v', 'v_err'
	]
	chParamUnits = [
		'Gauss', 'Gauss',
		'Gauss', 'Gauss',
		'Gauss', 'Gauss',
		'Damping', 'Damping',
		'FillFactor', 'FillFactor',
		'log10(OpticalDepth)', 'log10(OpticalDepth)',
		'km/s', 'km/s'
	]

	phParams = [
		'Bx', 'Bx_err',
		'By', 'By_err',
		'Bz', 'Bz_err',
		'T', 'T_err',
		'ff', 'ff_err',
		'v', 'v_err',
		'vmac', 'vmac_err',
		'vmic', 'vmic_err'
	]
	phParamUnits = [
		'Gauss', 'Gauss',
		'Gauss', 'Gauss',
		'Gauss', 'Gauss',
		'Kelvin', 'Kelvin',
		'FillFactor', 'FillFactor',
		'km/s', 'km/s',
		'km/s', 'km/s'
	]

	prstepFlags = [
		'PRSTEP1', 'PRSTEP2', 'PRSTEP3', 'PRSTEP4', 'PRSTEP5'
	]
	prstepValues = [
		'DARK-SUBTRACTION,FLATFIELDING',
		'POLARIZATION-CALIBRATION',
		'FRINGE-CORRECTION,PREFILTER-CORRECTION',
		'NORMALIZATION,WAVELENGTH-CALIBRATION',
		'HAZEL2-INVERSION'
	]
	prstepComments = [
		'FIRS Calibration Pipeline (C.Beck)',
		'FIRS Calibration Pipeline (C.Beck)',
		'firs-tools (S.Sellers)',
		'firs-tools (S.Sellers)',
		'firs-tools (S.Sellers)',
		'Hazel2 (A. Asensio Ramos)'
	]

	fitsHDUs = []

	# Start with 0th hduList extension, information only:
	ext0 = fits.PrimaryHDU()
	ext0.header = fitsFile[0].header
	ext0.header['DATE'] = (np.datetime64('now').astype(str), 'File created')
	ext0.header['DATA_LEV'] = 2
	# If the number of raster positions matches the number in the original file,
	# Then the start and end times are valid, and can be put in the file.
	# Otherwise, we have no way of knowing what the start/end times are.
	# This will be fixed when the Hazel prep routines are fixed and incorporated into firs-tools.
	if translation:
		if nx != fitsFile[1].header['NAXIS3'] - 1:
			del ext0.header['STARTOBS']
			del ext0.header['DATE_END']
			del ext0.header['ENDOBS']
	else:
		if ny != fitsFile[1].header['NAXIS2'] - 1:
			del ext0.header['STARTOBS']
			del ext0.header['DATE_END']
			del ext0.header['ENDOBS']
	del ext0.header['BTYPE']
	del ext0.header['BUNIT']
	ext0.header['FOVX'] = (nx * dx, 'arcsec')
	ext0.header['FOVY'] = (ny * dy, 'arcsec')
	del ext0.header['EXPTIME']
	del ext0.header['XPOSUR']
	del ext0.header['NSUMEXP']
	for i in range(len(prstepFlags)):
		ext0.header[prstepFlags[i]] = (prstepValues[i], prstepComments[i])

	fitsHDUs.append(ext0)

	# Now we pack our chromosphere results:
	for key in ch_key:
		chromosphere = h5File[key]
		for i in range(len(chParams)):
			columns = []
			if 'err' in chParams[i]:
				paramArray = chromosphere[chParams[i]][:, 0, -1].reshape(nx, ny)
			else:
				paramArray = chromosphere[chParams[i]][:, 0, -1, 0].reshape(nx, ny)
			if translation:
				paramArray = np.flipud(np.rot90(paramArray))
			paramArray = paramArray.reshape(1, nx, ny)
			columns.append(
				fits.Column(
					name=chParams[i],
					format=str(int(nx*ny))+'D',
					dim='('+str(nx)+","+str(ny)+")",
					unit=chParamUnits[i],
					array=paramArray
				)
			)
		fitsHDUs.append(
			fits.BinTableHDU.from_columns(
				columns
			)
		)
	# Now we pack our photospheric results.
	# Unlike the chromospheres, there's an additional axis, the height profile.
	# We'll use this profile as the length of each column in the fits table.

	for i in range(len(phParams)):
		photosphere = h5File[ph_key]
		columns = []
		logTau = photosphere['log_tau'][:]
		columns.append(
			fits.Column(
				name='logTau',
				format='D',
				unit='Optical Depth',
				array=logTau
			)
		)
		if 'err' in phParams[i]:
			"""
			Errors are not straightforward for the photosphere.
			There are several ways the errors are recorded:
				1.) Multiple nodes are fit. The error is an object array of errors at each fit node.
					e.g., 5 nodes fit, the error array is an array of shape nx, ny. 
					Each element of the error array is either:
						~A zero length array (could not fit the pixel)
						~An array of errors with a length equal to the number of nodes.
				2.) A single node is fit. This is the simplest example of case 1 above.
					Here, each element is either length zero or one.
				3.) The parameter is not fit. Here, when the fit succeeds, the element is a 1-array with a nan.
					This is vexatious. 
			So here's what we do:
				1.) Retrieve the list of nodes used in fitting. These are presented the same way as the errors.
					So, if the fit failed, the element is an empty list.
					So we check the 0th nodelist, and if it's length zero, we pull at random until it isn't.
					If the nodelist is length 1, and the element within it is a nan:
						~The error array is a zero-array of shape (nx, ny, log_tau)
					If the nodelist is length 1, and the element is not a nan:
						~Only one node was fit. We create an error array of the shape (nx, ny, log_tau).
						Then, fill the array by looping over the array of errors. Where there's a value,
						We duplicate that value along the log_tau axis.
						Where there's no value (fit failed), we fill with a 0 instead.
					If the nodelist is length greater than one:
						~Many nodes were fit. We create the same error array, but when filling, where 
						there's an array of values, we (linear) interpolate to the length of log_tau,
						and fill the array that way. 0s otherwise. 
			"""
			nodeArr = photosphere[phParams[i].replace("err", "nodes")][:, 0, -1].reshape(nx, ny)
			nodeList = nodeArr[0, 0]
			while len(nodeList) == 0:
				nodeList = nodeArr[
					np.random.randint(0, nx),
					np.random.randint(0, ny)
				]
			if (len(nodeList) == 1) & (np.isnan(nodeList[0])):
				columns.append(
					fits.Column(
						name=phParams[i],
						format=str(int(nx*ny))+"I",
						dim='(' + str(nx) + "," + str(ny) + ")",
						unit=phParamUnits[i],
						array=np.zeros((len(logTau), nx, ny))
					)
				)
			elif (len(nodeList) == 1) & (not np.isnan(nodeList[0])):
				dummy_err = np.zeros((len(logTau), nx, ny))
				err = photosphere[phParams[i]][:, 0, -1].reshape(nx, ny)
				for x in range(err.shape[0]):
					for y in range(err.shape[1]):
						if len(err[x, y]) != 0:
							dummy_err[:, x, y] = err[x, y]
				columns.append(
					fits.Column(
						name=phParams[i],
						format=str(int(nx*ny))+"D",
						dim='(' + str(nx) + "," + str(ny) + ")",
						unit=phParamUnits[i],
						array=dummy_err
					)
				)
			elif len(nodeList) > 1:
				dummy_err = np.zeros((len(logTau), nx, ny))
				err = photosphere[phParams[i]][:, 0, -1].reshape(nx, ny)
				for x in range(err.shape[0]):
					for y in range(err.shape[1]):
						if len(err[x, y]) != 0:
							dummy_err[:, x, y] = scinterp.interp1d(
								nodeList,
								err[x, y],
								kind='linear'
							)(np.arange(len(logTau)))
				columns.append(
					fits.Column(
						name=phParams[i],
						format=str(int(nx * ny)) + "D",
						dim='(' + str(nx) + "," + str(ny) + ")",
						unit=phParamUnits[i],
						array=dummy_err
					)
				)
		else:
			columns.append(
				fits.Column(
					name=phParams[i],
					format=str(int(nx * ny)) + "D",
					dim='(' + str(nx) + "," + str(ny) + ")",
					unit=phParamUnits[i],
					array=np.transpose(
						photosphere[phParams[i]][:, 0, -1, :].reshape(nx, ny, len(logTau)),
						(2, 0, 1)
					)
				)
			)
	fitsHDUs.append(
		fits.BinTableHDU.from_columns(
			columns
		)
	)

	# After an eternity, we can finally move on to doing the extensions that have data in them.
	# First we do the IQUV for the pre-fit data. Then the synthetic profiles.

	stks = ['I', 'Q', 'U', 'V']

	for i in range(4):
		realStokes = initFile['stokes'][:, :, i]
		realStokes = realStokes.reshape(nx, ny, realStokes.shape[1])
		if translation:
			realStokes = np.flipud(np.rot90(realStokes))
		ext = fits.ImageHDU(realStokes)
		ext.header['EXTNAME'] = ('Stokes-'+stks[i], "Normalized by Quiet Sun, Corrected for position angle")
		ext.header['CDELT1'] = (dx, 'arcsec')
		ext.header['CDELT2'] = (dy, 'arcsec')
		ext.header['CDELT3'] = fitsFile[1].header['CDELT3']
		ext.header['CTYPE1'] = 'HPLT-TAN'
		ext.header['CTYPE2'] = 'HPLT-TAN'
		ext.header['CTYPE3'] = 'WAVE'
		ext.header['CUNIT1'] = 'arcsec'
		ext.header['CUNIT2'] = 'arcsec'
		ext.header['CUNIT3'] = 'Angstrom'
		ext.header['CRVAL1'] = fitsFile[1].header['CRVAL1']
		ext.header['CRVAL2'] = fitsFile[1].header['CRVAL2']
		ext.header['CRVAL3'] = wavelength[0]
		ext.header['CRPIX1'] = nx / 2
		ext.header['CRPIX2'] = ny / 2
		ext.header['CRPIX3'] = 1
		ext.header['CROTAN'] = fitsFile[1].header['CROTAN']
		fitsHDUs.append(ext)

	# Finally, we can do the synthetic profiles...
	for i in range(4):
		synthStokes = h5File[sp_key]['stokes'][:, 0, -1, i, :]
		synthStokes = synthStokes.reshape(nx, ny, synthStokes.shape[1])
		if translation:
			synthStokes = np.flipud(np.rot90(synthStokes))
		ext = fits.ImageHDU(synthStokes)
		ext.header['EXTNAME'] = ('SYNTHETICStokes-' + stks[i], "Synthetic from Hazel2 Inversion")
		ext.header['CDELT1'] = (dx, 'arcsec')
		ext.header['CDELT2'] = (dy, 'arcsec')
		ext.header['CDELT3'] = fitsFile[1].header['CDELT3']
		ext.header['CTYPE1'] = 'HPLT-TAN'
		ext.header['CTYPE2'] = 'HPLT-TAN'
		ext.header['CTYPE3'] = 'WAVE'
		ext.header['CUNIT1'] = 'arcsec'
		ext.header['CUNIT2'] = 'arcsec'
		ext.header['CUNIT3'] = 'Angstrom'
		ext.header['CRVAL1'] = fitsFile[1].header['CRVAL1']
		ext.header['CRVAL2'] = fitsFile[1].header['CRVAL2']
		ext.header['CRVAL3'] = wavelength[0]
		ext.header['CRPIX1'] = nx / 2
		ext.header['CRPIX2'] = ny / 2
		ext.header['CRPIX3'] = 1
		ext.header['CROTAN'] = fitsFile[1].header['CROTAN']
		fitsHDUs.append(ext)

	# And the chisq map
	chi2 = h5File[sp_key]['chi2'][:, 0, -1].reshape(nx, ny)
	ext = fits.ImageHDU(chi2)
	ext.header['EXTNAME'] = ("CHISQ", 'Fit chi-square from Hazel Inversions')
	ext.header['CDELT1'] = (dx, 'arcsec')
	ext.header['CDELT2'] = (dy, 'arcsec')
	ext.header['CTYPE1'] = 'HPLT-TAN'
	ext.header['CTYPE2'] = 'HPLT-TAN'
	ext.header['CUNIT1'] = 'arcsec'
	ext.header['CUNIT2'] = 'arcsec'
	ext.header['CRVAL1'] = fitsFile[1].header['CRVAL1']
	ext.header['CRVAL2'] = fitsFile[1].header['CRVAL2']
	ext.header['CRPIX1'] = nx / 2
	ext.header['CRPIX2'] = ny / 2
	ext.header['CROTAN'] = fitsFile[1].header['CROTAN']
	fitsHDUs.append(ext)

	# The Wavelength Array...
	ext = fits.ImageHDU(wavelength)
	ext.header['EXTNAME'] = 'lambda-coordinate'
	ext.header['BTYPE'] = 'lambda axis'
	ext.header['BUNIT'] = '[AA]'
	fitsHDUs.append(ext)

	# And the time array (only if the full X-range is used.)
	if nx == fitsFile[1].header['NAXIS3'] - 1:
		fitsHDUs.append(fitsFile[-1])

	hdulist = fits.HDUList(fitsHDUs)
	hdulist.writeto(saveName)
	return
