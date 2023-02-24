import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector,SpanSelector
from scipy.io import readsav
import scipy.interpolate as scinterp
import scipy.signal as scisig
import sean_tools as st
from scipy.optimize import curve_fit

def rolling_median(data,window):
	"""Simple rolling median function, rolling by the central value. By default preserves the edges to provide an output array of the same shape as the input.
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

def read_firs(firs_file,sav_file):
	"""Simple routine to read FIRS binary file into numpy array for ease of use
	Parameters:
	-----------
	firs_file : str
		path to FIRS *.dat binary file for readin
	sav_file : str
		path to FIRS *.dat.sav IDL save file for readin dimension
	
	Returns:
	--------
	firs_data : array-like
		Formatted nd array of FIRS data
	"""
	firs_mets = readsav(sav_file)
	firs_nspex = int(firs_mets['dx_final'] + 1)
	firs_ysize = int(firs_mets['dy_final'] + 1)

	firs = np.fromfile(firs_file,dtype = np.float32)
	firs_xsize = int(len(firs)/4/firs_nspex/firs_ysize)

	return firs.reshape((firs_xsize,4,firs_ysize,firs_nspex))

def select_callback(eclick,erelease):
	"""
	Callback for line selection, with eclick and erelease being the press and release events.
	"""

	x1,y1 = eclick.xdata,eclick.ydata
	x2,y2 = erelease.xdata,erelease.ydata

	return sorted([x1,x2]),sorted([y1,y2])

#def onselect(xmin,xmax):
#	indmin,indmax = np.searchsorted(x,(xmin,xmax))
#	indmax = min(len(x) - 1,indmax)
#	region_x = x[indmin:indmax]
#	region_y = y[indmin:indmax]

def select_image_region(img_data,xdata = None):
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
		if xdata != None:
			ax.plot(xdata,img_data,drawstyle = 'steps-mid')
		else:
			xdata = np.arange(len(img_data))
			ax.plot(img_data,drawstyle = 'steps-mid')
			ax.set_xlim(xdata[0],xdata[-1])
		def onselect(xmin,xmax):
			indmin,indmax = np.searchsorted(xdata,(xmin,xmax))
			indmax = min(len(xdata) - 1,indmax)
			region_x = xdata[indmin:indmax]
			region_y = img_data[indmin:indmax]

		ax.set_title(f"Click and drag to select a Span.\n Press t to toggle.")
		
		selector = SpanSelector(
				ax,
				onselect,
				"horizontal",
				useblit = True,
				button = [1,3],
				interactive = True,
				drag_from_anywhere = True)
		plt.show()
	elif len(img_data.shape) == 2:
		ax.imshow(img_data,origin = 'lower',aspect = 'auto',interpolation = 'none')
		
		def onselect(xmin,xmax):
			indmin,indmax = np.searchsorted(x,(xmin,xmax))
			indmax = min(len(x) - 1,indmax)
			region_x = x[indmin:indmax]
			region_y = y[indmin:indmax]

		ax.set_title(f"Click and drag to draw a {RectangleSelector.__name__}.\n Press t to toggle")
		selector = RectangleSelector(
			ax,
			select_callback,
			useblit = True,
			button = [1,3],
			spancoords = 'pixels',
			interactive = True)
		plt.show()
	
	else:
		print("Invalid image dimensions, must be 1- or 2-D array")
		return None 

	return selector.extents

def firs_wavelength_cal(sample_int_spectrum,wavelims = [10818,10858]):
	"""Calculates the FIRS wavelength array from comparison to the FTS atlas.
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

	print("Select a recognizable, gaussian-esque line from your FIRS spectrum")
	line1_exts = select_image_region(sample_int_spectrum)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(sample_int_spectrum)
	ax.axvspan(line1_exts[0],line1_exts[1],alpha = 0.5,color = 'C3')
	plt.show(block = False)
	print("Great. Do it again.")

	line2_exts = select_image_region(sample_int_spectrum)
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(sample_int_spectrum)
	ax.axvspan(line1_exts[0],line1_exts[1],alpha = 0.5, color = 'C3')
	ax.axvspan(line2_exts[0],line2_exts[1],alpha = 0.5, color = 'C3')
	plt.show(block = False)

	fts_w,fts_i = st.FTS_window(wavelims[0],wavelims[1])

	print("Okay. Select the first line again, this time from the FTS atlas.")

	line1_fts = select_image_region(fts_i)

	print("And the second line")
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(sample_int_spectrum)
	ax.axvspan(line1_exts[0],line1_exts[1],alpha = 0.5, color = 'C3')
	ax.axvspan(line2_exts[0],line2_exts[1],alpha = 0.5, color = 'C3')
	plt.show(block = False)

	line2_fts = select_image_region(fts_i)

	line1_firs_i = sample_int_spectrum[int(line1_exts[0]):int(line1_exts[1])]
	line2_firs_i = sample_int_spectrum[int(line2_exts[0]):int(line2_exts[1])]

	line1_fts_i = fts_i[int(line1_fts[0]):int(line1_fts[1])]
	line2_fts_i = fts_i[int(line2_fts[0]):int(line2_fts[1])]

	line1_firs_fit,_ = curve_fit(
			st.gaussian,
			np.arange(len(line1_firs_i)),
			line1_firs_i,
			p0 = [
				line1_firs_i.min(),
				len(line1_firs_i)/2,
				len(line1_firs_i)/4,
				line1_firs_i[0]]
	)

	line2_firs_fit,_ = curve_fit(
			st.gaussian,
			np.arange(len(line2_firs_i)),
			line2_firs_i,
			p0 = [
				line2_firs_i.min(),
				len(line2_firs_i)/2,
				len(line2_firs_i)/4,
				line2_firs_i[0]]
	)

	line1_fts_fit,_ = curve_fit(
			st.gaussian,
			np.arange(len(line1_fts_i)),
			line1_fts_i,
			p0 = [
				line1_fts_i.min(),
				len(line1_fts_i)/2,
				len(line1_fts_i)/4,
				line1_fts_i[0]]
	)

	line2_fts_fit,_ = curve_fit(
			st.gaussian,
			np.arange(len(line2_fts_i)),
			line2_fts_i,
			p0 = [
				line2_fts_i.min(),
				len(line2_fts_i)/2,
				len(line2_fts_i)/4,
				line2_fts_i[0]]
	)

	line1_FTS_wvl = fts_w[int(line1_fts[0]) + int(line1_fts_fit[1])]
	line2_FTS_wvl = fts_w[int(line2_fts[0]) + int(line2_fts_fit[1])]

	line1_firs_center = int(line1_exts[0]) + int(line1_firs_fit[1])
	line2_firs_center = int(line2_exts[0]) + int(line2_firs_fit[1])

	angstrom_per_pixel = np.abs(line2_FTS_wvl - line1_FTS_wvl)/np.abs(line2_firs_center - line1_firs_center)

	zero_wvl = line1_FTS_wvl - (angstrom_per_pixel * line1_firs_center)

	wavelength_array = (np.arange(0,len(sample_int_spectrum))*angstrom_per_pixel) + zero_wvl

	return wavelength_array

def firs_prefilter_correction(firs_data,wavelength_array,degrade_to = 50,rolling_window = 8,return_pfcs = False):
	"""Applies a prefilter correction to FIRS intensity data, then correct QUV channels for I. Pre-filter is determined pixel-to-pixel by dividing the observed spectrum by the FTS spectrum, degrading the dividend to n points, then taking the rolling median of the degraded spectrum.
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
		An array of the same shape as the input firs_data, containing the prefilter corrected I, as well as QUV divided by the pre-filter corrected I
	prefilter_corrections : array-like, optional
		An array the calculated prefilter corrections
	"""

	fts_wave,fts_spec = st.FTS_window(wavelength_array[0],wavelength_array[-1])
	fts_spec_in_FIRS_resolution = scinterp.interp1d(fts_wave,fts_spec)(wavelength_array)
	degrade_wave = np.linspace(wavelength_array[0],wavelength_array[-1],num = degrade_to)
	firs_data_corr = np.zeros(firs_data.shape)
	prefilter_corrections = np.zeros(
			tuple(
				np.array(firs_data.shape)[
					np.array(firs_data.shape) != 4]))
	if len(firs_data.shape) == 2:
		if firs_data.shape[0] == 4:
			divided = firs_data[0,:] / fts_spec_in_FIRS_resolution
		else:
			divided = firs_data[:,0] / fts_spec_in_FIRS_resolution
		firsFTS_interp = scinterp.interp1d(
				wavelength_array,
				divided)(degrade_wave)
		pfc = scinterp.interp1d(
				degrade_wave,
				rolling_median(firsFTS_interp,rolling_window))(wavelength_array)
		pfc = pfc / np.nanmax(pfc)
		if firs_data.shape[0] == 4:
			firs_data_corr[0,:] = firs_data[0,:] / pfc
			firs_data_corr[1,:] = firs_data[1,:] / firs_data_corr[0,:]
			firs_data_corr[2,:] = firs_data[2,:] / firs_data_corr[0,:]
			firs_data_corr[4,:] = firs_data[3,:] / firs_data_corr[0,:]
			prefilter_corrections[:] = pfc
		else:
			firs_data_corr[:,0] = firs_data[:,0] / pfc
			firs_data_corr[:,1] = firs_data[:,1] / firs_data_corr[:,0]
			firs_data_corr[:,2] = firs_data[:,2] / firs_data_corr[:,0]
			firs_data_corr[:,3] = firs_data[:,3] / firs_data_corr[:,0]
			prefilter_corrections[:] = pfc
	else:
		for i in range(firs_data.shape[0]):
			for j in range(firs_data.shape[2]):
				divided = firs_data[i,0,j,:] / fts_spec_in_FIRS_resolution
				firsFTS_interp = scinterp.interp1d(
						wavelength_array,
						divided)(degrade_wave)
				pfc = scinterp.interp1d(
						degrade_wave,
						rolling_median(firsFTS_interp,rolling_window))(wavelength_array)
				pfc = pfc / np.nanmax(pfc)
				firs_data_corr[i,0,j,:] = firs_data[i,0,j,:] / pfc
				firs_data_corr[i,1,j,:] = firs_data[i,1,j,:] / firs_data_corr[i,0,j,:]
				firs_data_corr[i,2,j,:] = firs_data[i,2,j,:] / firs_data_corr[i,0,j,:]
				firs_data_corr[i,3,j,:] = firs_data[i,3,j,:] / firs_data_corr[i,0,j,:]
				prefilter_corrections[i,j,:] = pfc
	if return_pfcs:
		return firs_data_corr,prefilter_corrections
	else:
		return firs_data_corr


