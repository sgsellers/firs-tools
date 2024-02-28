import spectraTools as spex

from astropy.coordinates import SkyCoord
import astropy.io.fits as fits
import astropy.units as u

import FTS_atlas

import h5py

from importlib import resources

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, SpanSelector

import numpy as np
import numpy.polynomial.polynomial as npoly

import os

import scipy.integrate as scinteg
import scipy.interpolate as scinterp
from scipy.io import readsav
from scipy.optimize import curve_fit
import scipy.ndimage as scind

from sunpy.coordinates import frames

import tqdm

import warnings

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
    z = (x - a1) / a2
    y = a0 * np.exp(- z ** 2 / 2.) + c
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
        If False and atlas is set to "FTS", will return the solar irradiance between wavemin and wavemax.
            (Do not recommend)
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
            "Your selected wavelengths are not in FTS atlas bounds. Come back when I bother downloading the IR/UV atlas"
        )
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
    half_window = int(window / 2)
    if half_window >= 4:
        for i in range(half_window):
            rolled[i] = np.nanmedian(data[i:i + 1])
            rolled[-(i + 1)] = np.nanmedian(data[(-(i + 4)):(-(i + 1))])
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
    idx = (np.abs(array - value)).argmin()
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
    firs_xsize = int(len(firs) / 4 / firs_nspex / firs_ysize)

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
        xcen1 = np.where(spectrum[xmin1:xmax1 + 1] == np.nanmin(spectrum[xmin1:xmax1 + 1]))[0][0]
        selector1.extents = (xmin1 + xcen1 - 6, xmin1 + xcen1 + 6)

        xmin2, xmax2 = 788, 800
        xcen2 = np.where(spectrum[xmin2:xmax2 + 1] == np.nanmin(spectrum[xmin2:xmax2 + 1]))[0][0]
        selector2.extents = (xmin2 + xcen2 - 4, xmin2 + xcen2 + 4)

        xmin3, xmax3 = 669, 690
        selector3.extents = (xmin3, xmax3)

        xmin4, xmax4 = 2392, 2410
        selector4.extents = (xmin4, xmax4)
    else:
        xmin1, xmax1 = xdata[0], xdata[int(len(xdata) / 8)]
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
    line1_firs_x = -line1_firs_coef[1] / (2 * line1_firs_coef[2])

    line2_firs_coef = npoly.polyfit(
        np.arange(len(line2_firs_i)),
        line2_firs_i,
        2
    )

    line2_firs_x = -line2_firs_coef[1] / (2 * line2_firs_coef[2])

    line1_fts_coef = npoly.polyfit(
        np.arange(len(line1_fts_i)),
        line1_fts_i,
        2
    )

    line1_fts_x = -line1_fts_coef[1] / (2 * line1_fts_coef[2])

    line2_fts_coef = npoly.polyfit(
        np.arange(len(line2_fts_i)),
        line2_fts_i,
        2
    )

    line2_fts_x = -line2_fts_coef[1] / (2 * line2_fts_coef[2])

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
            n_brightest = int(firs_data.shape[0] / 8)
            bright_args = slit_brightness_argsort[-n_brightest:]
            mean_slit = np.zeros((firs_data.shape[2], firs_data.shape[3]))
            for i in bright_args:
                mean_slit += firs_data[i, 0, :, :]
            mean_slit = mean_slit / len(bright_args)

        slit_pfc = np.zeros(mean_slit.shape)
        for i in tqdm.tqdm(range(mean_slit.shape[0]), desc="Determining Prefilter Shape..."):
            divided = mean_slit[i, :] / fts_spec_in_firs_resolution
            firsfts_interp = scinterp.interp1d(
                wavelength_array,
                divided)(degrade_wave)
            pfc = scinterp.interp1d(
                degrade_wave,
                _rolling_median(firsfts_interp, rolling_window))(wavelength_array)
            pfc = pfc / np.nanmax(pfc)
            slit_pfc[i, :] = pfc

        for i in tqdm.tqdm(range(firs_data.shape[0]), desc="Applying Prefilter Shape..."):
            for j in range(firs_data.shape[2]):
                firs_data_corr[i, 0, j, :] = firs_data[i, 0, j, :] / slit_pfc[j, :]
                qtmp = firs_data[i, 1, j, :] / slit_pfc[j, :]  # firs_data_corr[i, 0, j, :]
                firs_data_corr[i, 1, j, :] = linear_spectral_tilt_correction(wavelength_array, qtmp)
                utmp = firs_data[i, 2, j, :] / slit_pfc[j, :]  # firs_data_corr[i, 0, j, :]
                firs_data_corr[i, 2, j, :] = linear_spectral_tilt_correction(wavelength_array, utmp)
                vtmp = firs_data[i, 3, j, :] / slit_pfc[j, :]  # firs_data_corr[i, 0, j, :]
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
    fftfreqs = np.fft.fftfreq(len(wavelength_array), wavelength_array[1] - wavelength_array[0])

    ft_cut1 = fftfreqs >= lopass_cutoff
    ft_cut2 = fftfreqs <= -lopass_cutoff
    quv_fringe_image = np.zeros((3, flat_map.shape[1], flat_map.shape[2]))
    for i in tqdm.tqdm(range(flat_map.shape[1]), desc="Creating Fringe Template..."):
        for j in range(1, 4):
            quv_ft = np.fft.fft(_rolling_median(flat_map[j, i, :], 16))
            quv_ft[ft_cut1] = 0
            quv_ft[ft_cut2] = 0
            quv_fringe_image[j - 1, i, :] = np.real(np.fft.ifft(quv_ft))
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

    for i in tqdm.tqdm(range(fringe_corrected_map.shape[0]), desc="Applying Fringe Correction..."):
        for j in range(3):
            for k in range(fringe_corrected_map.shape[2]):
                # map_med = np.nanmedian(map_data[i, j + 1, k, :50])
                # fringe_med = np.nanmedian(fringe_template[j, k, :50])

                # corr_factor = fringe_med - map_med
                # Pretty sure these should be divided off, not subtracted. Gonna give it a shot.
                # Will revert if necessary
                fringe_corrected_map[i, j + 1, k, :] = (map_data[i, j + 1, k, :] /
                                                        (fringe_template[j, k, :] /
                                                         np.nanmedian(fringe_template[j, k, :])))

                # fringe_corr = fringe_template[j, k, :] - corr_factor

                # fringe_corrected_map[i, j + 1, k, :] = map_data[i, j + 1, k, :] - fringe_corr

    return fringe_corrected_map


def firs_vqu_crosstalk(dataCube, wavelengthArray, plot=True):
    """Estimates crosstalk in the V->Q, U direction. This method assumes that:
        1.) The crosstalk is wavelength-independent over the FIRS wavelength range
        2.) The spatial patterns of polarization signal in Q, U, V are different.
    The second assumption here breaks down for quiet-Sun maps, or maps without significant
    polarization signals. To this end, the routine will print the cross-talk value.

    Should the user wish to correct for V->Q,U crosstalk in quiet-sun data, I would suggest first
    correcting for a different dataset on a nearby day with a good sunspot, then applying the crosstalk
    values obtained from this correction. If these are no nearby datasets with differing Q,U,V structures,
    the wavelength range can be altered from isolating the photospheric Si I line to the chromospheric
    He I line. This should enable the user to attempt the correction from filament data.

    Note that this method has not been tested.

    Note that crosstalk from Q, U -> V is not considered. In general, this should be a small effect,
    as the Q, U signals are fairly low-amplitude.

    Parameters:
    -----------
    dataCube : array-like
        FIRS IQUV datacube for determination of crosstalk coefficients
    wavelengthArray : array-like
        FIRS wavelength array for selection of the photospheric Si I line

    Returns:
    --------
    crosstalkCoefficients : list
        list of [V->Q, V->U] crosstalk.
    """
    # Position of photospheric Si I line for constructing integrated Q, U, V maps.
    # This is hardcoded for now.
    siidx_lo = _find_nearest(wavelengthArray, 10824.75)
    siidx_hi = _find_nearest(wavelengthArray, 10829)

    crosstalkRange = np.linspace(-0.15, 0.15, 50)
    correlationQV = np.zeros(50)
    correlationUV = np.zeros(50)
    for i in tqdm.tqdm(range(len(crosstalkRange)), desc="Determining V->Q,U Crosstalk..."):
        qmod = dataCube[:, 1, :, siidx_lo:siidx_hi] + crosstalkRange[i] * dataCube[:, 3, :, siidx_lo:siidx_hi]
        umod = dataCube[:, 2, :, siidx_lo:siidx_hi] + crosstalkRange[i] * dataCube[:, 3, :, siidx_lo:siidx_hi]

        qint = scinteg.trapezoid(
            np.abs(qmod),
            wavelengthArray[siidx_lo:siidx_hi],
            axis=-1
        )
        uint = scinteg.trapezoid(
            np.abs(umod),
            wavelengthArray[siidx_lo:siidx_hi],
            axis=-1
        )
        vint = scinteg.trapezoid(
            np.abs(dataCube[:, 3, :, siidx_lo:siidx_hi]),
            wavelengthArray[siidx_lo:siidx_hi],
            axis=-1
        )

        s1q = np.abs(qint) - np.nanmean(np.abs(qint))
        s1u = np.abs(uint) - np.nanmean(np.abs(uint))
        s2v = np.abs(vint) - np.nanmean(np.abs(vint))

        correlationQV[i] = np.nansum(s1q * s2v) / np.sqrt(np.nansum(s1q**2) * np.nansum(s2v**2))
        correlationUV[i] = np.nansum(s1u * s2v) / np.sqrt(np.nansum(s1u ** 2) * np.nansum(s2v ** 2))

    interp_range = np.linspace(crosstalkRange[0], crosstalkRange[-1], 1000)
    qv_interp = scinterp.interp1d(
        crosstalkRange,
        correlationQV,
        kind='quadratic')(interp_range)
    vqCrosstalk = interp_range[list(qv_interp).index(np.nanmin(qv_interp))]

    uv_interp = scinterp.interp1d(
        crosstalkRange,
        correlationUV,
        kind='quadratic')(interp_range)
    vuCrosstalk = interp_range[list(uv_interp).index(np.nanmin(uv_interp))]

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.plot(crosstalkRange, correlationQV, color='C0', label="V->Q Correlation")
        ax.plot(
            interp_range,
            qv_interp,
            color='C1',
            linestyle='--',
            label='Interpolated'
        )
        ax.axvline(vqCrosstalk, linestyle='--', color='k', label='Minimum: '+str(round(vqCrosstalk, 3)))
        ax.legend(loc='lower right')
        ax.set_title("V->Q Crosstalk")
        ax.set_xlabel("V2Q Values")
        ax.set_ylabel("Correlation Values")
        ax = fig.add_subplot(212)
        ax.plot(crosstalkRange, correlationUV, color='C0', label="V->U Correlation")
        ax.plot(
            interp_range,
            uv_interp,
            color='C1',
            linestyle='--',
            label='Interpolated'
        )
        ax.axvline(vuCrosstalk, linestyle='--', color='k', label='Minimum: ' + str(round(vuCrosstalk, 3)))
        ax.legend(loc='lower left')
        ax.set_title("V->U Crosstalk")
        ax.set_xlabel("V2U Values")
        ax.set_ylabel("Correlation Values")
        plt.tight_layout()
        plt.show()

    if (vqCrosstalk == interp_range[0]) or (vqCrosstalk == interp_range[-1]):
        warnings.warn("V->Q Crosstalk could not be determined via linear correlation. Defaulting to 0.")
        vqCrosstalk = 0
    if (vuCrosstalk == interp_range[0]) or (vuCrosstalk == interp_range[-1]):
        warnings.warn("V->U Crosstalk could not be determined via linear correlation. Defaulting to 0.")
        vuCrosstalk = 0

    crosstalkCoefficients = [vqCrosstalk, vuCrosstalk]
    return crosstalkCoefficients


def firs_coordinate_conversion(raw_file, correctTime=False):
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
    correctTime : bool
        Older FIRS data recorded time in local time. Newer is UTC. If true, updates datetime to UTC.
        If False, assumes that the correct UTC time is used in FIRS data.
    """

    raw_hdr = fits.open(raw_file)[0].header
    stony_lat = raw_hdr['DST_SLAT']
    stony_lon = raw_hdr['DST_SLNG']
    rotation_angle = raw_hdr['DST_GDRN'] - 13.3  # 13.3 is the offset of the DST guider head to solar north
    # There may still be 90 degree rotations, or other translations
    obstime = raw_hdr['OBS_STAR']
    if correctTime:
        obstime = np.datetime64(obstime) + _correct_datetimes(np.datetime64(obstime))
        obstime = str(obstime)
    date = raw_hdr['DATE_OBS'].replace('/', '-')
    stony_coord = SkyCoord(
        stony_lon * u.deg,
        stony_lat * u.deg,
        frame=frames.HeliographicStonyhurst,
        observer='earth',
        obstime=obstime
    )

    helio_coord = stony_coord.transform_to(frames.Helioprojective)
    return helio_coord, rotation_angle, date


def firs_deskew(flat_map_fname, lineIndices=[188, 254]):
    """
    Determines spectral skew along slit from flat map, and returns array of shift values
    :param flat_map_fname: str
        path to flat map
    :param lineIndices: list
        Low and high index for Si I spectral line, or other strong spectral line
    :return skews: numpy.ndarray
        Array of shifts for deskewing of FIRS data
    """
    # Only the Si Line
    flat_map = read_firs(flat_map_fname)[:, 0, :, lineIndices[0]:lineIndices[1]]
    print(flat_map.shape)
    core_position = _find_nearest(
        flat_map[int(flat_map.shape[0] / 2), :],
        flat_map[int(flat_map.shape[0] / 2), :].min()
    )
    print(core_position)
    skews = spex.spectral_skew(flat_map[:, core_position - 7: core_position + 5])
    return skews


# noinspection PyTypeChecker
def firs_construct_hdu(firs_data, firs_lambda, meta_file, coordinates,
                       rotation, date, dx, dy, exptime, coadd,
                       correctTime=False):
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
    correctTime : bool
        Older FIRS data recorded time in local time. Newer is UTC. If true, updates datetime to UTC.
        If False, assumes that the correct UTC time is used in FIRS data.

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

    if correctTime:
        utc_offset = _correct_datetimes(t0)
    else:
        utc_offset = np.timedelta64(0, "s")

    t0 += utc_offset
    t1 += utc_offset

    ext0 = fits.PrimaryHDU()
    ext0.header['DATE'] = (np.datetime64('now').astype(str), 'File created')
    ext0.header['ORIGIN'] = 'NMSU/SSOC'
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
    ext0.header['XPOSUR'] = (exptime * coadd, 'ms')
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
    ext1.header['CTYPE1'] = 'HPLN-TAN'
    ext1.header['CTYPE2'] = 'HPLT-TAN'
    ext1.header['CTYPE3'] = 'WAVE'
    ext1.header['CUNIT1'] = 'arcsec'
    ext1.header['CUNIT2'] = 'arcsec'
    ext1.header['CUNIT3'] = 'Angstrom'
    ext1.header['CRVAL1'] = coordinates.Tx.value
    ext1.header['CRVAL2'] = coordinates.Ty.value
    ext1.header['CRVAL3'] = firs_lambda[0]
    ext1.header['CRPIX1'] = firs_data.shape[0] / 2
    ext1.header['CRPIX2'] = firs_data.shape[2] / 2
    ext1.header['CRPIX3'] = 1
    ext1.header['CROTAN'] = rotation

    ext2 = fits.ImageHDU(np.flipud(np.rot90(firs_data[:, 1, :, :])))
    ext2.header['EXTNAME'] = ('Stokes-Q', "Corrected for I,V Crosstalk. Not normalized.")
    ext2.header['CDELT1'] = (dx, 'arcsec')
    ext2.header['CDELT2'] = (dy, 'arcsec')
    ext2.header['CDELT3'] = (firs_lambda[1] - firs_lambda[0], 'Angstom')
    ext2.header['CTYPE1'] = 'HPLN-TAN'
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
    ext3.header['EXTNAME'] = ('Stokes-U', "Corrected for I,V Crosstalk. Not normalized.")
    ext3.header['CDELT1'] = (dx, 'arcsec')
    ext3.header['CDELT2'] = (dy, 'arcsec')
    ext3.header['CDELT3'] = (firs_lambda[1] - firs_lambda[0], 'Angstom')
    ext3.header['CTYPE1'] = 'HPLN-TAN'
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
    ext4.header['EXTNAME'] = ('Stokes-V', "Corrected for I crosstalk. Not normalized")
    ext4.header['CDELT1'] = (dx, 'arcsec')
    ext4.header['CDELT2'] = (dy, 'arcsec')
    ext4.header['CDELT3'] = (firs_lambda[1] - firs_lambda[0], 'Angstom')
    ext4.header['CTYPE1'] = 'HPLN-TAN'
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

    ext6 = fits.ImageHDU(60 * 60 * meta_info['ttime'] + utc_offset.astype(float))
    ext6.header['EXTNAME'] = 'time-coordinate'
    ext6.header['BTYPE'] = 'time axis'
    ext6.header['BUNIT'] = '[s]'

    hdulist = fits.HDUList([ext0, ext1, ext2, ext3, ext4, ext5, ext6])

    return hdulist


def firs_contstruct_param_hdu(
        firs_fits,
        mean_cpl, net_cpl, mean_lpl,
        vmaps, wmaps,
        line_names, refwvls, indices
):
    """
    Constructs FITS HDUList of derived Level-1.5 parameters.

    :param firs_fits: str
        FIRS Level-1.5 file
    :param mean_cpl: list
        List of numpy.ndarray of Mean CPL maps
    :param net_cpl: list
        List of numpy.ndarray of Net CPL maps
    :param mean_lpl: list
        List of numpy.ndarray of Mean LPL maps
    :param vmaps: list
        List of numpy.ndarray of LOS V-maps
    :param wmaps: list
        List of numpy.ndarray of V-width maps
    :param line_names: list
        List of line names
    :param refwvls: list
        List of line reference wavelengths
    :param indices: list
        List of indices used for analysis of the form [[idxlo1, idxhi1], [idxlo2, idxhi2], ...]

    :return hdulist : astropy.io.fits.HDUList object
        Nicely formatted HDUList for writing to disk
    """

    firs_data = fits.open(firs_fits)

    cdelt1 = firs_data[1].header['CDELT1']
    cdelt2 = firs_data[1].header['CDELT2']
    ctype1 = firs_data[1].header['CTYPE1']
    ctype2 = firs_data[1].header['CTYPE2']
    cunit1 = firs_data[1].header['CUNIT1']
    cunit2 = firs_data[1].header['CUNIT2']
    crval1 = firs_data[1].header['CRVAL1']
    crval2 = firs_data[1].header['CRVAL2']
    crpix1 = firs_data[1].header['CRPIX1']
    crpix2 = firs_data[1].header['CRPIX2']
    crotan = firs_data[1].header['CROTAN']

    crkeys = ['CDELT1', 'CDELT2', 'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2',
              'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CROTAN']

    crvals = [cdelt1, cdelt2, ctype1, ctype2, cunit1, cunit2, crval1, crval2, crpix1, crpix2, crotan]

    firs_waves = firs_data['lambda-coordinate'].data

    hdulist = []

    ext0 = fits.PrimaryHDU()
    ext0.header = firs_data[0].header
    del ext0.header['BTYPE']
    del ext0.header['BUNIT']
    ext0.header['PRSTEP5'] = ('POL-ANALYSIS', "firs-tools (S.Sellers")
    ext0.header['PRSTEP6'] = ('MOMENT-ANALYSIS', 'firs-tools (S.Sellers)')
    hdulist.append(ext0)

    for i in range(len(mean_cpl)):
        ext = fits.ImageHDU(np.flipud(np.rot90(mean_cpl[i])))
        ext.header['EXTNAME'] = 'MEAN-CPL-'+str(i)
        ext.header['BTYPE'] = 'MEAN CIRCULAR POLARIZATION'
        ext.header['STARTOBS'] = ext0.header['STARTOBS']
        ext.header['ENDOBS'] = ext0.header['ENDOBS']
        for j in range(len(crkeys)):
            ext.header[crkeys[j]] = crvals[j]
        ext.header['WAVEBAND'] = (line_names[i], "Strongest Line in wavelength range")
        ext.header['REFWVL'] = refwvls[i]
        ext.header['WAVE1'] = (firs_waves[indices[i][0]], "Lower-bound wavelength for analysis")
        ext.header['WAVE2'] = (firs_waves[indices[i][0]], "Upper-bound wavelength for analysis")
        ext.header['METHOD'] = "Modified Sums"
        hdulist.append(ext)
    for i in range(len(net_cpl)):
        ext = fits.ImageHDU(np.flipud(np.rot90(net_cpl[i])))
        ext.header['EXTNAME'] = 'NET-CPL-' + str(i)
        ext.header['BTYPE'] = 'NET CIRCULAR POLARIZATION'
        ext.header['STARTOBS'] = ext0.header['STARTOBS']
        ext.header['ENDOBS'] = ext0.header['ENDOBS']
        for j in range(len(crkeys)):
            ext.header[crkeys[j]] = crvals[j]
        ext.header['WAVEBAND'] = (line_names[i], "Strongest Line in wavelength range")
        ext.header['REFWVL'] = refwvls[i]
        ext.header['WAVE1'] = (firs_waves[indices[i][0]], "Lower-bound wavelength for analysis")
        ext.header['WAVE2'] = (firs_waves[indices[i][0]], "Upper-bound wavelength for analysis")
        ext.header['METHOD'] = "Integration"
        hdulist.append(ext)
    for i in range(len(mean_lpl)):
        ext = fits.ImageHDU(np.flipud(np.rot90(mean_lpl[i])))
        ext.header['EXTNAME'] = 'MEAN-LPL-' + str(i)
        ext.header['BTYPE'] = 'MEAN LINEAR POLARIZATION'
        ext.header['STARTOBS'] = ext0.header['STARTOBS']
        ext.header['ENDOBS'] = ext0.header['ENDOBS']
        for j in range(len(crkeys)):
            ext.header[crkeys[j]] = crvals[j]
        ext.header['WAVEBAND'] = (line_names[i], "Strongest Line in wavelength range")
        ext.header['REFWVL'] = refwvls[i]
        ext.header['WAVE1'] = (firs_waves[indices[i][0]], "Lower-bound wavelength for analysis")
        ext.header['WAVE2'] = (firs_waves[indices[i][0]], "Upper-bound wavelength for analysis")
        ext.header['METHOD'] = "Modified Sums"
        hdulist.append(ext)
    for i in range(len(vmaps)):
        ext = fits.ImageHDU(np.flipud(np.rot90(vmaps[i])))
        ext.header['EXTNAME'] = 'VLOS-' + str(i)
        ext.header['BTYPE'] = 'LOS-V'
        ext.header['BUNIT'] = "km/s"
        ext.header['STARTOBS'] = ext0.header['STARTOBS']
        ext.header['ENDOBS'] = ext0.header['ENDOBS']
        for j in range(len(crkeys)):
            ext.header[crkeys[j]] = crvals[j]
        ext.header['WAVEBAND'] = (line_names[i], "Strongest Line in wavelength range")
        ext.header['REFWVL'] = refwvls[i]
        ext.header['WAVE1'] = (firs_waves[indices[i][0]], "Lower-bound wavelength for analysis")
        ext.header['WAVE2'] = (firs_waves[indices[i][0]], "Upper-bound wavelength for analysis")
        ext.header['METHOD'] = "Moment Analysis"
        hdulist.append(ext)
    for i in range(len(wmaps)):
        ext = fits.ImageHDU(np.flipud(np.rot90(wmaps[i])))
        ext.header['EXTNAME'] = 'VWIDTH-' + str(i)
        ext.header['BTYPE'] = 'VELOCITY-WIDTH'
        ext.header['BUNIT'] = "km/s"
        ext.header['STARTOBS'] = ext0.header['STARTOBS']
        ext.header['ENDOBS'] = ext0.header['ENDOBS']
        for j in range(len(crkeys)):
            ext.header[crkeys[j]] = crvals[j]
        ext.header['WAVEBAND'] = (line_names[i], "Strongest Line in wavelength range")
        ext.header['REFWVL'] = refwvls[i]
        ext.header['WAVE1'] = (firs_waves[indices[i][0]], "Lower-bound wavelength for analysis")
        ext.header['WAVE2'] = (firs_waves[indices[i][0]], "Upper-bound wavelength for analysis")
        ext.header['METHOD'] = "Moment Analysis"
        hdulist.append(ext)

    return fits.HDUList(hdulist)


def firs_to_fits(firs_map_fname, flat_map_fname, raw_file, outname, dx=0.3, dy=0.15,
                 exptime=125, coadd=10, plot=False, vquCrosstalk=True, correctTime=False, momentAnalysis=True):
    """This function converts FIRS .dat files to level 1.5 fits files with a wavelength array, time array, and corrected
    for fringeing. You will require a map containing a flat field that has been processed as a science map by the FIRS
    IDL pipeline.

    2024-02-26: Adding skew correction along slit and higher-order data products
    Moment analysis will include:
        ~I,
        ~LOS-V and
        ~V-width for Si I, and the two dominant components of He I
        Will also include QUV_tot for these lines

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
    vquCrosstalk : bool or list
        If true, attempts to correct for V -> Q, U crosstalk via linear correlation coefficient.
        If false, no correction is required.
        If vquCrosstalk is a list of floats, these are taken to be the [V->Q, V->U] crosstalk values.
    correctTime : bool
        Older FIRS data recorded time in local time. Newer is UTC. If true, updates datetime to UTC.
        If False, assumes that the correct UTC time is used in FIRS data.
    momentAnalysis : bool or list
        If Bool, performs moment analysis and stokes integration with default line center values
        If list, expects line indices of the form [[idxlo1, idxhi1], [idxl02, idxhi2], ...]

    Returns:
    --------
    None, but it writes a fits file.
    """

    # L1 data
    firs_data = read_firs(firs_map_fname)

    print("Performing Wavelength Calibration.")
    # Wave Cal
    firs_waves = firs_wavelength_cal_poly(
        np.nanmean(firs_data[:, 0, 100:400, :], axis=(0, 1)),
        plot=plot
    )

    print("Performing Prefilter Curvature Calibration.")
    # Prefilter Cal
    firs_data = firs_prefilter_correction(firs_data, firs_waves)

    print("Correcting for fringes via flat map template")
    # Fringe Cal
    firs_data = firs_fringecorr(firs_data, firs_waves, flat_map_fname, plot=plot)

    # V --> Q, U Crosstalk correction
    if type(vquCrosstalk) is bool:
        if vquCrosstalk:
            coeffCrosstalk = firs_vqu_crosstalk(firs_data, firs_waves, plot=plot)
        else:
            coeffCrosstalk = [0, 0]
    else:
        coeffCrosstalk = vquCrosstalk

    print("V --> Q, U Crosstalk Determined")
    print("V --> Q = ", str(coeffCrosstalk[0]))
    print("V --> U = ", str(coeffCrosstalk[1]))

    firs_data[:, 1, :, :] = firs_data[:, 1, :, :] + coeffCrosstalk[0] * firs_data[:, 3, :, :]
    firs_data[:, 2, :, :] = firs_data[:, 2, :, :] + coeffCrosstalk[0] * firs_data[:, 3, :, :]

    spex_skews = firs_deskew(flat_map_fname)
    print("Deskewing FIRS Data")
    for i in tqdm.tqdm(range(firs_data.shape[0])):
        for j in range(firs_data.shape[1]):
            for k in range(firs_data.shape[2]):
                firs_data[i, j, k, :] = scind.shift(firs_data[i, j, k, :], spex_skews[k], mode='nearest')

    # Redo Wave Cal with deskewed data
    firs_waves = firs_wavelength_cal_poly(
        np.nanmean(firs_data[:, 0, 100:400, :], axis=(0, 1)),
        plot=plot
    )

    if type(momentAnalysis) is list:
        refwvls, indices = firs_refwvls(firs_data, firs_waves, spectralIndices=momentAnalysis)
        line_names = ['Line-'+str(i) for i in range(len(refwvls))]
        mean_cpl, mean_lpl, net_cpl, vmaps, wmaps = firs_analysis(firs_data, firs_waves, indices, refwvls)
    elif momentAnalysis:
        refwvls, indices = firs_refwvls(firs_data, firs_waves)
        line_names = ['Si I 10827 A', 'He I 10829 A', 'He I 10830 A']
        mean_cpl, mean_lpl, net_cpl, vmaps, wmaps = firs_analysis(firs_data, firs_waves, indices, refwvls)
    else:
        mean_cpl = []
        mean_lpl = []
        net_cpl = []
        vmaps = []
        wmaps = []
        line_names = []
        refwvls = []
        indices = []

    coordinates, crotan, date = firs_coordinate_conversion(raw_file, correctTime=correctTime)

    print("Writing FIRS Level-1.5 fits file.")
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
        coadd,
        correctTime=correctTime
    )

    hdulist.writeto(outname, overwrite=True)

    # We'll write the parameter maps in a separate file to avoid unneccessarily large file sizes.
    if len(refwvls) > 0:
        params_outname = outname.split(".fits")[0] + "_derived_parameter_maps.fits"
        param_hdulist = firs_contstruct_param_hdu(
            outname,
            mean_cpl,
            net_cpl,
            mean_lpl,
            vmaps,
            wmaps,
            line_names,
            refwvls,
            indices
        )
        param_hdulist.writeto(params_outname, overwrite=True)
    return


def firs_analysis(firs_data, firs_wavelengths, analysis_indices, reference_wavelengths):
    """
    Performs moment analysis and determines mean circular/linear polarization plus net circular polarization maps
    for each of the spectral windows given. See Martinez Pillet et. al., 2011 discussion of mean polarizations.
    For net circular polarization, see Solanki & Montavon 1993.
    :param firs_data: numpy.ndarray
        FIRS 4d data of shape (ny, 4, ny, nlambda)
    :param firs_wavelengths: numpy.ndarray
        Array of wavelength values of shape nlambda
    :param analysis_indices: list
        List of form [[loidx1, hiidx1], [loidx2, hiidx2], ...]
    :param reference_wavelengths: list
        List of reference wavelengths, same length as analysis_indices
    :return mean_cpl: list
        List of numpy.ndarray mean CPL maps of shape (nx, ny)
    :return mean_lpl: list
        List of numpy.ndarray mean LPL maps of shape (nx, ny)
    :return net_cpl: list
        List of numpy.ndarray net CPL maps of shape (nx, ny)
    :return vmaps: list
        List of numpy.ndarray moment analysis LOS velocity maps of shape (nx, ny)
    :return wmaps: list
        List of numpy.ndarray moment analysis velocity width maps of shape (nx, ny)
    """
    mean_cont_brightness = np.nanmean(firs_data[:, 0, :, 100:150])
    qs_cont_values = []
    for j in range(firs_data.shape[0]):
        for k in range(firs_data.shape[2]):
            if np.nanmean(firs_data[j, 0, k, 100:150]) >= 0.8 * mean_cont_brightness:
                qs_cont_values.append(np.nanmean(firs_data[j, 0, k, 100:150]))
    continuum_intensity = np.nanmean(np.array(qs_cont_values))
    mean_cpl = []
    mean_lpl = []
    net_cpl = []
    vmaps = []
    wmaps = []
    with tqdm.tqdm(total=len(reference_wavelengths) * firs_data.shape[0] * firs_data.shape[2]) as pbar:
        for i in range(len(analysis_indices)):
            cpl_map = np.zeros((firs_data.shape[0], firs_data.shape[2]))
            lpl_map = np.zeros((firs_data.shape[0], firs_data.shape[2]))
            ncpl_map = np.zeros((firs_data.shape[0], firs_data.shape[2]))
            vlos_map = np.zeros((firs_data.shape[0], firs_data.shape[2]))
            vwid_map = np.zeros((firs_data.shape[0], firs_data.shape[2]))
            for j in range(firs_data.shape[0]):
                for k in range(firs_data.shape[2]):
                    _, v, w = spex.moment_analysis(
                        firs_wavelengths[analysis_indices[i][0]:analysis_indices[i][1]],
                        firs_data[j, 0, k, analysis_indices[i][0]:analysis_indices[i][1]],
                        reference_wavelengths[i]
                    )
                    vlos_map[j, k] = v
                    vwid_map[j, k] = w
                    cpl_map[j, k] = spex.mean_cpl(
                        firs_data[j, 3, k, analysis_indices[i][0]:analysis_indices[i][1]],
                        firs_wavelengths[analysis_indices[i][0]:analysis_indices[i][1]],
                        reference_wavelengths[i],
                        continuum_intensity
                    )
                    lpl_map[j, k] = spex.mean_lpl(
                        firs_data[j, 1, k, analysis_indices[i][0]:analysis_indices[i][1]],
                        firs_data[j, 2, k, analysis_indices[i][0]:analysis_indices[i][1]],
                        continuum_intensity
                    )
                    ncpl_map[j, k] = spex.net_cpl(
                        firs_data[j, 3, k, analysis_indices[i][0]:analysis_indices[i][1]],
                        firs_wavelengths[analysis_indices[i][0]:analysis_indices[i][1]]
                    )
                    pbar.update(1)
            mean_cpl.append(cpl_map)
            mean_lpl.append(lpl_map)
            net_cpl.append(ncpl_map)
            vmaps.append(vlos_map)
            wmaps.append(vwid_map)
    return mean_cpl, mean_lpl, net_cpl, vmaps, wmaps


def firs_refwvls(firs_data, firs_wavelengths, spectralIndices=None):
    """
    Determines reference wavelengths of a given spectral line. When spectralIndices are not given, defaults to Si I
    and two components of He I
    :param firs_data: numpy.ndarray
        Corrected datacube
    :param firs_wavelengths: numpy.ndarray
        Array of corresponding wavelengths.
    :param spectralIndices: None or List
        If list, expects line indices of the form [[idxlo1, idxhi1], [idxl02, idxhi2], ...]
        If none, uses default positions for Si I and He I
    :return refwvls: list
        List of reference wavelengths
    :return adjustedIndices: list
        Adjusted list of spectral indices to give an evenly spaced window.
    """
    mean_cont_brightness = np.nanmean(firs_data[:, 0, :, 100:150])
    if type(spectralIndices) is list:
        # Reference wavelength determinations. From profiles with continuum brighter than 80% of mean
        refwvls = []
        adjustedIndices = []
        for i in range(len(spectralIndices)):
            mean_wvls = []
            for j in range(firs_data.shape[0]):
                for k in range(firs_data.shape[2]):
                    if np.nanmean(firs_data[j, 0, k, 100:150]) >= 0.8*mean_cont_brightness:
                        mindx = _find_nearest(
                            firs_data[j, 0, k, spectralIndices[i][0]:spectralIndices[i][1]],
                            np.nanmin(firs_data[j, 0, k, spectralIndices[i][0]:spectralIndices[i][1]])
                        ) + spectralIndices[i][0]
                        mean_wvls.append(
                            float(
                                spex.find_line_core(
                                    firs_data[j, 0, k, mindx-7:mindx+7],
                                    wvl=firs_wavelengths[mindx-7:mindx+7]
                                )
                            )
                        )
            rwvl = np.nanmean(np.array(mean_wvls))
            mean_dist = np.nanmean(
                np.abs(
                    np.array([firs_wavelengths[spectralIndices[i][0]] - rwvl,
                              firs_wavelengths[spectralIndices[i][1]] - rwvl])
                )
            )
            adjustedIndices.append(
                [_find_nearest(firs_wavelengths, rwvl - mean_dist),
                 _find_nearest(firs_wavelengths, rwvl + mean_dist) + 1]
            )
            refwvls.append(rwvl)
    else:
        # Default line set. Si I 10827, He I 10829, He I 10830
        # From NIST
        # The He I 10830 is the weighted avg of the two components
        labRefwvl = np.array([10827.089, 10829.09115, 10830.30989])
        wvlOffsets = labRefwvl - labRefwvl[0]
        indices = [[190, 250], [257, 287], [287, 325]]
        mean_wvls = []
        for j in range(firs_data.shape[0]):
            for k in range(firs_data.shape[2]):
                if np.nanmean(firs_data[j, 0, k, 100:150]) >= 0.8 * mean_cont_brightness:
                    mindx = _find_nearest(
                        firs_data[j, 0, k, indices[0][0]:indices[0][1]],
                        np.nanmin(firs_data[j, 0, k, indices[0][0]:indices[0][1]])
                    ) + indices[0][0]
                    mean_wvls.append(
                        float(
                            spex.find_line_core(
                                firs_data[j, 0, k, mindx - 7:mindx + 7],
                                wvl=firs_wavelengths[mindx - 7:mindx + 7]
                            )
                        )
                    )
        sirwvl = np.nanmean(np.array(mean_wvls))
        refwvls = [sirwvl + i for i in wvlOffsets]
        adjustedIndices = []
        for i in range(len(refwvls)):
            mean_dist = np.nanmean(
                np.abs(
                    np.array([firs_wavelengths[indices[i][0]] - refwvls[i],
                              firs_wavelengths[indices[i][1]] - refwvls[i]])
                )
            )
            adjustedIndices.append(
                [_find_nearest(firs_wavelengths, refwvls[i] - mean_dist),
                 _find_nearest(firs_wavelengths, refwvls[i] + mean_dist) + 1]
            )
    return refwvls, adjustedIndices


def repackHazel(
        h5File, initFile, fitsFile, saveName,
        nx=None, ny=None,
        ch_key='ch1', ph_key='ph1', sp_key='he',
        translation=False,
        binSlits=1,
        binSpatial=2,
        overviewPlot=True
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
    binSlits : int
        Binning factor used in the rastering direction (i.e., number of slits summed)
    binSpatial : int
        Binning factor used along the slit
    overviewPlot : bool
        If true (default), creates an overview plot of Hazel results for the chromosphere
        and photosphere at tau=0
    """
    fitsFile = fits.open(fitsFile)
    dy = fitsFile[1].header['CDELT1'] * binSlits
    dx = fitsFile[1].header['CDELT2'] * binSpatial

    initFile = h5py.File(initFile, "r")

    h5File = h5py.File(h5File, 'r')
    if not nx:
        nx = int((fitsFile[1].header['NAXIS3'] - 1) / binSpatial)
    if not ny:
        ny = int((fitsFile[1].header['NAXIS2'] - 1) / binSlits)
    if type(ch_key) is str:
        ch_key = [ch_key]

    wavelength = h5File[sp_key]['wavelength'][:]
    chParams = [
        'Bx', 'Bx_err',
        'By', 'By_err',
        'Bz', 'Bz_err',
        'v', 'v_err',
        'deltav', 'deltav_err',
        'tau', 'tau_err',
        'a', 'a_err',
        'beta', 'beta_err',
        'ff', 'ff_err'
    ]
    chParamUnits = [
        'Gauss', 'Gauss',
        'Gauss', 'Gauss',
        'Gauss', 'Gauss',
        'km/s', 'km/s',
        'km/s', 'km/s',
        'log10(OpticalDepth)', 'log10(OpticalDepth)',
        'Damping', 'Damping',
        'plasmaB', 'plasmaB',
        'FillFactor', 'FillFactor'
    ]

    phParams = [
        'Bx', 'Bx_err',
        'By', 'By_err',
        'Bz', 'Bz_err',
        'T', 'T_err',
        'v', 'v_err',
        'vmac', 'vmac_err',
        'vmic', 'vmic_err',
        'ff', 'ff_err'
    ]
    phParamUnits = [
        'Gauss', 'Gauss',
        'Gauss', 'Gauss',
        'Gauss', 'Gauss',
        'Kelvin', 'Kelvin',
        'km/s', 'km/s',
        'km/s', 'km/s',
        'km/s', 'km/s',
        'FillFactor', 'FillFactor'
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
    # NOTE: This is currently broken due to binning slit positions
    # Just needs some extra logic, but I'm undercaffienated and overworked to figure it out
    # right this second.
    # if translation:
    #     if nx != fitsFile[1].header['NAXIS3'] - 1:
    #         del ext0.header['STARTOBS']
    #         del ext0.header['DATE_END']
    #         del ext0.header['ENDOBS']
    # else:
    #     if ny != fitsFile[1].header['NAXIS2'] - 1:
    #         del ext0.header['STARTOBS']
    #         del ext0.header['DATE_END']
    #         del ext0.header['ENDOBS']
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
        columns = []
        for i in range(len(chParams)):
            if 'err' in chParams[i]:
                paramArray = np.zeros((nx, ny))
                eArray = chromosphere[chParams[i]][:, 0, -1].reshape(nx, ny)
                for x in range(eArray.shape[0]):
                    for y in range(eArray.shape[1]):
                        if len(eArray[x, y]) != 0:
                            paramArray[x, y] = eArray[x, y]

            else:
                paramArray = chromosphere[chParams[i]][:, 0, -1, 0].reshape(nx, ny)
            if translation:
                paramArray = np.flipud(np.rot90(paramArray))
            columns.append(
                fits.Column(
                    name=chParams[i],
                    format=str(int(nx * ny)) + 'D',
                    dim='(' + str(paramArray.shape[1]) + "," + str(paramArray.shape[0]) + ")",
                    unit=chParamUnits[i],
                    array=paramArray.reshape(1, paramArray.shape[0], paramArray.shape[1])
                )
            )
        ext = fits.BinTableHDU.from_columns(columns)
        ext.header['EXTNAME'] = ("CHROMOSPHERE", 'Fit chromospheric parameters from Hazel Inversions')
        ext.header['LINE'] = ('He-I', 'He I 10830 [AA] Triple')
        fitsHDUs.append(ext)

    hdulist = fits.HDUList(fitsHDUs)
    hdulist.writeto(saveName)

    # Now we pack our photospheric results.
    # Unlike the chromospheres, there's an additional axis, the height profile.
    # We'll use this profile as the length of each column in the fits table.
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
    for i in range(len(phParams)):
        """ Slight explanation of the following code:
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
        Okay. Fuck.
        Nevermind. Apparently, fit parameters can have nans in their nodelist AND empties.
        I have ~no~ idea why some failed fits are rendered as nans, and others as empties, but absolutely fuck me.
        To tell for sure that the parameter isn't fit, ALL nodes must be nans or empties. 
        This ripples to the chromosphere.
        AND, just to fuck me in the ass some more,  vmac in the photosphere doesn't have a height profile.
        It's the only one I've found like that. I have no idea.
        """
        nodeKey = phParams[i].split("_")[0] + "_nodes"
        nodeArr = photosphere[nodeKey][:, 0, -1].reshape(nx, ny)
        nodeCount = len(nodeArr[0, 0])
        while nodeCount == 0:
            nodeCount = len(
                nodeArr[
                    np.random.randint(0, nx),
                    np.random.randint(0, ny)
                ]
            )
        nodeArrFull = np.zeros((nx, ny, nodeCount))
        for x in range(nodeArr.shape[0]):
            for y in range(nodeArr.shape[1]):
                if len(nodeArr[x, y]) != 0:
                    nodeArrFull[x, y, :] = nodeArr[x, y]
        nodeArrFull = np.nan_to_num(nodeArrFull)
        # Case: Parameter isn't fit for. All nans in node array (i.e., 0s)
        if len(nodeArrFull[nodeArrFull != 0]) == 0:
            if 'err' in phParams[i]:
                fill = np.zeros((len(logTau), nx, ny))
                if translation:
                    fill = np.flip(np.rot90(fill, axes=(1, 2)), axis=1)
                columns.append(
                    fits.Column(
                        name=phParams[i],
                        format=str(int(nx * ny)) + "I",
                        dim='(' + str(fill.shape[2]) + "," + str(fill.shape[1]) + ")",
                        unit=phParamUnits[i],
                        array=fill
                    )
                )
            else:
                fill = np.zeros((len(logTau), nx, ny)) + photosphere[phParams[i]][0, 0, -1, 0]
                if translation:
                    fill = np.flip(np.rot90(fill, axes=(1, 2)), axis=1)
                columns.append(
                    fits.Column(
                        name=phParams[i],
                        format=str(int(nx * ny)) + "I",
                        dim='(' + str(fill.shape[2]) + "," + str(fill.shape[1]) + ")",
                        unit=phParamUnits[i],
                        array=fill
                    )
                )
        # Case: Parameter is fit for, but it's vmac which is different than any other param.
        # No idea what happens if it's fit with more than one node, but that's a problem for later.
        elif 'vmac' in phParams[i]:
            dummy_arr = np.zeros((len(logTau), nx, ny))
            if 'err' in phParams[i]:
                param = photosphere[phParams[i]][:, 0, -1].reshape(nx, ny)
            else:
                param = photosphere[phParams[i]][:, 0, -1, 0].reshape(nx, ny)
                for x in range(param.shape[0]):
                    for y in range(param.shape[1]):
                        if type(param[x, y]) is np.ndarray:
                            if len(param[x, y]) != 0:
                                dummy_arr[:, x, y] = param[x, y]
                        else:
                            dummy_arr[:, x, y] = param[x, y]
                if translation:
                    dummy_arr = np.flip(np.rot90(dummy_arr, axes=(1, 2)), axis=1)
                columns.append(
                    fits.Column(
                        name=phParams[i],
                        format=str(int(nx * ny)) + "D",
                        dim='(' + str(dummy_arr.shape[2]) + "," + str(dummy_arr.shape[1]) + ")",
                        unit=phParamUnits[i],
                        array=dummy_arr
                    )
                )
        # Case: Fit for, not vmac, but only one node. Param should be fine to cast, but err needs padded out.
        elif nodeCount == 1:
            if "err" in phParams[i]:
                dummy_err = np.zeros((len(logTau), nx, ny))
                err = photosphere[phParams[i]][:, 0, -1].reshape(nx, ny)
                for x in range(err.shape[0]):
                    for y in range(err.shape[1]):
                        if len(err[x, y]) != 0:
                            dummy_err[:, x, y] = err[x, y]
                if translation:
                    dummy_err = np.flip(np.rot90(dummy_err, axes=(1, 2)), axis=1)
                columns.append(
                    fits.Column(
                        name=phParams[i],
                        format=str(int(nx * ny)) + "D",
                        dim='(' + str(dummy_err.shape[2]) + "," + str(dummy_err.shape[1]) + ")",
                        unit=phParamUnits[i],
                        array=dummy_err
                    )
                )
            else:
                colarr = photosphere[phParams[i]][:, 0, -1, :].reshape(nx, ny, len(logTau))
                if translation:
                    colarr = np.flipud(np.rot90(colarr))
                colarr = np.transpose(colarr, (2, 0, 1))
                columns.append(
                    fits.Column(
                        name=phParams[i],
                        format=str(int(nx * ny)) + "D",
                        dim='(' + str(colarr.shape[2]) + "," + str(colarr.shape[1]) + ")",
                        unit=phParamUnits[i],
                        array=colarr
                    )
                )
        # Case: Fit for, multiple nodes. Param cast as normal, err interpolated onto tau grid.
        else:
            if "err" in phParams[i]:
                nodeList = nodeArrFull[0, 0, :]
                while len(nodeList[nodeList > 0]) == 0:
                    nodeList = nodeArr[
                        np.random.randint(0, nx),
                        np.random.randint(0, ny)
                    ]
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
                if translation:
                    dummy_err = np.flip(np.rot90(dummy_err, axes=(1, 2)), axis=1)
                columns.append(
                    fits.Column(
                        name=phParams[i],
                        format=str(int(nx * ny)) + "D",
                        dim='(' + str(dummy_err.shape[2]) + "," + str(dummy_err.shape[1]) + ")",
                        unit=phParamUnits[i],
                        array=dummy_err
                    )
                )
            else:
                colarr = photosphere[phParams[i]][:, 0, -1, :].reshape(nx, ny, len(logTau))
                if translation:
                    colarr = np.flipud(np.rot90(colarr))
                colarr = np.transpose(colarr, (2, 0, 1))
                columns.append(
                    fits.Column(
                        name=phParams[i],
                        format=str(int(nx * ny)) + "D",
                        dim='(' + str(colarr.shape[2]) + "," + str(colarr.shape[1]) + ")",
                        unit=phParamUnits[i],
                        array=colarr
                    )
                )
    ext = fits.BinTableHDU.from_columns(columns)
    ext.header['EXTNAME'] = ('PHOTOSPHERE', 'Fit photospheric parameters from SIR Inversion (through Hazel)')
    ext.header['LINE'] = ('Si-I', 'Si I 10827 [AA]')

    fits.append(saveName, ext.data, ext.header)

    # After an eternity, we can finally move on to doing the extensions that have data in them.
    # First we do the IQUV for the pre-fit data. Then the synthetic profiles.

    stks = ['I', 'Q', 'U', 'V']

    for i in range(4):
        realStokes = initFile['stokes'][:, :, i]
        realStokes = realStokes.reshape(nx, ny, realStokes.shape[1])
        if translation:
            realStokes = np.flipud(np.rot90(realStokes))
        ext = fits.ImageHDU(realStokes)
        ext.header['EXTNAME'] = ('Stokes-' + stks[i] + "/Ic", "Normalized by Quiet Sun, Corrected for position angle")
        ext.header['CDELT1'] = (dx, 'arcsec')
        ext.header['CDELT2'] = (dy, 'arcsec')
        ext.header['CDELT3'] = fitsFile[1].header['CDELT3']
        ext.header['CTYPE1'] = 'HPLN-TAN'
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
        fits.append(saveName, ext.data, ext.header)

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
        ext.header['CTYPE1'] = 'HPLN-TAN'
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
        fits.append(saveName, ext.data, ext.header)

    # And the chisq map
    chi2 = h5File[sp_key]['chi2'][:, 0, -1].reshape(nx, ny)
    if translation:
        chi2 = np.flipud(np.rot90(chi2))
    ext = fits.ImageHDU(chi2)
    ext.header['EXTNAME'] = ("CHISQ", 'Fit chi-square from Hazel Inversions')
    ext.header['CDELT1'] = (dx, 'arcsec')
    ext.header['CDELT2'] = (dy, 'arcsec')
    ext.header['CTYPE1'] = 'HPLN-TAN'
    ext.header['CTYPE2'] = 'HPLT-TAN'
    ext.header['CUNIT1'] = 'arcsec'
    ext.header['CUNIT2'] = 'arcsec'
    ext.header['CRVAL1'] = fitsFile[1].header['CRVAL1']
    ext.header['CRVAL2'] = fitsFile[1].header['CRVAL2']
    ext.header['CRPIX1'] = nx / 2
    ext.header['CRPIX2'] = ny / 2
    ext.header['CROTAN'] = fitsFile[1].header['CROTAN']
    fits.append(saveName, ext.data, ext.header)

    # The Wavelength Array...
    ext = fits.ImageHDU(wavelength)
    ext.header['EXTNAME'] = 'lambda-coordinate'
    ext.header['BTYPE'] = 'lambda axis'
    ext.header['BUNIT'] = '[AA]'
    fits.append(saveName, ext.data, ext.header)

    # And the time array (only if the full X-range is used.)
    if nx == fitsFile[1].header['NAXIS3'] - 1:
        fits.append(saveName, fitsFile['time-coordinate'].data, fitsFile['time-coordinate'].header)

    if overviewPlot:
        plotHazelResults(saveName)

    return


def plotHazelResults(fitsFile):
    """Plots Hazel results from a level-2 fits HDUList.

    Parameters:
    -----------
    fitsFile : Astropy FITS HSUList object

    """

    params = {
        "savefig.dpi": 300,
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "figure.titleweight": "bold",
        "axes.titlesize": 14,
        "font.size": 12,
        "legend.fontsize": 12,
        "font.family": "serif",
        "image.origin": "lower"
    }
    plt.rcParams.update(params)

    fitsFile = fits.open(fitsFile)

    chromosphere = fitsFile[1].data

    photosphere_idx = 2
    while "PHOTOSPHERE" not in fitsFile[photosphere_idx].header['EXTNAME']:
        photosphere_idx += 1
    photosphere = fitsFile[photosphere_idx].data

    chisq_idx = -3
    while "CHISQ" not in fitsFile[chisq_idx].header['EXTNAME']:
        chisq_idx += 1
    chisq = fitsFile[chisq_idx].data

    tauZero = list(photosphere['logTau']).index(np.abs(photosphere['logTau']).min())

    plotExts = [0, fitsFile[0].header['FOVY'], 0, fitsFile[0].header['FOVX']]

    photoB = np.sqrt(
        photosphere['Bz'][tauZero, :, :]**2 +
        photosphere['By'][tauZero, :, :]**2 +
        photosphere['Bx'][tauZero, :, :]**2
    )
    chromoB = np.sqrt(
        chromosphere['Bz'][0, :, :]**2 +
        chromosphere['By'][0, :, :]**2 +
        chromosphere['Bx'][0, :, :]**2
    )
    photoT = photosphere['T'][tauZero, :, :]
    chromoBeta = chromosphere['beta'][0, :, :]
    photoV = photosphere['v'][12, :, :] - np.nanmean(photosphere['v'][tauZero, :, :])
    chromoV = chromosphere['v'][0, :, :] - np.nanmean(photosphere['v'][tauZero, :, :])
    chromoTau = chromosphere['tau'][0, :, :]

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(ncols=4, nrows=2, hspace=0.2, wspace=0.4)

    ax_pB = fig.add_subplot(gs[0, 0])
    ax_cB = fig.add_subplot(gs[1, 0])

    ax_pT = fig.add_subplot(gs[0, 1])
    ax_cBet = fig.add_subplot(gs[1, 1])

    ax_pV = fig.add_subplot(gs[0, 2])
    ax_cV = fig.add_subplot(gs[1, 2])

    ax_chisq = fig.add_subplot(gs[0, 3])
    ax_cTau = fig.add_subplot(gs[1, 3])

    # Populating....
    # B First

    pBmap = ax_pB.imshow(
        photoB,
        cmap='Purples',
        extent=plotExts,
        aspect='auto',
        vmin=0,
        vmax=np.nanmean(photoB) + 3*np.nanstd(photoB)
    )
    plt.colorbar(mappable=pBmap, ax=ax_pB)
    ax_pB.set_title("|B| [Gauss]")
    ax_pB.set_ylabel("Extent [arcseconds]")
    ax_pB.set_xticks([])

    cBmap = ax_cB.imshow(
        chromoB,
        cmap='Purples',
        extent=plotExts,
        aspect='auto',
        vmin=0,
        vmax=np.nanmean(photoB) + 2 * np.nanstd(photoB)
    )
    plt.colorbar(mappable=cBmap, ax=ax_cB)
    ax_cB.set_title("|B| [Gauss]")
    ax_cB.set_ylabel("Extent [arcseconds]")
    ax_cB.set_xlabel("Extent [arcseconds]")

    # Now temp/Beta

    pTmap = ax_pT.imshow(
        photoT,
        cmap='inferno',
        extent=plotExts,
        aspect='auto',
        vmin=np.nanmean(photoT) - 3 * np.nanstd(photoT),
        vmax=np.nanmean(photoT) + 3 * np.nanstd(photoT)
    )
    plt.colorbar(mappable=pTmap, ax=ax_pT)
    ax_pT.set_title("T [K]")
    ax_pT.set_xticks([])
    ax_pT.set_yticks([])

    cBetmap = ax_cBet.imshow(
        chromoBeta,
        cmap='inferno',
        extent=plotExts,
        aspect='auto',
        vmin=np.nanmean(chromoBeta) - 3 * np.nanstd(chromoBeta),
        vmax=np.nanmean(chromoBeta) + 3 * np.nanstd(chromoBeta)
    )
    plt.colorbar(mappable=cBetmap, ax=ax_cBet)
    ax_cBet.set_title("Plasma-$\\beta$")
    ax_cBet.set_yticks([])
    ax_cBet.set_xlabel("Extent [arcseconds]")

    # Velocities

    pVmap = ax_pV.imshow(
        photoV,
        cmap='seismic',
        extent=plotExts,
        aspect='auto',
        vmin=np.nanmean(photoV) - 5 * np.nanstd(photoV),
        vmax=np.nanmean(photoV) + 5 * np.nanstd(photoV)
    )
    plt.colorbar(mappable=pVmap, ax=ax_pV)
    ax_pV.set_title("v [km/s]")
    ax_pV.set_xticks([])
    ax_pV.set_yticks([])

    cVmap = ax_cV.imshow(
        chromoV,
        cmap='seismic',
        extent=plotExts,
        aspect='auto',
        vmin=np.nanmean(chromoV) - 5 * np.nanstd(chromoV),
        vmax=np.nanmean(chromoV) + 5 * np.nanstd(chromoV)
    )
    plt.colorbar(mappable=cVmap, ax=ax_cV)
    ax_cV.set_title("v [km/s]")
    ax_cV.set_yticks([])
    ax_cV.set_xlabel("Extent [arcseconds]")

    # Chisq and tau

    chsqmap = ax_chisq.imshow(
        chisq,
        cmap='viridis',
        extent=plotExts,
        aspect='auto',
        vmin=0,
        vmax=5*np.nanmedian(chisq)
    )
    plt.colorbar(mappable=chsqmap, ax=ax_chisq)
    ax_chisq.set_title("$\\chi^2$")
    ax_chisq.set_xticks([])
    ax_chisq.set_yticks([])

    taumap = ax_cTau.imshow(
        chromoTau,
        cmap='cividis',
        extent=plotExts,
        aspect='auto',
        vmin=np.nanmean(chromoTau) - 3 * np.nanstd(chromoTau),
        vmax=np.nanmean(chromoTau) + 3 * np.nanstd(chromoTau)
    )
    plt.colorbar(mappable=taumap, ax=ax_cTau)
    ax_cTau.set_title("$\\tau$")
    ax_cTau.set_yticks([])
    ax_cTau.set_xlabel("Extent [arcseconds]")

    fig.suptitle("SIR Photospheric ($\\tau=1$) + Hazel Chromospheric Inversion Summary")
    fig.text(
        0.05, 0.7,
        "Photospheric Fit\nParameters [Si I 10825]\n________",
        rotation=90,
        weight='bold',
        ha='center',
        va='center',
        fontsize=16
    )
    fig.text(
        0.05, 0.28,
        "Chromospheric Fit\nParameterss [He I 10830]\n________",
        rotation=90,
        weight='bold',
        ha='center',
        va='center',
        fontsize=16
    )

    dtstr = str(
        np.datetime64(
            fitsFile[0].header['STARTOBS'],
            "s"
        )
    ).replace("-", "").replace(":", "").replace("T", "_")
    savestr = "hazel_inversion_summary_"+dtstr+".png"
    plt.savefig(savestr, bbox_inches='tight')
    return


def hazelPrep(inFile, outPath,
              xRange=None, yRange=None, waveRange=None,
              translation=False,
              stokesLimit=3, binSlits=1, binSpatial=2):
    """NOTE: Requires the Hazel package to be installed!
    Writes initial files for Hazel inversions from a level-1.5 FIRS fits file.
    This includes:
        ~The stokes data cube with associated errors, LOS vectors (for each point), boundary conditions.
        ~The cube of weights (all ones. You can specify weights in the configfile)
        ~The wavelength array used
        ~The reference chromosphere (3d cube; checks coordinate for on/off-limb and sets as appropriate)
        ~The reference photosphere (1d)
    It does NOT include the config file or the inversion python code. Bring them yourself. Freeloader.

    Parameters:
    -----------
    inFile : str
        Path to level-1.5 FIRS data
    outPath : str
        Directory to save output files. There are several output files, so this needs to be a directory.
    xRange : None or list
        Default is None-type to use full xrange. Otherwise, user can provide xRange as [xmin, xmax]
    yRange : None or list
        Default is None-type to use full yrange. Otherwise, user can provide yRange as [ymin, ymax]
    waveRange : None or list
        Default is None-type to use [10824, 10831.5] (captures Si I and He I but not telluric features).
        Otherwise, user can provide waveRange as [wMin, wMax]
    translation: bool
        True to rotate data cube 90 degrees and flip n/s to bring alignment into agreement with CROTAN.
        This is a legacy option for fits files made before this behaviour was fixed in firs-tools
    stokesLimit: float
        Sets all Stokes QUV profiles for Si I and He I that are below this sigma-level to zero to avoid
        fitting noise. Default (for now) is 3. Does the comparison separately for the He I and Si I line.
        If one is above the noise but not the other, only the profile below the limit is set to zero.
        If both are below the noise, the entire slice is set to zero.
    binSlits: int
        Bins that number of slit positions. Sums IQUV before QS normalization.
    binSpatial: int
        Bins that number along the slit. Sums IQUV before QS normalization.
    """
    import hazel

    if not xRange:
        xRange = [0, -1]

    if not yRange:
        yRange = [0, -1]

    if not waveRange:
        waveRange = [10824, 10831.5]

    firs_file = fits.open(inFile)

    xcen = firs_file[0].header['XCEN']
    ycen = firs_file[0].header['YCEN']
    fovx = firs_file[0].header['FOVX']
    fovy = firs_file[0].header['FOVY']
    rot = firs_file[0].header['ROT']
    dx = firs_file[1].header['CDELT1']
    dy = firs_file[1].header['CDELT2']
    if fovx == 0:
        fovx = 0.3
    if fovy == 0:
        fovy = 0.15

    waveidx_lo = _find_nearest(firs_file['lambda-coordinate'].data, waveRange[0])
    waveidx_hi = _find_nearest(firs_file['lambda-coordinate'].data, waveRange[1])

    # Approximate spectral line ranges. Hardcoded for now.
    # Scratch that, probably hardcoded forever.
    # They're not going anywhere.
    # Note that these are the indices within the sliced data.
    siidx_lo = _find_nearest(firs_file['lambda-coordinate'].data, 10826) - waveidx_lo
    siidx_hi = _find_nearest(firs_file['lambda-coordinate'].data, 10828) - waveidx_lo

    heidx_lo = _find_nearest(firs_file['lambda-coordinate'].data, 10828) - waveidx_lo
    heidx_hi = -1

    def binArray(data, axis, binvalue, binfunc):
        """Bins data along axis by value"""
        dims = np.array(data.shape)
        argdims = np.arange(data.ndim)
        argdims[0], argdims[axis] = argdims[axis], argdims[0]
        data = data.transpose(argdims)
        data = [
            binfunc(
                np.take(
                    data,
                    np.arange(int(b*binvalue), int(b*binvalue + binvalue)),
                    0
                ),
                0
            )
            for b in np.arange(dims[axis]//binvalue)
        ]
        data = np.array(data).transpose(argdims)
        return data

    stokes_i = firs_file[1].data[
        xRange[0]:xRange[1],
        yRange[0]:yRange[1],
        waveidx_lo:waveidx_hi
    ]
    if translation:
        stokes_i = np.flip(np.rot90(stokes_i, axes=(0, 1)), axis=0)
    stokes_i = binArray(stokes_i, 0, binSpatial, np.nansum)
    stokes_i = binArray(stokes_i, 1, binSlits, np.nansum)

    stokes_q = firs_file[2].data[
        xRange[0]:xRange[1],
        yRange[0]:yRange[1],
        waveidx_lo:waveidx_hi
    ]
    if translation:
        stokes_q = np.flip(np.rot90(stokes_q, axes=(0, 1)), axis=0)
    stokes_q = stokes_q - np.nanmedian(stokes_q)
    stokes_q = binArray(stokes_q, 0, binSpatial, np.nansum)
    stokes_q = binArray(stokes_q, 1, binSlits, np.nansum)

    stokes_u = firs_file[3].data[
               xRange[0]:xRange[1],
               yRange[0]:yRange[1],
               waveidx_lo:waveidx_hi
               ]
    if translation:
        stokes_u = np.flip(np.rot90(stokes_u, axes=(0, 1)), axis=0)
    stokes_u = stokes_u - np.nanmedian(stokes_u)
    stokes_u = binArray(stokes_u, 0, binSpatial, np.nansum)
    stokes_u = binArray(stokes_u, 1, binSlits, np.nansum)

    stokes_v = firs_file[4].data[
               xRange[0]:xRange[1],
               yRange[0]:yRange[1],
               waveidx_lo:waveidx_hi
               ]
    if translation:
        stokes_v = np.flip(np.rot90(stokes_v, axes=(0, 1)), axis=0)
    stokes_v = stokes_v - np.nanmedian(stokes_v)
    stokes_v = binArray(stokes_v, 0, binSpatial, np.nansum)
    stokes_v = binArray(stokes_v, 1, binSlits, np.nansum)

    # Assuming the FIRS data is oriented correctly at this point, y is the 0th axis.
    # At CROTAN = 0, this is north-south
    xyGrid = np.zeros((2, firs_file[1].data.shape[0], firs_file[1].data.shape[1]))
    if translation:
        xyGrid = np.flip(np.rot90(xyGrid, axes=(1, 2)), axis=1)

    # Remember that x/y are flipped in between fits header and data cube. Y is axis 0, X is axis 1.
    # To get the coordinate grid, we'll first find the coordinates of the "bottom left" of the grid
    # Then we'll build outward, row by column.
    # Bit of irritating trig happening here.
    beta = np.arctan(fovy / fovx) * (180./np.pi)
    delta = (360. - rot) - beta
    halfDiag = 0.5 * np.sqrt(fovx**2 + fovy**2)
    Dy = halfDiag * np.sin(np.pi * delta / 180.)
    Dx = halfDiag * np.cos(np.pi * delta / 180.)
    x0y0 = (xcen - Dx, ycen + Dy)
    xyGrid[:, 0, 0] = x0y0
    for y in range(xyGrid.shape[1]):
        rowX0 = xyGrid[0, 0, 0] - y * dy * np.cos(np.pi/180. * (90 - rot))
        rowY0 = xyGrid[1, 0, 0] + y * dy * np.sin(np.pi/180. * (90 - rot))
        xyGrid[0, y, 0] = rowX0
        xyGrid[1, y, 0] = rowY0
        for x in range(xyGrid.shape[2]):
            colX = xyGrid[0, y, 0] + x * dx * np.sin(np.pi/180. * (90 - rot))
            colY = xyGrid[1, y, 0] + x * dx * np.cos(np.pi/180. * (90 - rot))
            xyGrid[0, y, x] = colX
            xyGrid[1, y, x] = colY

    xyGrid = xyGrid[:, xRange[0]:xRange[1], yRange[0]:yRange[1]]
    xyGrid = binArray(xyGrid, 1, binSpatial, np.nanmean)
    xyGrid = binArray(xyGrid, 2, binSlits, np.nanmean)

    # Now we can assemble our alpha/theta/gamma grids for Hazel
    alpha = 180. * np.arctan(xyGrid[0, :, :] / xyGrid[1, :, :]) / np.pi
    theta = np.arcsin(
        np.sqrt(
            xyGrid[0, :, :]**2 + xyGrid[1, :, :]**2
        ) / 960.
    ) * 180/np.pi
    gamma = 360 - (90 + alpha)
    for x in range(gamma.shape[0]):
        for y in range(gamma.shape[1]):
            if (gamma[x, y] < 0) or (gamma[x, y] > 180):
                gamma[x, y] = 90 - alpha[x, y]
    phi = 0
    mu = np.cos(theta * np.pi / 180.)
    clv_factor = np.zeros(stokes_i.shape)
    for x in range(mu.shape[0]):
        for y in range(mu.shape[1]):
            clv_factor[x, y, :] = hazel.util.i0_allen(10830, mu[x, y]) / hazel.util.i0_allen(10830, 1.0)

    v_noise_slice = firs_file[4].data[xRange[0]:xRange[1], yRange[0]:yRange[1], 0:100]
    if translation:
        v_noise_slice = np.flip(np.rot90(v_noise_slice, axes=(0, 1)), axis=0)
    v_noise_slice = binArray(v_noise_slice, 0, binSpatial, np.nansum)
    v_noise_slice = binArray(v_noise_slice, 1, binSlits, np.nansum)

    u_noise_slice = firs_file[3].data[xRange[0]:xRange[1], yRange[0]:yRange[1], 0:100]
    if translation:
        u_noise_slice = np.flip(np.rot90(u_noise_slice, axes=(0, 1)), axis=0)
    u_noise_slice = binArray(u_noise_slice, 0, binSpatial, np.nansum)
    u_noise_slice = binArray(u_noise_slice, 1, binSlits, np.nansum)

    q_noise_slice = firs_file[2].data[xRange[0]:xRange[1], yRange[0]:yRange[1], 0:100]
    if translation:
        q_noise_slice = np.flip(np.rot90(q_noise_slice, axes=(0, 1)), axis=0)
    q_noise_slice = binArray(q_noise_slice, 0, binSpatial, np.nansum)
    q_noise_slice = binArray(q_noise_slice, 1, binSlits, np.nansum)

    stokes_i_noise_region = firs_file[1].data[xRange[0]:xRange[1], yRange[0]:yRange[1], 0:100]
    if translation:
        stokes_i_noise_region = np.flip(np.rot90(stokes_i_noise_region, axes=(0, 1)), axis=0)
    stokes_i_noise_region = binArray(stokes_i_noise_region, 0, binSpatial, np.nanmean)
    stokes_i_noise_region = binArray(stokes_i_noise_region, 1, binSlits, np.nanmean)

    # Now we'll do the final normalization for Stokes-I, assemble our noise arrays
    # Hazel expects each profile to be normalized by the local quiet-sun intensity.
    # Rather than trying to account for every possible case,
    # we'll normalize slit position by slit position,
    # using the continuum where the continuum is between 60 < I < 90 percent of the max continuum
    norm_cube = np.nanmean(
        firs_file[1].data[
            xRange[0]:xRange[1],
            yRange[0]:yRange[1],
            0:60
        ], axis=-1
    )
    if translation:
        norm_cube = np.flip(np.rot90(norm_cube), axis=0)
    # Slit positions are axis 1.
    # Fudge along slit to 40:-40 to avoid hairlines.
    norm_cube = np.nan_to_num(norm_cube[40:-40, :])
    norm_cube = binArray(norm_cube, 0, binSpatial, np.nansum)
    norm_cube = binArray(norm_cube, 1, binSlits, np.nansum)

    # Find values corresponding to 60th and 90th percentile.
    # Should avoid sunspots and very brightest points.
    pct_vals = np.percentile(norm_cube, [60, 90])
    for i in range(norm_cube.shape[1]):
        cube_slice = norm_cube[:, i]
        if len(cube_slice[(cube_slice >= pct_vals[0]) & (cube_slice <= pct_vals[1])]) < 50:
            normValue = np.nanmean(norm_cube[(norm_cube >= pct_vals[0]) & (norm_cube <= pct_vals[1])])
        else:
            normValue = np.nanmean(cube_slice[(cube_slice >= pct_vals[0]) & (cube_slice <= pct_vals[1])])
        stokes_i[:, i, :] = (stokes_i[:, i, :] / normValue) * clv_factor[:, i, :]
        stokes_i_noise_region[:, i, :] = (stokes_i_noise_region[:, i, :] / normValue) * clv_factor[:, i, :100]
        stokes_q[:, i, :] = stokes_q[:, i, :] * (clv_factor[:, i, :] / normValue)
        q_noise_slice[:, i, :] = q_noise_slice[:, i, :] * (clv_factor[:, i, :100] / normValue)
        stokes_u[:, i, :] = stokes_u[:, i, :] * (clv_factor[:, i, :] / normValue)
        u_noise_slice[:, i, :] = u_noise_slice[:, i, :] * (clv_factor[:, i, :100] / normValue)
        stokes_v[:, i, :] = stokes_v[:, i, :] * (clv_factor[:, i, :] / normValue)
        v_noise_slice[:, i, :] = v_noise_slice[:, i, :] * (clv_factor[:, i, :100] / normValue)

    npix = int(stokes_i.shape[0] * stokes_i.shape[1])
    nlam = stokes_i.shape[2]

    # Next we flatten stokes vectors
    stokes_3d = np.zeros((npix, nlam, 4), dtype=np.float64)
    stokes_3d[:, :, 0] = stokes_i.reshape((npix, nlam))
    stokes_3d[:, :, 1] = stokes_q.reshape((npix, nlam))
    stokes_3d[:, :, 2] = stokes_u.reshape((npix, nlam))
    stokes_3d[:, :, 3] = stokes_v.reshape((npix, nlam))

    stokes_v_noise = np.nanstd(v_noise_slice, axis=-1).reshape(npix)
    stokes_u_noise = np.nanstd(u_noise_slice, axis=-1).reshape(npix)
    stokes_q_noise = np.nanstd(q_noise_slice, axis=-1).reshape(npix)
    stokes_i_noise = np.nanstd(stokes_i_noise_region - 1, axis=-1).reshape(npix)

    # And the Stokes noise
    sigma_3d = np.zeros(stokes_3d.shape)
    los_3d = np.zeros((npix, 3))
    theta = theta.reshape(npix)
    gamma = gamma.reshape(npix)

    boundary_3d = np.zeros(stokes_3d.shape)
    boundary_default = np.array([1.25, 0.0, 0.0, 0.0])

    for i in range(npix):
        sigma_3d[i, :, 0] = stokes_i_noise[i] * np.ones(nlam)
        sigma_3d[i, :, 1] = stokes_q_noise[i] * np.ones(nlam)
        sigma_3d[i, :, 2] = stokes_u_noise[i] * np.ones(nlam)
        sigma_3d[i, :, 3] = stokes_v_noise[i] * np.ones(nlam)

        los_3d[i, :] = np.array([theta[i], phi, gamma[i]])

        boundary_3d[i, :, :] = np.repeat(np.atleast_2d(boundary_default), nlam, axis=0)

    stokes_3d = np.nan_to_num(stokes_3d)
    sigma_3d = np.nan_to_num(sigma_3d)
    sigma_3d[sigma_3d == 0] = np.nanmedian(sigma_3d[sigma_3d != 0])
    los_3d = np.nan_to_num(los_3d, nan=90.0)

    # A bit of quick cleaning... We want to make sure we're not fitting noise profiles.
    # So we'll check each profile for peaks above or below some (user-defined, default 3)
    # multiple of the noise.
    for i in range(npix):
        for j in range(1, 4):
            siStokes = np.abs(stokes_3d[i, siidx_lo:siidx_hi, j])
            heStokes = np.abs(stokes_3d[i, heidx_lo:heidx_hi, j])
            limit = stokesLimit * sigma_3d[i, 0, j]
            # Rather than selecting the max of the stokes profiles,
            # Since we didn't do like, a great job of de-spiking,
            # We'll only take a profile if there are at least 3 pixels above the limit
            # No Q profiles
            if (len(siStokes[siStokes >= limit]) < 2) & (len(heStokes[heStokes >= limit]) < 2):
                stokes_3d[i, :, j] = np.zeros(nlam)
            # No He I Q Profile, but Si I profile
            elif (len(siStokes[siStokes >= limit]) >= 2) & (len(heStokes[heStokes >= limit]) < 2):
                stokes_3d[i, heidx_lo:heidx_hi, j] = 0
            # Rare case, He I Q Profile, but no Si I Profile
            elif (len(siStokes[siStokes >= limit]) < 2) & (len(siStokes[siStokes >= limit]) >= 2):
                stokes_3d[i, siidx_lo:siidx_hi, j] = 0
            # Default case is to just leave it alone, so no else case needed.

    f = h5py.File(os.path.join(outPath, "10830_inversionReady.h5"), mode="w")
    db_stokes = f.create_dataset('stokes', stokes_3d.shape, dtype=np.float64)
    db_stokes[:] = np.nan_to_num(stokes_3d)
    db_sigma = f.create_dataset('sigma', sigma_3d.shape, dtype=np.float64)
    db_sigma[:] = np.nan_to_num(sigma_3d)
    db_los = f.create_dataset('LOS', los_3d.shape, dtype=np.float64)
    db_los[:] = np.nan_to_num(los_3d)
    db_boundary = f.create_dataset('boundary', boundary_3d.shape, dtype=np.float64)
    db_boundary[:] = np.nan_to_num(boundary_3d)
    f.close()

    np.savetxt(os.path.join(outPath, "10830_inversionReady.wavelength"),
               firs_file['lambda-coordinate'].data[waveidx_lo:waveidx_hi],
               header='lambda')

    f = open(os.path.join(outPath, "10830_inversionReady.weights"), "w")
    f.write('# WeightI WeightQ WeightU WeightV\n')
    for i in range(nlam):
        f.write('1.0    1.0    1.0    1.0\n')
    f.close()

    # Reference Atmosphere. Use HSRA photosphere for all points (even offlimb)
    phot = hazel.tools.File_photosphere(mode='multi')
    phot.set_default(n_pixel=npix, default='hsra')
    phot.save(os.path.join(outPath, 'reference_photosphere'))

    # For the chromosphere, we have to blend an On/Off limb chromosphere.
    # To do this, we first find if there are ANY offlimb points.
    # Then we replace all offlimb points with an offlimb chromosphere
    xyGridFlat = xyGrid.reshape(2, npix)

    onOffLimb = np.zeros(npix, dtype=object)
    for i in range(len(onOffLimb)):
        if np.sqrt(xyGridFlat[0, i]**2 + xyGridFlat[1, i]**2) >= 960:
            onOffLimb[i] = 'offlimb'
        else:
            onOffLimb[i] = 'disk'
    chrom = hazel.tools.File_chromosphere(mode='multi')
    chrom.set_default(n_pixel=npix, default='disk')

    if 'offlimb' in onOffLimb:
        chrom_off = hazel.tools.File_chromosphere(mode='multi')
        chrom_off.set_default(n_pixel=1, default='disk')
        for i in range(len(onOffLimb)):
            if onOffLimb[i] == 'offlimb':
                chrom.model['model'][i, :] = chrom_off.model['model'][0, :]
    chrom.save(os.path.join(outPath, 'reference_chromosphere'))
    return
