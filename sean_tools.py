import numpy as np
from scipy.ndimage import uniform_filter,median_filter

def gaussian(x,a0,a1,a2,c):
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
    y = a0*np.exp(- z**2 /2.) + c
    return y

def all_same(x):
    """A function that determines whether an array or list is composed solely of identical elements

    Parameters
    ----------
    x : array-like
        An array of values. We will determine if it is all the same value

    Returns
    -------
    bool
        True is the array is composed of a single, repeated element, False otherwise
    """
    x = list(x)
    if (x.count(x[0]) == len(x)):
        return True
    else:
        return False

def find_nearest(array,value):
    """Determines the index of the closest value in an array to a sepecified other value

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

def find_nearest_coordinate_pair(coord_pair,coord_array):
    """Determines the closest coordinate pair from an array of coordinate pairs

    Parameters
    ----------
    coord_pair : array-like
        An arbitrary set of coordinates. While this was written to find the nearest pair, it should work equally well with an arbitrary set of coordinates, provided the length of coord_pair matches the length of onle ONE axis of coord_array.
    coord_array : array-like
        An arbitrary set of coordinate sets, with shape X, Y, len(coord_pair)

    Returns
    -------
    indices : tuple
        A tuple of size 2, matching the X/Y pair in coord_array nearest coord_pair
    """

    difference_array = coord_array - coord_pair
    
    vector_array = np.zeros(coord_array.shape[:-1])

    for i in range(len(coord_pair)):
        vector_array += difference_array[:,:,i]**2

    vector_array = np.sqrt(vector_array)

    indices = np.where(vector_array == np.amin(vector_array))

    return indices

def FTS_window(wavemin,wavemax,atlas = 'FTS',norm = True,lines = False,atlas_dir = "/home/sean/storage/FTS_atlas"):
    """For a given wavelength range, return the solar reference spectrum within that range.

    Parameters
    ----------
    wavemin : float
        The minimum desired wavelength (in angstroms)
    wavemax : float
        The maximum desired wavelength (in angstroms)
    atlas : str
        Currently accepts "Wallace" and "FTS" (as these are the only two downloaded). Wallace takes the 2011 Wallace updated atlas, FTS takes the base 1984 FTS atlas.
    norm : bool
        If False and atlas is set to "FTS", will return the solar irradiance between wavemin and wavemax. Do not recommend
    lines : bool
        If True, returns additional arrays denoting line centers and names between wavemin and wavemax.
    atlas_dir : str
        Path to the FTS files. This is written for my concatenated FTS atli in single-file formats. As an argument in case someone else ever needs this file.

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

    if wavemin >= wavemax:
        print("Minimum Wavelength is greater than or equal to the Maximum Wavelength. Reverse those, bud.")
        return None
    if (wavemin <= 2960) or (wavemax >= 13000):
        print("Your selected wavelengths are not in FTS atlas bounds. Come back when I bother downloading the IR/UV atlas")
        return None

    if atlas.lower() == "wallace":
        if (wavemax <= 5000.) or (wavemin <= 5000.):
            atlas_angstroms = np.load(atlas_dir + "Wallace2011_290-1000nm_Wavelengths.npy")
            atlas_spectrum = np.load(atlas_dir + "Wallace2011_290-1000nm_Observed.npy")
        else:
            atlas_angstroms = np.load(atlas_dir + "Wallace2011_500-1000nm_Wavelengths.npy")
            atlas_spectrum = np.load(atlas_dir + "Wallace2011_500-1000nm_Corrected.npy")
    else:
        atlas_angstroms = np.load(atlas_dir + "FTS1984_296-1300nm_Wavelengths.npy")
        if not norm:
            print("Using full solar irradiance. I hope you know what you're doing")
            atlas_spectrum = np.load(atlas_dir + "FTS1984_296-1300nm_Irradiance.npy")
        if norm:
            atlas_spectrum = np.load(atlas_dir + "FTS1984_296-1300nm_Atlas.npy")
    
    idx_lo = find_nearest(atlas_angstroms,wavemin) - 5
    idx_hi = find_nearest(atlas_angstroms,wavemax) + 5
    #selection = (atlas_angstroms < wavemax) & (atlas_angstroms > wavemin)

    wave = atlas_angstroms[idx_lo:idx_hi]
    spec = atlas_spectrum[idx_lo:idx_hi]
    
    if lines:
        line_centers_full = np.load(atlas_dir + "RevisedMultiplet_Linelist_2950-13200_CentralWavelengths.npy")
        line_names_full = np.load(atlas_dir + "RevisedMultiplet_Linelist_2950-13200_IonNames.npy")
        line_selection = (line_centers_full < wavemax) & (line_centers_full > wavemin)
        line_centers = line_centers_full[line_selection]
        line_names = line_names_full[line_selection]
        return wave,spec,line_centers,line_names
    else:
        return wave,spec

def hs_mean(frame,window_size = 21):
    """Compute the Helmli-Scherer mean for a given image frame. Lower values denote "better" seeing.
    Parameters:
    -----------
    frame : array-like
        The image used for computation of the HS mean
    window_size : int
        Size of the window used in creating the mean filtered image. Passes through to scipy.ndimage.uniform_filter. I believe it can only be odd values, and may be a sequence if you'd like the box to not be a square.
    Returns:
    --------
    hs_mean : a single value describing the level of resolved structure in the image.
    """

    med_filt_frame = uniform_filter(frame,size = window_size)

    FM = med_filt_frame / frame

    idx_FM = (med_filt_frame < frame)

    FM[idx_FM] = frame[idx_FM]/med_filt_frame[idx_FM]

    hs_mean = np.mean(1./FM)

    return hs_mean

def mfgs(image,kernel_size = 3):
    """
    Median-filter gradient similarity metric from Deng et. al. 2015
    Parameters:
    -----------
    image : array-like
        Numpy array with image data
    kernel_size : int
        smoothing window for median filtering

    Returns:
    --------
    mfgs_metric : float
        Image quality assesment
    """

    med_img = median_filter(image,size = kernel_size)
    grad_img = np.gradient(image)

    grad = (np.sum(np.abs(grad_img[0][image != 0])) + np.sum(np.abs(grad_img[1][image != 0])))
    med_grad = np.sum(np.abs(np.gradient(med_img)))

    mfgs_metric = ((2 * med_grad * grad) /
                   (med_grad**2 + grad**2))

    return mfgs_metric
