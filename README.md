# FIRS-Tools

This code constitutes the Level-1 to Level-1.5 and Level-1.5 to Level-2 FIRS pipeline for He 10830 data taken at the DST. At some point, this will be expanded and merged to my fork of SSOSoft.

For now, it remains separate as the code is designed to sit on the end of successful reductions via the existing IDL pipeline. I have been making progress in adapting the low-level functions used in this pipeline (see the HSGPy repo), and should have a working integrated pipeline ready shortly.

For now, the three functions anyone reading should concern themselves with are firs_to_fits, hazelPrep, and repackHazel.

# FIRS_TO_FITS

This is the Level-1 to 1.5 pipeline. It corrects for fringes, prefilter curvature, linear trends of Stokes-QUV with wavelength, performs a wavelength calibration, corrects for polarization crosstalk in the V->Q, U direction (but not vice versa until we upgrade the LCVRS), and packs the fixed file into fits format.

It requires a flat field that has been processed by the IDL pipeline as if it were a science map. This dependency will be removed when the full version is integrated into SSOSoft.

# hazelPrep

This is the first third of the Level-2 pipeline. This creates the files necessary to run the Hazel inversion code on the data. This includes normalization, noise estimation, and selecting the wavelength range for inversion. Note that the code assumed you're inverting both the Si I and He I lines, using the SIR backend to Hazel. If you don't need the photospheric inversions, simply adjust the wavelength range and diregard the model photosphere files. There are also keywords to mask out stokes profiles below a noise threshold. Currently 3-sigma is the default. It also takes care of the Hazel metafiles, such as the LOS vectors, which are calculated at every pixel. This is required to deploy to code on scale, as near the limb, these change FAST. When the code is done, just give it a config file and run Hazel on your favorite supercomputer. 

# repackHazel

The last third of the Level-2 pipeline. This takes your nice Hazel results and repacks them to FITS. There's an extension for each chromosphere you used, one for the photosphere, one for each Stokes-IQUV profile that was fit for AND the inverted profiles (in case you want to check them yourself), the Chi-squared map, and the wavelength array from the fits. It also makes a nice overview plot of your results for quicklook purposes if you let it. I wouldn't publish it in a paper, but for a glance at your fit quality, it's quite nice.
