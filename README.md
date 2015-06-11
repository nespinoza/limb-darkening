# limb-darkening

This repository stores codes to generate limb-darkening coefficients using arbitrary response functions. 

This code assumes you don't have the ATLAS and PHOENIX model atmospheres in your system, so it downloads 
them depending on your specific needs. However, if you already have models, you can create a new folder 
called "atlas_models/raw_models" for ATLAS and "phoenix_models/raw_models" for PHOENIX.

If you use this code for your research, please consider citing Espinoza & Jordan (2015; http://arxiv.org/abs/1503.07020).

DEPENDENCIES
------------

This code makes use of three important libraries:

	+ Numpy.
	+ Scipy.
	+ Pyfits.

All of them are open source and can be easily installed in any machine. Furthermore, 
it makes use of wget to download the model atmospheres needed for your specific needs.

USAGE
-----

Before running the code, you must generate a file in the "input_files" directory with 
the following input data separated by *TABS* (see the "input_files" folder for an example input file):

    Name:         Target name.

    Teff:         Effective temperature of the target star.

    log(g):       Logarithm of the gravity of the star.

    M/H:          Metallicity of the star. It is usually safe to assume it is ~[Fe/H].

    Vturb:        Microturbulent velocity (km/s). If not known, set this to -1, which 
                  will define it to be 2 km/s.

    RF:           RF stands for the Response Function. Input values can be:

                  KpHiRes  : Kepler HIRES
                  KpLowRes : Kepler LOWRES
                  IRAC1    : Spitzer IRAC1
                  IRAC2    : Spitzer IRAC2

                  It can also be a filename, in which case the response function
                  with that filename must be in the "response_functions" folder. 
                  e.g., if RF is set to my_response_function.dat, the file 
                  my_response_function.dat must be in the "response_functions" 
                  folder, where the first column must be the wavelength *IN 
                  ANGSTROMS* and the second column the response.

    FT:           FT stands for Fitting Technique. Input values have to be one of 
                  the following:

                  A17:  LDs using ATLAS with all its 17 angles
                  A100: LDs using ATLAS models interpolating 100 mu-points with a 
                        cubic spline (i.e., like Claret & Bloemen, 2011)
                  AS:   LDs using ATLAS with 15 angles for linear, quadratic and 
                        three-parameter laws, bit 17 angles for the non-linear 
                        law (i.e., like Sing, 2010)
                  P:    LDs using PHOENIX models (Husser et al., 2013).
                  PS:   LDs using PHOENIX models using the methods of Sing (2010).
                  PQS:  LDs using PHOENIX quasi-spherical models (mu>=0.1 only)
                  P100: LDs using PHOENIX models and interpolating 100 mu-points 
                        with cubic spline (i.e., like Claret & Bloemen, 2011)


                  All the PHOENIX limb-darkening coefficients are calculated in 
                  order to be comparable to plane-parallel models (see Section 
                  2.2 in Espinoza & Jordan, 2015).

                  You can indicate various fitting techniques at the same time per 
                  target. For example, if for one target you want all the methods 
                  to be calculated, just put:  A17,A100,AS,P in that column.
	
    min_w:        Minimum wavelength of the bin you wish to integrate. If set to 
                  -1, all the filter passband will be integrated.

    max_w:        Maximum wavelength of the bin you wish to integrate. If set to 
                  -1, all the filter passband will be integrated.

After this is done, you can edit the options in the get_lds.py file in order to define with which 
input file you want to run the code, the output filename (that will be stored in the "results" 
folder, see below) and some optional definitions. After all this is done, you simply run:

		python get_lds.py

And the code will calculate the limb-darkening coefficients for the targets defined in the 
input file. If you automatize the creation of the input filename, you can run everything 
directly from terminal by running:

                python get_lds.y -ifile input_filename -ofile output_filename

Where input_filename is the location of your input filename (e.g., /home/myfolder/input_file.dat), 
and output_filename is the name of the file that will be saved under the "results" folder.

OUTPUTS
-------

The code will generate limb-darkening coefficients for the given targets in the 
input file under the "results" folder. Each file has a description for each column; see 
the example files in that folder for an example output.


Frequently Asked Questions (FAQ)
--------------------------------

Q: *Why are you not including my favorite bandpass (e.g., ugriz, UBVRI) in the "Standard" response functions?*

A: This is because the actual response function of your favorite instrument depends not only on the filter being used, 
   but also on the properties of the mirrors, lenses, etc. being used by your favorite instrument. The 
   "Standard" response functions, on the other hand, have all that in consideration in the final response 
   function curves given here. Although some authors have decided to assume some coatings of the mirrors being 
   used and using that provide response functions for some of the most common filters, we have decided not to 
   do this in order to force the users to measure or ask for the real response functions being measured by 
   the different instrument being used, which can dramatically affect the final limb-darkening coefficients.

Q: *Why are the HST bandpasses not in the repository anymore?*

A: Because the updated versions can be obtained via PySynphot (http://ssb.stsci.edu/pysynphot/docs/index.html). 
   Furthermore, those vary with time, so the response function that you should use is the one corresponding to 
   the dates of your observations!
