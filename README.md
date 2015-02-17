# limb-darkening

This repository stores codes to generate limb-darkening coefficients using arbitrary response functions. 

If the raw models from the target model atmosphere are not on the system (in which case, they should be 
downloaded on the "atlas_models/raw_models" folder for ATLAS models and under "phoenix_models/raw_models" 
for PHOENIX models), the code wil automatically download the closest match to the given input stellar 
parameters of each target.

If you use this code for your research, please consider citing Espinoza & Jordan (2015).

USAGE
-----

You must generate a file in the "input_files" directory, with the following input data separated by *TABS* 
(see the "input_files" folder for an example input file):

    Name:         Target name.

    Teff:         Effective temperature of the target star.

    log(g):       Logarithm of the gravity of the star.

    Fe/H:         Metallicity of the star.

    Vturb:        Microturbulent velocity (km/s). If not known, set this to -1.

    RF:           RF stands for the Response Function. Input values can be:

                  KpHiRes  : Kepler HIRES
                  KpLowRes : Kepler LOWRES
                  IRAC1    : Spitzer IRAC1
                  IRAC2    : Spitzer IRAC2
                  WFC3     : Hubble Space Telescope WFC3

                  It can also be a filename, in which case the response function
                  with that filename must be in the "response_functions" folder. e.g.,
                  if RF is set to my_response_function.dat, the file my_response_function.dat
                  must be in the "response_functions" folder, where the first column must be
                  the wavelength *IN ANGSTROMS* and the second column the response.

    FT:           FT stands for Fitting Technique. Input values have to be one of the following:

                  A17:  LDs using ATLAS with all its 17 angles
                  A100: LDs using ATLAS models interpolating 100 mu-points with a cubic spline 
                        (i.e., like Claret & Bloemen, 2011)
                  AS:   LDs using ATLAS with 15 angles for linear, quadratic and three-parameter 
                        laws, bit 17 angles for the non-linear law (i.e., like Sing, 2010)
                  P:    LDs using PHOENIX models (Husser et al., 2013).
                  PS:   LDs using PHOENIX models using the methods of Sing (2010).
                  PQS:  LDs using PHOENIX quasi-spherical models (mu>=0.1 only)
                  P100: LDs using PHOENIX models and interpolating 100 mu-points with cubic 
                        spline (i.e., like Claret & Bloemen, 2011)

                  You can indicate various fitting techniques at the same time per target. For example, 
                  if for one target you want all the methods to be calculated, just put:  A17,A100,AS,P 
                  in that column.

    min_w:        Minimum wavelength of the bin you wish to integrate. If set to -1, all the filter 
                  passband will be integrated.

    max_w:        Maximum wavelength of the bin you wish to integrate. If set to -1, all the filter 
                  passband will be integrated.

OUTPUTS
-------

The code will generate limb-darkening coefficients for the given targets in the 
input file under the "results" folder. See the file in that folder for an example 
output.
