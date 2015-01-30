# limb-darkening

This repository stores codes to generate limb-darkening coefficients using arbitrary response functions.

USAGE

You must input a file with the following input data separated by TABS (see the 'examples' folder for example input files):

    Name:         Target name.

    Teff:         Effective temperature of the target star.

    log(g):       Logarithm of the gravity of the star.

    Fe/H:         Metallicity of the star.

    Vturb:        Microturbulent velocity (km/s). If not known, set this to -1.

    RF:           RF stands for the Response Function. Input values can be:

                  1: Kepler HIRES
                  2: Kepler LOWRES
                  3: IRAC1
                  4: IRAC2
                  5: WFC3

                  It can also be a filename, in which case the response function
                  with that filename must be in the response_functions folder. e.g.,
                  if RF is set to my_response_function.dat, the file my_response_function.dat
                  must be in the response_functions folders, where the first column must be
                  the wavelength IN ANGSTROMS and the second column the response.

    FT:           FT stands for Fitting Technique. Input values have to be one of the following:

                  A17:  LDs using ATLAS all 17 angles
                  A100: LDs using ATLAS models interpolating 100 mu-points with cubic spline (ala Claret & Bloemen, 2011)
                  AS:   LDs using ATLAS with 15 angles for linear,quad and third, 17 for four-parameter law (ala Sing, 2010)
                  P:    LDs using PHOENIX models (Husser et al., 2013).
                  PS:   LDs using PHOENIX models using the method of Sing 2010.
                  PQS:  LDs using PHOENIX quasi-spherical models (mu>=0.1 only)
                  P100: LDs using PHOENIX models and interpolating 100 mu-points with cubic spline (ala Claret & Bloemen, 2011)
                  You can indicate various fitting techniques at the same time per target. For example, if for one target
                  you want all the methods to be calculated, just put:  A17,A100,AS,P in that column.

    min_w:        Minimum wavelength of the bin you wish to integrate. If set to -1, all the filter passband will be integrated.

    max_w:        Maximum wavelength of the bin you wish to integrate. If set to -1, all the filter passband will be integrated.
